from torch import cat, cos, eye, fft, sin, ones, equal, randn, stack, zeros, tensor, zeros_like, Tensor
from torch.nn.functional import affine_grid, interpolate, grid_sample
from resize_right import resize, interp_methods
RESIZE_KWARGS = {'interp_method': interp_methods.lanczos3, 'by_convs': True, 'max_numerator': 1000}


def get_crop(org_size: tuple, translate: Tensor, size: tuple) -> Tensor:
    assert (tensor(org_size) - tensor(size)).min() >= 0, f'Crop size {size} too large for tensor with size {org_size}'
    return ((translate.flip(-1) + 1) * (tensor(org_size) - tensor(size)) / 2).round().int()


def apply_crop(x: Tensor, crop: Tensor, size: tuple = None) -> Tensor:
    if all(equal(c, crop[0]) for c in crop):
        slices = get_crop_slices(crop[0], size, x.shape[-3:])
        return x[..., slices[0], slices[1], slices[2]]
    else:
        x_list = []
        for i in range(len(x)):
            slices = get_crop_slices(crop[i], size, x.shape[-3:])
            x_list.append(x[i, ..., slices[0], slices[1], slices[2]])
        return stack(x_list)


def get_crop_slices(crop, size, shape):
    return [slice(max(0, c), min(c + s, os)) for c, s, os in zip(crop, size, shape)]


def get_affine(zoom: Tensor = None, rotate: Tensor = None, translate: Tensor = None, shear: Tensor = None) -> Tensor:
    not_none_arg = [arg for arg in [zoom, rotate, translate, shear] if arg is not None][0]
    zoom = zeros_like(not_none_arg) if zoom is None else zoom
    rotate = zeros_like(not_none_arg) if rotate is None else rotate
    translate = zeros_like(not_none_arg) if translate is None else translate
    shear = zeros_like(not_none_arg) if shear is None else shear
    for v in [zoom, rotate, translate, shear]:
        check_vector_shape(v)
    mat3x3 = (rotation_matrix(rotate) * (1 + zoom[:, None])) @ shear_matrix(shear)
    return cat([mat3x3, translate[..., None]], dim=-1)


def rotation_matrix(radiant: Tensor) -> Tensor:
    check_vector_shape(radiant)
    cos_r, sin_r = cos(radiant).T, sin(radiant).T
    zero = zeros(len(radiant), dtype=radiant.dtype, device=radiant.device)
    r = [[1 + zero, zero, zero, zero, cos_r[0], -sin_r[0], zero, sin_r[0], cos_r[0]],  # x-axis rotation
         [cos_r[1], zero, sin_r[1], zero, 1 + zero, zero, -sin_r[1], zero, cos_r[1]],  # y-axis rotation
         [cos_r[2], -sin_r[2], zero, sin_r[2], cos_r[2], zero, zero, zero, 1 + zero]]  # z-axis rotation
    r = [stack(rr).T.view(-1, 3, 3) for rr in r]
    return (r[0] @ r[1]) @ r[2]


def shear_matrix(shear: Tensor) -> Tensor:
    check_vector_shape(shear)
    one = ones(len(shear), dtype=shear.dtype, device=shear.device)
    return stack([one, shear[:, 0], shear[:, 1],
                  shear[:, 0], one, shear[:, 2],
                  shear[:, 1], shear[:, 2], one]).view(3, 3, -1).permute(2, 0, 1)


def check_vector_shape(v):
    assert v.ndim == 2, f'Tensor "v" must be 2D, but got: {v.ndim}D'
    assert v.shape[-1] == 3, f'Tensor "v" have length of 3 in the last dim., but got: {v.shape[-1]}'


def apply_affine(x: Tensor, affine: Tensor, size: tuple = None, mode: str = 'bilinear', upsample: float = 1.,
                 pad_mode: str = 'zeros', align_corners: bool = True) -> Tensor:
    size = x.shape[-3:] if size is None else size
    grid_size = [int(upsample * s) for s in size] if upsample > 1 and mode != 'nearest' else size
    grid = affine_grid(affine, [len(x), 3, *grid_size], align_corners=align_corners)
    return sample(x, grid=grid, size=size, mode=mode, pad_mode=pad_mode, align_corners=align_corners)


def sample(x: Tensor, grid: Tensor, size: tuple = None, mode: str = 'bilinear',
           pad_mode: str = 'zeros', align_corners: bool = True) -> Tensor:
    kwargs = {'mode': mode, 'padding_mode': pad_mode, 'align_corners': align_corners}
    if x.shape[-3:] != grid.shape[-4:-1]:
        up_kwargs = {'scale_factors': [1, 1, *[grid.shape[i] / x.shape[i+1] for i in range(-4, 0)]], **RESIZE_KWARGS}
        down_kwargs = {'out_shape': [*x.shape[:2], *(x.shape[-3:] if size is None else size)],
                       'antialiasing': False, **RESIZE_KWARGS}
        return x.__class__(resize(grid_sample(resize(x, **up_kwargs), grid, **kwargs), **down_kwargs))
    else:
        return grid_sample(x, grid, **kwargs)


def get_warp_grid(magnitude: (float, Tensor) = .02, k_size: tuple = (2, 2, 2),
                  k: Tensor = None, size: tuple = None, device=None) -> Tensor:
    grid = get_identity_grid(size, device)
    if k is None:
        k = randn((size[0], 3, *k_size), device=device)
        k[..., 0, 0, 0] = 0
    disp = fft.fftn(to_ndim(magnitude, k.ndim) * k, s=size[-3:]).real
    return grid + disp.permute(0, 2, 3, 4, 1)


def get_identity_grid(size: tuple, device=None) -> Tensor:
    affine = eye(4, device=device)[None, :3].repeat(size[0], 1, 1)
    return affine_grid(affine, [size[0], 3, *size[-3:]], align_corners=True)


def get_bias_field(intensity: (float, Tensor) = .2, k_size: tuple = (2, 2, 2),
                   k: Tensor = None, size: tuple = None, device=None) -> Tensor:
    if k is None:
        k = randn((*size[:-3], *k_size), device=device)
        k[..., 0, 0, 0] = 0
    bias = fft.fftn(to_ndim(intensity, k.ndim) * k, s=size[-3:]).real
    return 1 + bias


def downsample(x: Tensor, scale: (int, float, list, tuple) = .5, dim: int = None, mode: str = 'nearest') -> Tensor:
    if dim is not None and isinstance(scale, (int, float)):
        assert 0 <= dim < 3, f'Dimension must be 0, 1 or 2, but got: {dim}'
        scale = [(scale if isinstance(scale, (int, float)) else scale[0]) if i == dim else 1 for i in range(3)]
    return interpolate(interpolate(x, scale_factor=scale, mode='nearest'), size=x.shape[-3:], mode=mode)


def modify_k_space(x: Tensor, gain: (float, Tensor) = 1., offset: (float, Tensor) = 0.) -> Tensor:
    if isinstance(gain, Tensor):
        assert len(x) == len(gain), f'Tensor "x" and "gain" must have same length, but got: {len(x)}!={len(gain)}'
    if isinstance(offset, Tensor):
        assert len(x) == len(offset), f'Tensor "x" and "offset" must have same length, but got: {len(x)}!={len(offset)}'
    k = fft.fftn(x, s=x.shape[-3:])
    k = k * to_ndim(gain, k.ndim) + to_ndim(offset, k.ndim)
    return fft.irfftn(k, s=k.shape[-3:]).to(x.dtype)


def to_ndim(param: (int, float, Tensor), ndim: int) -> Tensor:
    if isinstance(param, (int, float)):
        return param
    add_dims = (ndim - param.ndim) * [None]
    return param[(slice(None), *add_dims)]
