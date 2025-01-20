from numpy import quantile
from torch import cat, cos, eye, fft, sin, ones, randn, stack, zeros, arange, tensor, zeros_like, cartesian_prod, Tensor
from torch.nn.functional import affine_grid, interpolate, grid_sample
from collections.abc import Iterable
from resize_right import resize
from resize_right.interp_methods import lanczos3


def get_crop(org_size: tuple, translate: Tensor, size: tuple) -> Tensor:
    assert (tensor(org_size) - tensor(size)).min() >= 0, f'Crop size {size} too large for tensor with size {org_size}'
    return ((translate.flip(-1) + 1) * (tensor(org_size) - tensor(size)) / 2).round().int()


def apply_crop(x: Tensor, crop: Tensor, size: tuple = None) -> Tensor:
    if all([is_scalar(crop[:, i], strict=False) for i in range(crop.shape[1])]):
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


def compose_affine(translate: Tensor = None, rotate: Tensor = None,
                   zoom: Tensor = None, shear: Tensor = None) -> Tensor:
    not_none_arg = [arg for arg in [translate, rotate, zoom, shear] if arg is not None][0]
    translate = zeros_like(not_none_arg) if translate is None else translate
    rotate = zeros_like(not_none_arg) if rotate is None else rotate
    zoom = zeros_like(not_none_arg) if zoom is None else zoom
    shear = zeros_like(not_none_arg) if shear is None else shear
    for v in [translate, rotate, zoom, shear]:
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
    return sample(x, grid, size, mode, pad_mode, align_corners)


def sample(x: Tensor, grid: Tensor, size: tuple = None, mode: str = 'bilinear',
                pad_mode: str = 'zeros', align_corners: bool = True) -> Tensor:
    kwargs = {'mode': mode, 'padding_mode': pad_mode, 'align_corners': align_corners}
    if x.shape[-3:] != grid.shape[-4:-1]:
        resize_kwargs = {'interp_method': lanczos3, 'by_convs': True}
        up_kwargs = {'scale_factors': [1, 1, *[grid.shape[i] / x.shape[i+1] for i in range(-4, 0)]], **resize_kwargs}
        down_kwargs = {'out_shape': [*x.shape[:2], *(x.shape[-3:] if size is None else size)], 'antialiasing': False, **resize_kwargs}
        return x.__class__(resize(grid_sample(resize(x, **up_kwargs), grid, **kwargs), **down_kwargs))
    else:
        return grid_sample(x, grid, **kwargs)


def get_warp_grid(magnitude: (float, Tensor) = .01, steps: (int, Tensor) = 2, nodes: (int, Tensor) = 2,
                  x_randn: Tensor = None, x: Tensor = None, size: tuple = None, device=None) -> Tensor:
    if x is not None:
        assert x.ndim == 5, f'Tensor "x" must be 5D, but got: {x.ndim}D'
    v = small_displacement3d(magnitude=magnitude, nodes=nodes, x_randn=x_randn, x=x, size=size, device=device)
    grid = get_identity_grid(x, size, device)
    disp = shoot_displacement3d(v, steps=steps, grid=grid[:1])
    return disp + grid


def shoot_displacement3d(v, steps: (int, Tensor) = 2, grid: Tensor = None) -> Tensor:
    grid = get_identity_grid(size=(1, *v.shape[:4])) if grid is None else grid
    #v = v / (2 ** steps)
    if is_scalar(steps, strict=False):
        return shoot(v=v, steps=steps[0] if isinstance(steps, Tensor) else steps, grid=grid)
    return cat([shoot(v=v[i:i + 1], steps=s, grid=grid) for i, s in enumerate(steps)])


def get_identity_grid(x: Tensor = None, size: tuple = None, device=None):
    assert x is not None or size is not None, 'Either "x" or "size" must be provided'
    size, device = get_size_and_device(x, size, device)
    id_affine = eye(4, device=device)[None, :3].repeat(size[0], 1, 1)
    return affine_grid(id_affine, [size[0], 3, *size[-3:]], align_corners=True)  # TODO: size currently must be 4D. Add code to support 3D if bs can be infered!


def small_displacement3d(magnitude: (float, Tensor) = .01, nodes: (int, Tensor) = 2, x_randn: Tensor = None,
                         x: Tensor = None, size: tuple = None, device=None) -> Tensor:
    size, device = get_size_and_device(x, size, device)
    nodes = size[0] * [nodes] if is_scalar(nodes) else nodes
    magnitudes = size[0] * [magnitude] if is_scalar(magnitude) else magnitude
    displacements, n = [], 0
    for node, mag in zip(nodes, magnitudes):
        node_shape = [3] + 3 * [max(node, 2)]
        dn = tensor(node_shape).prod().item()
        x_rand = randn(*node_shape, device=device) if x_randn is None else x_randn[n:n + dn].view(*node_shape).to(device)
        disp = resize(mag * x_rand, out_shape=[3, *size[-3:]], antialiasing=False, pad_mode='replicate',
                      interp_method=lanczos3, by_convs=True, max_numerator=1000)
        # hardcoded: lanczos3 (=best quality), by_convs=True (=fastest), max_numerator=1000 (resize up to 1:1000)
        displacements.append(disp.to(x.dtype) if x is not None else disp)
        n += dn
    return stack(displacements).permute(0, 2, 3, 4, 1)


def shoot(v: Tensor, steps: int = 2, grid: Tensor = None):
    v = v.permute(0, 4, 1, 2, 3)
    grid = get_identity_grid(v) if grid is None else grid
    for _ in range(steps):
        v = v + grid_sample(v, grid + v.permute(0, 2, 3, 4, 1), align_corners=True, padding_mode='reflection')
    return v.permute(0, 2, 3, 4, 1)


def downsample(x: Tensor, scale: (float, list) = .5, dim: int = None, mode: str = 'nearest') -> Tensor:
    if dim is not None and is_scalar(scale):
        assert 0 <= dim < 3, f'Dimension must be 0, 1 or 2, but got: {dim}'
        scale = [(scale if isinstance(scale, float) else scale[0]) if i == dim else 1 for i in range(3)]
    return interpolate(interpolate(x, scale_factor=scale, mode='nearest'), x.shape[-3:], mode=mode)


def get_bias_field(intensity: (float, Tensor) = 1., order: int = 4, mode_grid=None, x_randn: Tensor = None,
                   x: Tensor = None, size: (list, tuple) = None, device=None) -> Tensor:
    size, device = get_size_and_device(x, size, device)
    grid = get_mode_grid(order=order, x=x, size=size, device=device) if mode_grid is None else mode_grid
    if x_randn is None:
        x_randn = randn(grid.shape[:2], device=device)  # 2 * torch.rand(grid.shape[:2], device=device) - 1
    bias = grid * x_randn[:, :grid.shape[1]].clip(min=-1, max=1)[:, :, None, None, None]
    bias = bias.mean(dim=1, keepdim=True)
    bias = bias - bias.mean(dim=(2, 3, 4), keepdim=True)
    return (1 + bias) ** (100 * (ndim_conform(intensity, bias.ndim) if isinstance(intensity, Tensor) else intensity))


def get_mode_grid(order: int, x: Tensor = None, size: (list, tuple) = None, device=None):
    size, device = get_size_and_device(x, size, device)
    modes = 3 * [arange(order, device=device)]
    modes = cartesian_prod(*tuple(modes))[1:]
    modes = modes[modes.sum(1) < order]  # filter out based on cumulative order (e.g. x³ + y² -> 5th cumulative order)
    grid = get_identity_grid(size=[len(modes), *size[-3:]], device=device)
    grid = grid ** modes[:, None, None, None]  # polynomial basis functions
    #grid = cos or exp(1.0j * torch.pi * grid ** modes[:, None, None, None]).real  # fourier basis functions
    return grid.mean(dim=-1)[None].repeat(size[0], 1, 1, 1, 1)  # grid.prod(dim=-1)[None]


def get_num_modes(order: int) -> int:
    return order * (order + 1) * (order + 2) // 6


def get_order(num_modes: int) -> int:
    assert 2 <= int(num_modes ** .45) < 10, 'get_order only correct for 2 <= orders < 10 (>4 is not recommended anyway)'
    return int(num_modes ** .45)


def modify_k_space(x: Tensor, gain: (float, Tensor) = 1., offset: (float, Tensor) = 0.) -> Tensor:
    if isinstance(gain, Tensor):
        assert len(x) == len(gain), f'Tensor "x" and "gain" must have same length, but got: {len(x)}!={len(gain)}'
    if isinstance(offset, Tensor):
        assert len(x) == len(offset), f'Tensor "x" and "offset" must have same length, but got: {len(x)}!={len(offset)}'
    k = fft.fftn(x, s=x.shape[2:])
    k = k * (gain if isinstance(gain, float) else ndim_conform(gain, k.ndim)) + (offset if isinstance(offset, float) else ndim_conform(offset, k.ndim))
    return fft.irfftn(k, s=k.shape[2:]).to(x.dtype)


def ndim_conform(param: (int, float, Tensor), ndim: int) -> Tensor:
    if is_scalar(param):
        return param
    add_dims = (ndim - param.ndim) * [None]
    return param[(slice(None), *add_dims)]


def is_scalar(arg: (int, float, Tensor), strict=True) -> bool:  # always False for ndim > 1
    if isinstance(arg, Iterable):
        if hasattr(arg, 'ndim'):
            if arg.ndim > 1:
                return False
        return len(arg) == 1 or (all(item == arg[0] for item in arg) and not strict)
    return True


def is_zero(arg: (int, float, list, tuple, Tensor)) -> bool:  # always False for ndim > 1
    if isinstance(arg, Iterable):
        if hasattr(arg, 'shape'):
            return all((arg == 0).flatten())
        else:
            return all(item == 0 for item in arg)
    else:
        return arg == 0


def get_size_and_device(x: Tensor = None, size: (list, tuple) = None, device=None):
    assert x is not None or size is not None, 'Either "x" or "size" must be provided'
    if size is None:
        size = x.shape
    elif x is not None and size is not None:
        if x.ndim > len(size):
            size = tuple(list(x.shape[:(x.ndim - len(size))]) + list(size))
    return size, x.device if x is not None else device
