from torch import cat, fft, ones, rand, randn, rot90, einsum, tensor, ones_like, Tensor

from .utils import (get_crop, apply_crop, get_affine, apply_affine, get_warp_grid, sample,
                    get_bias_field, to_ndim, downsample, modify_k_space, get_identity_grid)


def flip3d(x: Tensor, dim: int = 0) -> Tensor:
    return x.flip(-3 + dim)


def dihedral3d(x: Tensor, k: int) -> Tensor:
    if (k % 24) < 8:
        return rot90(x if (k % 24) < 4 else rot90(x, 2, (-3, -1)), k % 4, (-2, -1))
    elif (k % 24) < 16:
        return rot90(rot90(x, 1 if (k % 24) < 12 else -1, (-3, -1)), k % 4, (-3, -2))
    else:
        return rot90(rot90(x, 1 if (k % 24) < 20 else -1, (-3, -2)), k % 4, (-3, -1))


def crop3d(x: Tensor, translate: Tensor, size: tuple) -> Tensor:
    assert x.ndim in [3, 4, 5], f'Tensor "x" must be 3D, 4D or 5D, but got: {x.ndim}D'
    assert translate.abs().max() <= 1, f'Translate must be in [-1, 1] but got: {translate}'
    crop = get_crop(x.shape[-3:], translate, size)
    return apply_crop(x, crop, size)


def zoom3d(x: Tensor, zoom: Tensor, size: tuple = None, mode: str = 'bilinear',
           upsample: (int, float) = 1., pad_mode: str = 'zeros', align_corners: bool = True) -> Tensor:
    affine = get_affine(zoom=zoom)
    return apply_affine(x, affine, size, mode=mode, upsample=upsample, pad_mode=pad_mode, align_corners=align_corners)


def rotate3d(x: Tensor, rotate: Tensor, size: tuple = None, mode: str = 'bilinear',
             upsample: (int, float) = 1., pad_mode: str = 'zeros', align_corners: bool = True) -> Tensor:
    affine = get_affine(rotate=rotate)
    return apply_affine(x, affine, size, mode=mode, upsample=upsample, pad_mode=pad_mode, align_corners=align_corners)


def translate3d(x: Tensor, translate: Tensor, size: tuple = None, mode: str = 'bilinear',
                upsample: (int, float) = 1., pad_mode: str = 'zeros', align_corners: bool = True) -> Tensor:
    affine = get_affine(translate=translate)
    return apply_affine(x, affine, size, mode=mode, upsample=upsample, pad_mode=pad_mode, align_corners=align_corners)


def shear3d(x: Tensor, shear: Tensor, size: tuple = None, mode: str = 'bilinear',
            upsample: (int, float) = 1., pad_mode: str = 'zeros', align_corners: bool = True) -> Tensor:
    affine = get_affine(shear=shear)
    return apply_affine(x, affine, size, mode=mode, upsample=upsample, pad_mode=pad_mode, align_corners=align_corners)


def affine3d(x: Tensor, zoom: Tensor = None, rotate: Tensor = None, translate: Tensor = None, shear: Tensor = None,
             size: tuple = None, mode: str = 'bilinear', upsample: (int, float) = 1., pad_mode: str = 'zeros',
             align_corners: bool = True) -> Tensor:
    affine = get_affine(zoom=zoom, rotate=rotate, translate=translate, shear=shear)
    return apply_affine(x, affine, size, mode=mode, upsample=upsample, pad_mode=pad_mode, align_corners=align_corners)


def warp3d(x: Tensor, magnitude: (int, float, Tensor) = .02, k_size: tuple = (2, 2, 2), k: Tensor = None,
           size: tuple = None, mode: str = 'bilinear', upsample: (int, float) = 1., pad_mode: str = 'zeros') -> Tensor:
    size = x.shape[-3:] if size is None else size
    assert len(size) == 3, 'Size must have 3(=spatial) dimensions'
    grid_size = [int(upsample * s) for s in size] if upsample > 1 and mode != 'nearest' else size
    grid = get_warp_grid(magnitude, k_size=k_size, k=k, size=(len(x), *grid_size), device=x.device).to(x.dtype)
    return sample(x, grid, size, mode=mode, pad_mode=pad_mode, align_corners=True)


def affinewarp3d(x: Tensor, zoom: Tensor = None, rotate: Tensor = None, translate: Tensor = None, shear: Tensor = None,
                 magnitude: (int, float, Tensor) = .02, k_size: tuple = (2, 2, 2), k: Tensor = None, size: tuple = None,
                 mode: str = 'bilinear', upsample: (int, float) = 1., pad_mode: str = 'zeros') -> Tensor:
    size = x.shape[-3:] if size is None else size
    grid_size = [int(upsample * s) for s in size] if upsample > 1 and mode != 'nearest' else size
    grid = get_warp_grid(magnitude, k_size=k_size, k=k, size=(len(x), *grid_size), device=x.device).to(x.dtype)
    affine = get_affine(zoom=zoom, rotate=rotate, translate=translate, shear=shear)
    grid = cat([grid, ones_like(grid[..., :1])], dim=-1)
    grid = einsum('bij,bxyzj->bxyzi', affine, grid)
    return sample(x, grid, size, mode=mode, pad_mode=pad_mode, align_corners=True)


def bias_field3d(x: Tensor, intensity: (int, float, Tensor) = .2, k_size: tuple = (2, 2, 2), k: Tensor = None) -> Tensor:
    bias = get_bias_field(intensity, k_size=k_size, k=k, size=x.shape, device=x.device).to(x.dtype)
    return x * bias


def contrast(x: Tensor, lighting: (int, float, Tensor)) -> Tensor:
    return x * (1 + to_ndim(lighting, x.ndim))


def noise3d(x: Tensor, intensity: (int, float, Tensor) = .1, batch: bool = False) -> Tensor:  # normal, gaussian noise
    return x + to_ndim(intensity, x.ndim) * randn(x.shape[1:] if batch else x.shape, device=x.device)


def chi_noise3d(x: Tensor, intensity: (float, Tensor) = .1, dof: int = 3, batch: bool = False) -> Tensor:
    noise = to_ndim(intensity, x.ndim) * randn([*(x.shape[(1 if batch else 0):]), dof], device=x.device)
    return ((x[..., None] + noise) ** 2).mean(-1).sqrt()


def downsample3d(x: Tensor, scale: (int, float, Tensor) = .5, dim: int = None, mode: str = 'nearest') -> Tensor:
    if not isinstance(scale, Tensor) or len(x) == 1:
        return downsample(x, scale.squeeze().tolist() if isinstance(scale, Tensor) else scale, dim=dim, mode=mode)
    else:
        assert len(x) == len(scale), 'Batch size(=.shape[0]) of "x" and "scale" must be equal'
        assert scale.ndim == 1, '"scale" must be a 1D tensor or a float'
        return cat([downsample(x[i:i+1], s.item(), dim=dim, mode=mode) for i, s in enumerate(scale)])


def ghosting3d(x: Tensor, intensity: (int, float, Tensor) = .2, n_ghosts: int = 2, dim: int = 0) -> Tensor:
    assert dim in [0, 1, 2], 'dim must be 0 (sagittal), 1 (coronal) or 2 (axial)'
    gain = ones_like(x)
    factor = to_ndim(intensity, x.ndim) if isinstance(intensity, Tensor) else intensity
    freqs = [range(1, x.shape[-3 + i], n_ghosts) if i == dim else slice(None) for i in range(3)]
    gain[..., freqs[0], freqs[1], freqs[2]] = 1 - factor * n_ghosts ** (1 / 3)
    return modify_k_space(x, gain=gain)


def spike3d(x: Tensor, intensity: (int, float, Tensor) = .2, frequencies: Tensor = None) -> Tensor:
    frequencies = .1 * rand((len(x), 3)) + .1 if frequencies is None else frequencies
    assert len(x) == len(frequencies), 'Batch size(=.shape[0]) of "x" and "freqs" must be equal'
    gain = ones_like(x)
    freqs = (frequencies.cpu() * tensor(x.shape[-3:])).int()
    for i, f in enumerate(freqs):
        gain[i, ..., f[0], f[1], f[2]] = 1 + (intensity[i] if isinstance(intensity, Tensor) else intensity) * f.min() * 100
    return modify_k_space(x, gain=gain)


def ringing3d(x: Tensor, intensity: (int, float, Tensor) = .5, frequency: float = .7, dim: int = 0) -> Tensor:
    assert dim in [0, 1, 2], 'dim must be 0 (sagittal), 1 (coronal) or 2 (axial)'
    freq = (get_identity_grid(size=(1, *x.shape[-3:]), device=x.device)[0] + 1) / 2
    freq = sum([freq[..., i] ** 2 for i in range(2) if i != dim]).sqrt()
    gain = ones_like(x)
    lowpass = (frequency * 1.05) > freq
    highpass = (frequency * .95) < freq
    gain[..., lowpass & highpass] = 1 - 10 * to_ndim(intensity, x.ndim - 2)
    return modify_k_space(x, gain=gain)


def motion3d(x: Tensor, intensity: (int, float, Tensor) = .4, translate: (float, Tensor) = .02) -> Tensor:
    if not isinstance(translate, Tensor):
        translate = translate * ones((len(x), 3), dtype=x.dtype, device=x.device)
    offset = to_ndim(intensity, x.ndim) * fft.fftn(translate3d(x, translate=translate, mode='nearest'), s=x.shape[-3:])
    return modify_k_space(x, gain=1 - intensity, offset=offset)
