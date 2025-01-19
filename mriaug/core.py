from torch import cat, fft, ones, rand, randn, rot90, einsum, tensor, ones_like, Tensor

from .utils import (get_crop, apply_crop, compose_affine, apply_affine, get_warp_grid, sample,
                    ndim_conform, downsample, get_bias_field, get_identity_grid, modify_k_space)


def flip3d(x: Tensor, dim: int = 0) -> Tensor:
    return x.flip(-3 + dim)


def dihedral3d(x, k):
    if k < 8:
        return rot90(x if k < 4 else rot90(x, 2, (-3, -1)), k % 4, (-2, -1))
    elif k < 16:
        return rot90(rot90(x, 1 if k < 12 else -1, (-3, -1)), k % 4, (-3, -2))
    else:
        return rot90(rot90(x, 1 if k < 20 else -1, (-3, -2)), k % 4, (-3, -1))


def crop3d(x: Tensor, translate: Tensor, size: tuple) -> Tensor:
    assert x.ndim in [3, 4, 5], f'Tensor "x" must be 3D, 4D or 5D, but got: {x.ndim}D'
    assert translate.abs().max() <= 1, f'Translate must be in [-1, 1] but got: {translate}'
    crop = get_crop(x.shape[-3:], translate, size)
    return apply_crop(x, crop, size)


def translate3d(x: Tensor, translate: Tensor, size: tuple = None, mode: str = 'bilinear',
                upsample: (int, float) = 1., pad_mode: str = 'zeros', align_corners: bool = True) -> Tensor:
    affine = compose_affine(translate=translate)
    return apply_affine(x, affine, size, mode, upsample, pad_mode, align_corners)


def rotate3d(x: Tensor, rotate: Tensor, size: tuple = None, mode: str = 'bilinear',
             upsample: (int, float) = 1., pad_mode: str = 'zeros', align_corners: bool = True) -> Tensor:
    affine = compose_affine(rotate=rotate)
    return apply_affine(x, affine, size, mode, upsample, pad_mode, align_corners)


def zoom3d(x: Tensor, zoom: Tensor, size: tuple = None, mode: str = 'bilinear',
           upsample: (int, float) = 1., pad_mode: str = 'zeros', align_corners: bool = True) -> Tensor:
    affine = compose_affine(zoom=zoom)
    return apply_affine(x, affine, size, mode, upsample, pad_mode, align_corners)


def shear3d(x: Tensor, shear: Tensor, size: tuple = None, mode: str = 'bilinear',
            upsample: (int, float) = 1., pad_mode: str = 'zeros', align_corners: bool = True) -> Tensor:
    affine = compose_affine(shear=shear)
    return apply_affine(x, affine, size, mode, upsample, pad_mode, align_corners)


def affine3d(x: Tensor, translate: Tensor = None, rotate: Tensor = None, zoom: Tensor = None, shear: Tensor = None,
             size: tuple = None, mode: str = 'bilinear', upsample: (int, float) = 1., pad_mode: str = 'zeros',
             align_corners: bool = True) -> Tensor:
    affine = compose_affine(translate, rotate, zoom, shear)
    return apply_affine(x, affine, size, mode, upsample, pad_mode, align_corners)


def warp3d(x: Tensor, magnitude: (float, Tensor) = .01, steps: (int, Tensor) = 2, nodes: (int, Tensor) = 2,
           x_randn: Tensor = None, size: tuple = None, mode: str = 'bilinear', upsample: (int, float) = 1.,
           pad_mode: str = 'zeros', device=None) -> Tensor:
    size = x.shape[-3:] if size is None else size
    grid_size = [int(upsample * s) for s in size] if upsample > 1 and mode != 'nearest' else size
    grid = get_warp_grid(magnitude=magnitude, steps=steps, nodes=nodes, x_randn=x_randn, x=x, size=grid_size, device=device)
    return sample(x, grid, size, mode, pad_mode, align_corners=True)


def affinewarp3d(x: Tensor, translate: Tensor = None, rotate: Tensor = None, zoom: Tensor = None, shear: Tensor = None,
                 magnitude: (int, float, Tensor) = .01, steps: (int, Tensor) = 2, nodes: (int, Tensor) = 2,
                 x_randn: Tensor = None, size: tuple = None, mode: str = 'bilinear', upsample: (int, float) = 1.,
                 pad_mode: str = 'zeros', device=None) -> Tensor:
    size = x.shape[-3:] if size is None else size
    grid_size = [int(upsample * s) for s in size] if upsample > 1 and mode != 'nearest' else size
    grid = get_warp_grid(magnitude=magnitude, steps=steps, nodes=nodes, x_randn=x_randn, x=x, size=grid_size, device=device)
    affine = compose_affine(translate, rotate, zoom, shear)
    grid = cat([grid, ones_like(grid[..., :1])], dim=-1)
    grid = einsum('bij,bxyzj->bxyzi', affine, grid)
    return sample(x, grid, size, mode, pad_mode, align_corners=True)


def contrast(x: Tensor, lighting: (int, float, Tensor),
             clip_min: (int, float, Tensor) = 0., clip_max: (int, float, Tensor) = 1.) -> Tensor:
    return (x * (1 + ndim_conform(lighting, x.ndim))).clip(min=clip_min, max=clip_max)


def noise3d(x: Tensor, intensity: (int, float, Tensor) = .1, batch: bool = False) -> Tensor:  # normal, gaussian noise
    return x + ndim_conform(intensity, x.ndim) * randn(x.shape[1:] if batch else x.shape, device=x.device)


def chi_noise3d(x: Tensor, intensity: (float, Tensor) = .1, dof: int = 3, batch: bool = False) -> Tensor:
    noise = ndim_conform(intensity, x.ndim) * randn([*(x.shape[1:] if batch else x.shape), dof], device=x.device)
    return ((x[..., None] + noise) ** 2).mean(-1).sqrt()


def downsample3d(x: Tensor, scale: (int, float, Tensor) = .5, dim: int = None, mode: str = 'nearest') -> Tensor:
    if not isinstance(scale, Tensor) or len(x) == 1:
        return downsample(x, scale.squeeze().tolist() if isinstance(scale, Tensor) else scale, dim, mode)
    else:
        assert len(x) == len(scale), 'Batch size(=.shape[0]) of "x" and "scale" must be equal'
        if dim is not None:
            assert scale.ndim == 1, 'If "dim" is specified, "scale" must be a 1D tensor or a float'
        return cat([downsample(x[i:i+1], s if s.ndim == 0 else s.tolist(), dim, mode) for i, s in enumerate(scale)])


def bias_field3d(x: Tensor, intensity: (int, float, Tensor) = 1., order: int = 4,
                 mode_grid=None, x_randn: Tensor = None) -> Tensor:
    bias = get_bias_field(intensity, order, mode_grid, x_randn, x=x)
    return x * bias


def ghosting3d(x: Tensor, intensity: (int, float, Tensor) = .2, num_ghosts: int = 2, dim: int = 0) -> Tensor:
    assert dim in [0, 1, 2], 'dim must be 0 (sagittal), 1 (coronal) or 2 (axial)'
    gain = ones_like(x)
    factor = ndim_conform(intensity, x.ndim) if isinstance(intensity, Tensor) else intensity
    freqs = [range(1, x.shape[2 + i], num_ghosts) if i == dim else slice(None) for i in range(3)]
    gain[:, :, freqs[0], freqs[1], freqs[2]] = 1 - factor * num_ghosts ** (1 / 3)
    return modify_k_space(x, gain)


def spike3d(x: Tensor, intensity: (int, float, Tensor) = .2, frequencies: Tensor = None) -> Tensor:
    frequencies = rand((len(x), 3)) if frequencies is None else frequencies
    assert len(x) == len(frequencies), 'Batch size(=.shape[0]) of "x" and "freqs" must be equal'
    gain = ones_like(x)
    freqs = (frequencies.cpu() * tensor(x.shape[-3:]) / 10 + 10).int().clip(max=tensor(x.shape[-3:]) - 1)  # .../ 10 + 10 -> freqs are in a sane range
    for i, f in enumerate(freqs):
        gain[i, :, f[0], f[1], f[2]] *= (intensity[i] if isinstance(intensity, Tensor) else intensity) * f.min() * 100  # min(freq) * 100 for a sane magnitude
    return modify_k_space(x, gain)


def ringing3d(x: Tensor, intensity: (int, float, Tensor) = .5, frequency: float = .7, band: float = .05, dim: int = 0) -> Tensor:
    assert dim in [0, 1, 2], 'dim must be 0 (sagittal), 1 (coronal) or 2 (axial)'
    freq = (get_identity_grid(size=(1, *x.shape[-3:]), device=x.device)[0] + 1) / 2
    freq = sum([freq[..., i] ** 2 for i in range(2) if i != dim]).sqrt()
    gain = ones_like(x)
    lowpass = (frequency + band / 2) > freq
    highpass = (frequency - band / 2) < freq
    gain[..., lowpass & highpass] = 1 - 10 * (ndim_conform(intensity, x.ndim - 2) if isinstance(intensity, Tensor) else intensity)
    return modify_k_space(x, gain)


def motion3d(x: Tensor, intensity: (int, float, Tensor) = .4, translate: (float, Tensor) = .01) -> Tensor:
    if not isinstance(translate, Tensor):
        translate = translate * ones((len(x), 3), dtype=x.dtype, device=x.device)
    offset = ndim_conform(intensity, x.ndim) * fft.fftn(translate3d(x, translate=translate, mode='nearest'), s=x.shape[-3:])
    return modify_k_space(x, gain=1 - intensity, offset=offset)
