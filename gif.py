import torch
import mriaug
import subprocess
import nibabel as nib
from niftiview import TEMPLATES, NiftiImage
torch.random.manual_seed(0)


def save_image(t: torch.Tensor, filepath: str, fpath=None):
    nii = NiftiImage(arrays=t.cpu().numpy()[0, 0], affines=None)#img.affine)
    if fpath is not None:
        nii.nics[0].filepath = fpath
    im = nii.get_image(height=296, vrange=(0, .9), fpath=fpath is not None)
    im.save(filepath)


if __name__ == '__main__':
    GIFSKI_PATH = '/home/lfisch/.cargo/bin/gifski'
    FPS, QUALITY = 20, 80
    N, N2 = 10, 5

    img = nib.as_closest_canonical(nib.load(TEMPLATES['ch2']))
    x = img.get_fdata()
    x = x / x.max()
    x = torch.from_numpy(x)[None, None].float()

    zoom = torch.tensor([[-.2, -.2, -.2]])
    rotate = torch.tensor([[0, .1, 0]])
    translate = torch.tensor([[0, 0, .2]])
    shear = torch.tensor([[0, .05, 0]])

    for i in range(N+N2):
        save_image(mriaug.zoom3d(x, min(i+1, N)/N * zoom), f'data/gif/00_zoom_{i:02}.png', fpath='Zoom')  # None if i < N else 'Zoom'
    for i in range(N+N2):
        save_image(mriaug.rotate3d(x, min(i+1, N)/N * rotate), f'data/gif/01_rotate_{i:02}.png', fpath='Rotate')
    for i in range(N+N2):
        save_image(mriaug.translate3d(x, min(i+1, N)/N * translate), f'data/gif/02_translate_{i:02}.png', fpath='Translate')
    for i in range(N+N2):
        save_image(mriaug.shear3d(x, min(i+1, N)/N * shear), f'data/gif/03_shear_{i:02}.png', fpath='Shear')
    warp_k = torch.randn((1, 3, 2, 2, 2))
    warp_k[..., 0, 0, 0] = 0
    for i in range(N+N2):
        save_image(mriaug.warp3d(x, magnitude=min(i+1, N)/N * .01, k=warp_k), f'data/gif/04_warp_{i:02}.png', fpath='Warp')
    bias_k = warp_k.clone()
    for i in range(N+N2):
        save_image(mriaug.bias_field3d(x, intensity=min(i+1, N)/N * .2, k=bias_k), f'data/gif/05_bias_field_{i:02}.png', fpath='Bias Field')
    for i in range(N+N2):
        save_image(mriaug.contrast(x, lighting=min(i+1, N)/N * .5), f'data/gif/06_contrast_{i:02}.png', fpath='Contrast')
    noise = x - mriaug.chi_noise3d(x, intensity=.1, dof=3)
    for i in range(N+N2):
        save_image(x + min(i+1, N)/N * noise, f'data/gif/07_chi_noise_{i:02}.png', fpath='Chi Noise')
    for i in range(N+N2):
        save_image(mriaug.downsample3d(x, scale=max(.25, 1 - .75 * (i+1)/N), dim=2), f'data/gif/08_downsample_{i:02}.png', fpath='Downsample')
    for i in range(N+N2):
        save_image(mriaug.ghosting3d(x, intensity=min(i+1, N)/N * .5), f'data/gif/09_ghosting_{i:02}.png', fpath='Ghosting')
    frequencies = .1 * torch.rand((1, 3)) + .1
    for i in range(N+N2):
        save_image(mriaug.spike3d(x, intensity=min(i+1, N)/N * 1., frequencies=frequencies), f'data/gif/10_spike_{i:02}.png', fpath='Spike')
    for i in range(N+N2):
        save_image(mriaug.ringing3d(x, intensity=min(i+1, N)/N * 1.), f'data/gif/11_ringing_{i:02}.png', fpath='Ringing')
    for i in range(N+N2):
        save_image(mriaug.motion3d(x, intensity=min(i+1, N)/N * .5, translate=min(i+1, N) / N * .05), f'data/gif/12_motion_{i:02}.png')

    subprocess.run(f'{GIFSKI_PATH} --fps {FPS} -Q {QUALITY} -o data/gif/gif.gif data/gif/*.png',
                   stdout=subprocess.PIPE, universal_newlines=True, shell=True)
