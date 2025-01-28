import torch
import mriaug
import nibabel as nib
from niftiview import TEMPLATES, NiftiImage
torch.random.manual_seed(0)


def save_image(t: torch.Tensor, filepath: str):
    nii = NiftiImage(arrays=t.cpu().numpy()[0, 0], affines=img.affine)
    im = nii.get_image(height=296, vrange=(0, .9))
    im.save(filepath)


if __name__ == '__main__':
    img = nib.as_closest_canonical(nib.load(TEMPLATES['ch2']))
    x = img.get_fdata()
    x = x / x.max()
    x = torch.from_numpy(x)[None, None].float()
    save_image(x, 'data/original.png')

    size = (160, 196, 160)
    zoom = torch.tensor([[-.2, 0, 0]])
    rotate = torch.tensor([[0, .1, 0]])
    translate = torch.tensor([[0, 0, .2]])
    shear = torch.tensor([[0, .05, 0]])

    save_image(mriaug.flip3d(x, dim=0), 'data/flip.png')
    save_image(mriaug.dihedral3d(x, k=2), 'data/dihedral.png')
    save_image(mriaug.crop3d(x, translate, size), 'data/crop.png')
    save_image(mriaug.zoom3d(x, zoom), 'data/zoom.png')
    save_image(mriaug.rotate3d(x, rotate), 'data/rotate.png')
    save_image(mriaug.translate3d(x, translate), 'data/translate.png')
    save_image(mriaug.shear3d(x, shear), 'data/shear.png')
    save_image(mriaug.affine3d(x, translate, rotate, zoom, shear), 'data/affine.png')
    save_image(mriaug.warp3d(x, magnitude=.01), 'data/warp.png')
    save_image(mriaug.affinewarp3d(x, zoom, rotate, translate, shear, magnitude=.01), 'data/affinewarp.png')
    save_image(mriaug.bias_field3d(x, intensity=.2), 'data/bias_field.png')
    save_image(mriaug.contrast(x, lighting=.5), 'data/contrast.png')
    save_image(mriaug.noise3d(x, intensity=.05), 'data/noise.png')
    save_image(mriaug.chi_noise3d(x, intensity=.05, dof=3), 'data/chi_noise.png')
    save_image(mriaug.downsample3d(x, scale=.25, dim=2), 'data/downsample.png')
    save_image(mriaug.ghosting3d(x, intensity=.5), 'data/ghosting.png')
    save_image(mriaug.spike3d(x, intensity=.2), 'data/spike.png')
    save_image(mriaug.ringing3d(x, intensity=.5), 'data/ringing.png')
    save_image(mriaug.motion3d(x, intensity=.5), 'data/motion.png')
