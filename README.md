# mriaug
`pip install mriaug` to use a **3D image library** that **outperform alternatives** like [`torchio`](https://github.com/fepegar/torchio) and [`MONAI`](https://github.com/Project-MONAI/MONAI) by

- **only** using **PyTorch** → full autograd and GPU support 🔥
- consisting of only **~300 lines of code** → no room for bugs 🐛
    
while offering **~20 different augmentations** (incl. MRI-specific operations) 🩻

⚠️ For random augmentations (e.g. `random_flip3d` instead of `flip3d`) use `mriaug` either
- via [`niftiai`](https://github.com/codingfisch/niftiai), a deep learning framework for 3D images that contains random `mriaug` augmentations 🧠
- or as a backend to build your own random augmentations (use [`niftiai/augment.py`](https://github.com/codingfisch/niftiai/blob/main/niftiai/augment.py) as a cheat sheet) 💪

## Usage 💡
Let's create a 3D image tensor (with additional batch and channel dimension) and apply `flip3d`
```python
import torch
from mriaug import flip3d

shape = (1, 1, 4, 4, 4)
x = torch.linspace(0, 1, 4**3).view(*shape)
x_flipped = flip3d(x)
print(x[..., 0, 0])  # tensor([[[0.0000, 0.2540, 0.5079, 0.7619]]])
print(x_flipped[..., 0, 0])  # tensor([[[0.7619, 0.5079, 0.2540, 0.0000]]])
```
Explore the following gallery to understand the usage and effect of all ~20 augmentations!

## Gallery 🧠

Let's load an example 3D image `x`, show it with [`niftiview`](https://github.com/codingfisch/niftiview) (used to create all images below)

![](data/original.png)

define some arguments

```python
size = (160, 196, 160)
translate = torch.tensor([[0, 0, .2]])
rotate = torch.tensor([[0, .1, 0]])
zoom = torch.tensor([[-.2, 0, 0]])
shear = torch.tensor([[0, .05, 0]])
```

and **run all augmentations** (see [`runall.py`](https://github.com/codingfisch/mriaug/blob/main/runall.py)):

### [`flip3d(x)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L7)
![](data/flip.png)

### [`dihedral3d(x, k=2)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L12)
![](data/dihedral.png)

### [`crop3d(x, translate, size)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L21)
![](data/crop.png)

### [`translate3d(x, translate)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L35)
![](data/translate.png)

### [`rotate3d(x, rotate)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L41)
![](data/rotate.png)

### [`zoom3d(x, zoom)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L47)
![](data/zoom.png)

### [`shear3d(x, shear)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L53)
![](data/shear.png)

### [`affine3d(x, translate, rotate, zoom, shear)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L59)
![](data/affine.png)

### [`warp3d(x, magnitude=.01)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L66)
![](data/warp.png)

### [`affinewarp3d(x, translate, rotate, zoom, shear, magnitude=.01)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L73)
![](data/affinewarp.png)

### [`contrast(x, lighting=.5)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L88)
![](data/contrast.png)

### [`noise3d(x, intensity=.05)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L92)
![](data/noise.png)

### [`chi_noise3d(x, intensity=.05, dof=3)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L96) set dof=2 for Rician noise
![](data/chi_noise.png)

### [`downsample3d(x, scale=.25, dim=2)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L101)
![](data/downsample.png)

### [`bias_field3d(x, intensity=.1)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L111)
![](data/bias_field.png)

### [`ghosting3d(x, intensity=.5)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L117)
![](data/ghosting.png)

### [`spike3d(x, intensity=.2)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L126)
![](data/spike.png)

### [`ringing3d(x, intensity=.5)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L138)
![](data/ringing.png)

### [`motion3d(x, intensity=.5)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L149)
![](data/motion.png)
