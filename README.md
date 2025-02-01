# mriaug
`pip install mriaug` to use a **3D image library** that is **[~50x faster and simpler](https://github.com/codingfisch/mriaug?tab=readme-ov-file#speed-)** than [`torchio`](https://github.com/fepegar/torchio) by

- **only** using **PyTorch** → full GPU(+autograd) support 🔥
- being tiny: **~200 lines of code** → no room for bugs 🐛
    
while offering **~20 different augmentations** (incl. MRI-specific operations) 🩻

👶 Normal users should use `mriaug` via [`niftiai`](https://github.com/codingfisch/niftiai), a deep learning framework for 3D images, since it
- provides `aug_transforms3d`: A convenient function that compiles all `mriaug`mentations!
- simplifies all the code needed for data loading, training, visualization...check it out [here](https://github.com/codingfisch/niftiai)!

👴 Experienced users can build their own framework upon `mriaug` (use [`niftiai/augment.py`](https://github.com/codingfisch/niftiai/blob/main/niftiai/augment.py) as a cheat sheet)

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
Explore the [gallery](https://github.com/codingfisch/mriaug?tab=readme-ov-file#gallery-) to understand the usage and effect of all ~20 augmentations!

## Speed 💨
The popular libraries `torchio` and [`MONAI`](https://github.com/Project-MONAI/MONAI) (utilizes `torchio`) often use [`ITK`](https://github.com/SimpleITK/SimpleITK) (CPU only) like this

*PyTorch tensor → NumPy array → NiBabel image → ITK operation (C/C++) → NumPy array → PyTorch tensor*

to augment a PyTorch tensor 😬 That's complicated and does not use the (for neural net training needed) GPU 🐌

Instead, `mriaug` directly uses PyTorch (CPU & GPU support) resulting in
- **~50x fewer lines of code**: `torchio`: ~10,000 LOC, `mriaug`: ~200 LOC 🤓
- **~50x speedup** on GPU 🔥 based on the table below (run [`speed.py`](https://github.com/codingfisch/mriaug/blob/main/runall.py) to reproduce) 💨

<details>
  <summary><b>Click here</b>, to see <b>runtimes</b> on a 256³ image in seconds (on AMD 5950X CPU and NVIDIA RTX 3090 GPU)</summary>

| Transformation | `torchio` | `mriaug` on CPU | `mriaug` on GPU | Speedup vs. torchio |
|----------------|-----------|-----------------|-----------------|---------------------|
| Flip           | 0.014     | 0.012           | 0.002           | **7.5x**            |
| Affine         | 0.297     | 0.608           | 0.011           | **27.9x**           |
| Warp           | 0.951     | 0.850           | 0.009           | **103.3x**          |
| Bias Field     | 3.258     | 0.081           | 0.002           | **1813.0x**         |
| Noise          | 0.117     | 0.105           | 0.001           | **230.4x**          |
| Downsample     | 0.282     | 0.013           | 0.000           | **592.3x**          |
| Ghosting       | 0.241     | 0.170           | 0.003           | **78.3x**           |
| Spike          | 0.265     | 0.172           | 0.003           | **88.8x**           |
| Motion         | 0.696     | 0.540           | 0.009           | **78.6x**           |
</details>

## Gallery 🧠

Let's load an example 3D image `x`, show it with [`niftiview`](https://github.com/codingfisch/niftiview) (used to create all images below)

![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/original.png)

define some arguments

```python
size = (160, 196, 160)
zoom = torch.tensor([[-.2, 0, 0]])
rotate = torch.tensor([[0, .1, 0]])
translate = torch.tensor([[0, 0, .2]])
shear = torch.tensor([[0, .05, 0]])
```

and **run all augmentations** (see [`runall.py`](https://github.com/codingfisch/mriaug/blob/main/runall.py)):

### [`flip3d(x, dim=0)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L7)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/flip.png)

### [`dihedral3d(x, k=2)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L11)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/dihedral.png)

### [`crop3d(x, translate, size)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L20)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/crop.png)

### [`zoom3d(x, zoom)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L27)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/zoom.png)

### [`rotate3d(x, rotate)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L33)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/rotate.png)

### [`translate3d(x, translate)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L39)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/translate.png)

### [`shear3d(x, shear)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L45)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/shear.png)

### [`affine3d(x, zoom, rotate, translate, shear)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L51)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/affine.png)

### [`warp3d(x, magnitude=.01)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L58)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/warp.png)

### [`affinewarp3d(x, zoom, rotate, translate, shear, magnitude=.01)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L67)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/affinewarp.png)

### [`bias_field3d(x, intensity=.2)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L79)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/bias_field.png)

### [`contrast(x, lighting=.5)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L84)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/contrast.png)

### [`noise3d(x, intensity=.05)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L88)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/noise.png)

### [`chi_noise3d(x, intensity=.05, dof=3)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L92) set dof=2 for Rician noise
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/chi_noise.png)

### [`downsample3d(x, scale=.25, dim=2)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L97)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/downsample.png)

### [`ghosting3d(x, intensity=.5)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L107)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/ghosting.png)

### [`spike3d(x, intensity=.2)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L116)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/spike.png)

### [`ringing3d(x, intensity=.5)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L126)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/ringing.png)

### [`motion3d(x, intensity=.5)`](https://github.com/codingfisch/mriaug_beta/blob/main/mriaug/core.py#L137)
![](https://raw.githubusercontent.com/codingfisch/mriaug/refs/heads/main/data/motion.png)
