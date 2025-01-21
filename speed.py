import timeit
THREADS = 'auto'  # '1' or 'auto'
N = 8
OPS = ['flip', 'affine', 'warp', 'bias_field', 'noise', 'downsample', 'ghosting', 'spike', 'motion']

zoom = [-.2, .0, .0]
scales = tuple(1 + z for z in zoom)
rotate = [.0, .1, .0]
degrees = tuple(180 * r / 3.141 for r in rotate)
translate = [.0, .0, .2]
translation = tuple(100 * t for t in translate)

torchio_setups = {op: '' for op in OPS}
torchio_run_args = {op: '' for op in OPS}
torchio_setups['flip'] = 'flip = Flip(axes=0)'
torchio_setups['affine'] = f'affine = Affine(scales={scales}, degrees={degrees}, translation={translation})'
torchio_setups['warp'] = 'warp = ElasticDeformation(control_points=(10 * (torch.rand(7, 7, 7, 3) - .5)).numpy(), max_displacement=(1., 1., 1.))'
torchio_setups['bias_field'] = f'bias_field = BiasField(coefficients=torch.rand(20).tolist(), order=3)'
torchio_setups['noise'] = 'noise = Noise(mean=0., std=.05, seed=0)'
torchio_setups['downsample'] = f'downsample = RandomAnisotropy(axes=2, downsampling=4)'
torchio_setups['ghosting'] = f'ghosting = Ghosting(num_ghosts=2, axis=0, intensity=.5, restore=None)'
torchio_setups['spike'] = f'spike = Spike(spikes_positions=torch.rand(1, 3).numpy(), intensity=.2)'
torchio_setups['motion'] = f'motion = Motion(degrees=[{degrees}], translation=[{translation}], times=[.5], image_interpolation="nearest")'

mriaug_setups = {op: '' for op in OPS}
mriaug_run_args = {op: '' for op in OPS}
mriaug_setups['affine'] = f'zoom, rotate, translate = torch.tensor([{zoom}]), torch.tensor([{rotate}]), torch.tensor([{translate}])'
mriaug_run_args['affine'] = ', zoom=zoom, rotate=rotate, translate=translate'
mriaug_run_args['warp'] = ', magnitude=1.'
mriaug_run_args['bias_field'] = ', intensity=2.'
mriaug_run_args['noise'] = ', intensity=.05'
mriaug_run_args['downsample'] = ', scale=.25, dim=2'
mriaug_run_args['ghosting'] = ', intensity=.5'
mriaug_run_args['spike'] = ', intensity=.2'
mriaug_run_args['motion'] = ', intensity=.5'

speedups = []
thread_boilerplate = lambda x: '' if x == 'auto' else f'import os; os.environ["OMP_NUM_THREADS"] = "{x}"'  # default 4
table = '| Transformation | `torchio` | `mriaug` on CPU | `mriaug` on GPU | Speedup vs. torchio |\n'
table += '|----------------|---------|--------------|--------------|--------------|\n'

for s in [256]:
    for op in OPS:
        torchio_class = torchio_setups[op].split(' = ')[1].split('(')[0]
        torchio_setup = f'''{thread_boilerplate(THREADS)}
import torch
from torchio.transforms import {torchio_class}

{torchio_setups[op]}
x = torch.rand(1, {s}, {s}, {s}).float()
'''
        torchio_run = f'{op}(x{torchio_run_args[op]})'
        torchio_time = timeit.timeit(torchio_run, setup=torchio_setup, number=N) / N

        mriaug_setup = f'''{thread_boilerplate(THREADS)}
import torch
from mriaug import {op}3d

x = torch.rand(1, 1, {s}, {s}, {s}).float()
{mriaug_setups[op]}

'''
        mriaug_run = f'{op}3d(x{mriaug_run_args[op]})'
        mriaug_cpu_time = timeit.timeit(mriaug_run, setup=mriaug_setup, number=N) / N

        mriaug_setup += 'x = x.cuda()\n'
        if op == 'affine':
            mriaug_setup += f'zoom, rotate, translate = zoom.cuda(), rotate.cuda(), translate.cuda()'
        mriaug_run += ';torch.cuda.synchronize()'
        mriaug_gpu_time = timeit.timeit(mriaug_run, setup=mriaug_setup, number=N) / N

        name = 'Bias Field' if op == 'bias_field' else op.capitalize()
        speedups.append(torchio_time / mriaug_gpu_time)
        table += f'| {name} | {torchio_time:.3f} | {mriaug_cpu_time:.3f} | {mriaug_gpu_time:.3f} | **{speedups[-1]:.1f}x** |\n'

print(table)
print(f'Median speedup: {sorted(speedups)[len(OPS) // 2]:.1f}x')
