import unittest
from torch import eye, int, cuda, ones, rand, equal, randn, zeros, device, tensor, allclose, linspace
from torch.nn.functional import affine_grid

from mriaug import utils
SHAPES = ((1, 1, 7, 8, 6), (1, 3, 7, 8, 6), (3, 1, 7, 8, 6), (3, 2, 7, 8, 6))


class TestUtils(unittest.TestCase):
    def test_apply_crop(self):
        for shape in SHAPES:
            x = rand(*shape)
            crop = zeros(shape[0], 3, dtype=int)
            crop[0, 0] = 1
            size = [s - 2 for s in shape[-3:]]
            result = utils.apply_crop(x, crop=crop, size=size)
            self.assertEqual(result.shape, (*shape[:2], *size))

    def test_get_affine(self):
        for bs in (1, 3):
            zero_arg = zeros((bs, 3))
            translate = zero_arg.clone()
            translate[0, 0] = 1
            affine = utils.get_affine(translate=translate, zoom=zero_arg, rotate=zero_arg, shear=zero_arg)
            self.assertTrue(equal(affine[:, :3, 3], translate))
            self.assertTrue(equal(affine[:, :3, :3], eye(3)[None].repeat(bs, 1, 1)))

    def test_rotation_matrix(self):
        for bs in (1, 3):
            radiant = zeros((bs, 3))
            mat3x3 = utils.rotation_matrix(radiant)
            self.assertTrue(equal(mat3x3, eye(3)[None].repeat(bs, 1, 1)))

    def test_shear_matrix(self):
        for bs in (1, 3):
            shear = zeros((bs, 3))
            mat3x3 = utils.rotation_matrix(shear)
            self.assertTrue(equal(mat3x3, eye(3)[None].repeat(bs, 1, 1)))

    def test_apply_affine(self):
        #from niftiview import NiftiImageGrid
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            affine = eye(4)[None, :3].repeat(shape[0], 1, 1)
            size = tuple(s + 2 for s in shape[-3:])
            x_moved = utils.apply_affine(x, affine, size=size)
            self.assertEqual(size, tuple(x_moved.shape[-3:]))
            x_moved_linear = utils.apply_affine(x, affine)
            x_moved_nearest = utils.apply_affine(x, affine, mode='nearest')
            self.assertTrue(equal(x, x_moved_nearest))
            #print('Linear: ', (x - x_moved_linear).abs().max(), (x - x_moved_linear).abs().mean())
            self.assertTrue(allclose(x, x_moved_linear))
            x_moved_lanczos = utils.apply_affine(x, affine, upsample=2)
            #print('Lanczos: ', (x - x_moved_lanczos).abs().max(), (x - x_moved_lanczos).abs().mean())
            #NiftiImageGrid(arrays=x[:, 0].numpy()).get_image(vrange=(0, 1)).show()
            #NiftiImageGrid(arrays=x_moved_lanczos[:, 0].numpy()).get_image(vrange=(0, 1)).show()
            self.assertTrue(allclose(x, x_moved_lanczos, rtol=2e-1, atol=1e-2))  # lanczos deviates stronger

    def test_sample(self):
        #from niftiview import NiftiImageGrid
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            affine = eye(4)[None, :3].repeat(shape[0], 1, 1)
            grid = affine_grid(affine, [shape[0], 3, *shape[-3:]])
            grid_large = affine_grid(affine, [shape[0], 3, *[s + 2 for s in shape[-3:]]])
            for pad_mode in ['zeros', 'border', 'reflection']:
                x_moved = utils.sample(x, grid, pad_mode=pad_mode)
                #NiftiImageGrid(arrays=x[:, 0].numpy()).get_image(vrange=(0, 1)).show()
                #NiftiImageGrid(arrays=x_moved[:, 0].numpy()).get_image(vrange=(0, 1)).show()
                #print(f'{pad_mode}: ', (x - x_moved).abs().max(), (x - x_moved).abs().mean())
                self.assertTrue(allclose(x, x_moved, rtol=5e-1, atol=5e-2))
                x_moved_large = utils.sample(x, grid_large, pad_mode=pad_mode)
                self.assertTrue(allclose(x, x_moved_large, rtol=5e-1, atol=5e-2))

    def test_get_warp_grid(self):
        for shape in SHAPES:
            grid = utils.get_warp_grid(magnitude=0, size=shape)
            id_grid = utils.get_identity_grid(size=shape)
            self.assertTrue(equal(grid, id_grid))

    def test_get_identity_grid(self):
        for shape in SHAPES:
            grid = utils.get_identity_grid(size=shape)
            self.assertEqual(shape[-3:], tuple(grid.shape[-4:-1]))
            self.assertEqual(shape[0], grid.shape[0])

    def test_get_bias_field(self):
        k_size = (2, 2, 2)
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            bias = utils.get_bias_field(k_size=k_size, size=x.shape)
            self.assertEqual(x.shape[-3:], bias.shape[-3:])
            self.assertEqual(x.shape[0], bias.shape[0])
            k = randn((*shape[:2], *k_size))
            bias0 = utils.get_bias_field(k=k, size=x.shape)
            self.assertFalse(equal(bias, bias0))
            bias1 = utils.get_bias_field(k=k, size=x.shape)
            self.assertTrue(equal(bias0, bias1))

    def test_downsample(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            x_down = utils.downsample(x, scale=.5)
            self.assertEqual(x_down.shape, x.shape)
            x_down0 = utils.downsample(x, scale=.5, dim=2)
            self.assertEqual(x_down.shape, x.shape)
            x_down1 = utils.downsample(x, scale=[1, 1, .5])
            self.assertTrue(equal(x_down0, x_down1))

    def test_modify_k_space(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            x_modified = utils.modify_k_space(x, gain=1, offset=0)
            self.assertTrue(allclose(x, x_modified, atol=1e-7))

    def test_to_ndim(self):
        x = rand(*SHAPES[0])
        self.assertTrue(utils.to_ndim(0, ndim=x.ndim) == 0)
        self.assertTrue(utils.to_ndim(1, ndim=x.ndim) == 1)
        self.assertTrue(utils.to_ndim(0., ndim=x.ndim) == 0.)
        self.assertTrue(utils.to_ndim(1., ndim=x.ndim) == 1.)
        param = ones(3)
        param = utils.to_ndim(param, ndim=x.ndim)
        self.assertTrue(param.ndim == x.ndim)


if __name__ == '__main__':
    unittest.main()
