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

    def test_compose_affine(self):
        for bs in (1, 3):
            zero_arg = zeros((bs, 3))
            translate = zero_arg.clone()
            translate[0, 0] = 1
            affine = utils.compose_affine(translate, rotate=zero_arg, zoom=zero_arg, shear=zero_arg)
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
            size = [s + 2 for s in shape[-3:]]
            x_moved = utils.apply_affine(x, affine, size=size)
            self.assertEqual(size, list(x_moved.shape[-3:]))
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

    def test_grid_sample(self):
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

    def test_shoot_displacement3d(self):
        for shape in SHAPES:
            x = rand(*shape)
            grid = utils.get_identity_grid(x)
            self.assertEqual(x.shape[-3:], grid.shape[-4:-1])
            self.assertEqual(x.shape[0], grid.shape[0])
            grid = utils.get_identity_grid(size=shape)
            self.assertEqual(x.shape[-3:], grid.shape[-4:-1])
            self.assertEqual(x.shape[0], grid.shape[0])

    def test_get_identity_grid(self):
        for shape in SHAPES:
            x = rand(*shape)
            grid = utils.get_identity_grid(x)
            self.assertEqual(x.shape[-3:], grid.shape[-4:-1])
            self.assertEqual(x.shape[0], grid.shape[0])
            grid = utils.get_identity_grid(size=shape)
            self.assertEqual(x.shape[-3:], grid.shape[-4:-1])
            self.assertEqual(x.shape[0], grid.shape[0])

    def test_small_displacement3d(self):
        for shape in SHAPES:
            x = rand(*shape)
            v = utils.small_displacement3d(x=x)
            self.assertEqual(x.shape[-3:], v.shape[-4:-1])
            self.assertEqual(x.shape[0], v.shape[0])
            v = utils.small_displacement3d(size=shape)
            self.assertEqual(x.shape[-3:], v.shape[-4:-1])
            self.assertEqual(x.shape[0], v.shape[0])
            x_randn = randn(*shape).view(-1)
            v0 = utils.small_displacement3d(size=shape, x_randn=x_randn)
            self.assertFalse(equal(v0, v))
            v1 = utils.small_displacement3d(size=shape, x_randn=x_randn)
            self.assertTrue(equal(v0, v1))

    def test_shoot(self):
        for shape in SHAPES:
            v = utils.small_displacement3d(size=shape)
            v_shot = utils.shoot(v)
            self.assertEqual(v.shape, v_shot.shape)
            self.assertTrue(v.std() < v_shot.std())
            grid = utils.get_identity_grid(size=shape)
            v_shot = utils.shoot(v, grid=grid)
            self.assertEqual(v.shape, v_shot.shape)
            self.assertTrue(v.std() < v_shot.std())

    def test_downsample(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            x_down = utils.downsample(x, scale=.5)
            self.assertEqual(x_down.shape, x.shape)
            x_down0 = utils.downsample(x, scale=.5, dim=2)
            self.assertEqual(x_down.shape, x.shape)
            x_down1 = utils.downsample(x, scale=[1, 1, .5])
            self.assertTrue(equal(x_down0, x_down1))

    def test_get_bias_field(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            bias = utils.get_bias_field(x=x)
            self.assertEqual(x.shape[-3:], bias.shape[-3:])
            self.assertEqual(x.shape[0], bias.shape[0])
            bias = utils.get_bias_field(size=x.shape)
            self.assertEqual(x.shape[-3:], bias.shape[-3:])
            self.assertEqual(x.shape[0], bias.shape[0])
            x_randn = randn(*shape).view(shape[0], -1)
            bias0 = utils.get_bias_field(size=x.shape, x_randn=x_randn)
            self.assertFalse(equal(bias, bias0))
            bias1 = utils.get_bias_field(size=x.shape, x_randn=x_randn)
            self.assertTrue(equal(bias0, bias1))

    def test_get_mode_grid(self):
        for shape in SHAPES:
            grid = utils.get_mode_grid(order=3, size=shape)
            self.assertEqual(tuple(grid.shape[-3:]), shape[-3:])
            self.assertEqual(grid.shape[0], shape[0])

    def test_modify_k_space(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            x_modified = utils.modify_k_space(x, gain=1, offset=0, renorm=False)
            self.assertTrue(allclose(x, x_modified, atol=1e-7))

    def test_ndim_conform(self):
        x = rand(*SHAPES[0])
        self.assertTrue(utils.ndim_conform(0, ndim=x.ndim) == 0)
        self.assertTrue(utils.ndim_conform(1, ndim=x.ndim) == 1)
        self.assertTrue(utils.ndim_conform(0., ndim=x.ndim) == 0.)
        self.assertTrue(utils.ndim_conform(1., ndim=x.ndim) == 1.)
        param = ones(3)
        param = utils.ndim_conform(param, ndim=x.ndim)
        self.assertTrue(param.ndim == x.ndim)

    def test_is_scalar(self):
        self.assertTrue(utils.is_scalar(0))
        self.assertTrue(utils.is_scalar(1))
        self.assertTrue(utils.is_scalar(ones(3), strict=False))
        self.assertFalse(utils.is_scalar(ones((2, 3)), strict=True))

    def test_is_zero(self):
        self.assertTrue(utils.is_zero(0))
        self.assertTrue(utils.is_zero((0, 0, 0)))
        self.assertTrue(utils.is_zero([0, 0, 0]))
        self.assertTrue(utils.is_zero(tensor([0, 0, 0])))
        self.assertTrue(utils.is_zero(tensor([[0, 0, 0]])))

    def test_get_size_and_device(self):
        devices = [device('cpu'), device('cuda')] if cuda.is_available() else [device('cpu')]
        for dev in devices:
            shape = SHAPES[0]
            x = rand(shape, device=dev)
            size, d = utils.get_size_and_device(x)
            self.assertEqual(size, shape)
            self.assertEqual(d.type, dev.type)
            shape = [s + 1 for s in shape]
            size, d = utils.get_size_and_device(x, size=shape)
            self.assertEqual(size, shape)
            self.assertEqual(d.type, dev.type)
            dev = device('cpu')
            size, d = utils.get_size_and_device(x, size=shape, device=dev)
            self.assertEqual(size, shape)
            self.assertEqual(d.type, x.device.type)

if __name__ == '__main__':
    unittest.main()
