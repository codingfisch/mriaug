import unittest
from torch import ones, rand, equal, randn, zeros, tensor, allclose, linspace

from mriaug import core
SHAPES = ((1, 1, 7, 8, 6), (1, 3, 7, 8, 6), (3, 1, 7, 8, 6), (3, 2, 7, 8, 6))


class TestCore(unittest.TestCase):
    def test_flip3d(self):
        for shape in SHAPES:
            x = rand(*shape)
            x_flipped = core.flip3d(x)
            self.assertEqual(x_flipped.shape, x.shape)

    def test_dihedral3d(self):
        shape = SHAPES[0]
        x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
        xs = [core.dihedral3d(x, k) for k in range(24)]
        for i, x0 in enumerate(xs):
            for j, x1 in enumerate(xs[i+1:]):
                self.assertFalse(equal(x0, x1), f'k0={i} and k1={j} produce equal tensors')

    def test_crop3d(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            translate = zeros((shape[0], 3))
            size = tuple(s - 2 for s in shape[-3:])
            x_cropped = core.crop3d(x, translate, size)
            self.assertEqual(x_cropped.shape, (*shape[:2], *size))
            translate[0, 0] = 1
            x_cropped = core.crop3d(x, translate, size)
            self.assertEqual(x_cropped.shape, (*shape[:2], *size))

    def test_zoom3d(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            zoom = zeros((shape[0], 3))
            size = tuple(s - 2 for s in shape[-3:])
            x_zoomed = core.zoom3d(x, zoom, size=size)
            self.assertEqual(x_zoomed.shape, (*shape[:2], *size))
            x_zoomed = core.zoom3d(x, zoom)
            self.assertTrue(allclose(x, x_zoomed))
            x_zoomed = core.zoom3d(x, zoom, upsample=2)
            self.assertTrue(allclose(x, x_zoomed, rtol=5e-2, atol=2e-1))

    def test_rotate3d(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            rotate = zeros((shape[0], 3))
            size = tuple(s - 2 for s in shape[-3:])
            x_rotated = core.rotate3d(x, rotate, size=size)
            self.assertEqual(x_rotated.shape, (*shape[:2], *size))
            x_rotated = core.rotate3d(x, rotate)
            self.assertTrue(allclose(x, x_rotated))
            x_rotated = core.rotate3d(x, rotate, upsample=2)
            self.assertTrue(allclose(x, x_rotated, rtol=5e-2, atol=2e-1))

    def test_translate3d(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            translate = zeros((shape[0], 3))
            size = tuple(s - 2 for s in shape[-3:])
            x_translated = core.translate3d(x, translate, size=size)
            self.assertEqual(x_translated.shape, (*shape[:2], *size))
            x_translated = core.translate3d(x, translate)
            self.assertTrue(allclose(x, x_translated))
            x_translated = core.translate3d(x, translate, upsample=2)
            self.assertTrue(allclose(x, x_translated, rtol=5e-2, atol=2e-1))

    def test_shear3d(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            shear = zeros((shape[0], 3))
            size = tuple(s - 2 for s in shape[-3:])
            x_sheared = core.shear3d(x, shear, size=size)
            self.assertEqual(x_sheared.shape, (*shape[:2], *size))
            x_sheared = core.shear3d(x, shear)
            self.assertTrue(allclose(x, x_sheared))
            x_sheared = core.shear3d(x, shear, upsample=2)
            self.assertTrue(allclose(x, x_sheared, rtol=5e-2, atol=2e-1))

    def test_affine3d(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            param = zeros((shape[0], 3))
            size = tuple(s - 2 for s in shape[-3:])
            x_affine = core.affine3d(x, zoom=param, rotate=param, translate=param, shear=param, size=size)
            self.assertEqual(x_affine.shape, (*shape[:2], *size))
            x_affine = core.affine3d(x, zoom=param, rotate=param, translate=param, shear=param)
            self.assertTrue(allclose(x, x_affine))
            x_affine = core.affine3d(x, zoom=param, rotate=param, translate=param, shear=param, upsample=2)
            self.assertTrue(allclose(x, x_affine, rtol=5e-2, atol=2e-1))

    def test_warp3d(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            size = tuple(s - 2 for s in shape[-3:])
            x_warped = core.warp3d(x, size=size)
            self.assertEqual(x_warped.shape, (*shape[:2], *size))
            x_warped = core.warp3d(x, magnitude=0)
            self.assertTrue(allclose(x, x_warped))
            x_warped = core.warp3d(x, magnitude=0, upsample=2)
            self.assertTrue(allclose(x, x_warped, rtol=5e-2, atol=2e-1))

    def test_affinewarp3d(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            param = zeros((shape[0], 3))
            size = tuple(s - 2 for s in shape[-3:])
            x_warped = core.affinewarp3d(x, zoom=param, rotate=param, translate=param, shear=param, size=size)
            self.assertEqual(x_warped.shape, (*shape[:2], *size))
            x_warped = core.affinewarp3d(x, zoom=param, rotate=param, translate=param, shear=param, magnitude=0)
            self.assertTrue(allclose(x, x_warped))
            x_warped = core.affinewarp3d(x, zoom=param, rotate=param, translate=param, shear=param, magnitude=0, upsample=2)
            self.assertTrue(allclose(x, x_warped, rtol=5e-2, atol=2e-1))

    def test_bias_field3d(self):
        k_size = (2, 2, 2)
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            intensity = zeros(shape[0])
            x_biased = core.bias_field3d(x, intensity, k_size=k_size)
            self.assertTrue(equal(x, x_biased))
            x_biased = core.bias_field3d(x, intensity=0, k_size=k_size)
            self.assertTrue(equal(x, x_biased))
            x_biased = core.bias_field3d(x, intensity=.1, k_size=k_size)
            self.assertFalse(equal(x, x_biased))
            k = randn((*shape[:2], *k_size))
            x_biased = core.bias_field3d(x, intensity=0, k=k)
            self.assertTrue(equal(x, x_biased))
            if shape[0] > 1:
                intensity[0] = .2
                x_biased = core.bias_field3d(x, intensity, k=k)
                self.assertFalse(equal(x_biased[0], x_biased[1]))

    def test_contrast(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            x_no_light = core.contrast(x, lighting=0)
            self.assertTrue(equal(x, x_no_light))
            x_light = core.contrast(x, lighting=.1)
            self.assertFalse(equal(x_light, x_no_light))

    def test_noise3d(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            x_no_noise = core.noise3d(x, intensity=0)
            self.assertTrue(equal(x, x_no_noise))
            x_noise = core.noise3d(x, intensity=.1)
            self.assertFalse(equal(x_no_noise, x_noise))

    def test_chi_noise3d(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            x_no_noise = core.chi_noise3d(x, intensity=0)
            self.assertTrue(allclose(x, x_no_noise))
            x_noise = core.chi_noise3d(x, intensity=.1)
            self.assertFalse(equal(x_no_noise, x_noise))

    def test_downsample3d(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            x_sampled = core.downsample3d(x, scale=1)
            self.assertTrue(equal(x, x_sampled))
            scale = ones((shape[0], 3))
            x_sampled = core.downsample3d(x, scale)
            self.assertTrue(equal(x, x_sampled))
            scale[0, 0] = .5
            x_sampled = core.downsample3d(x, scale)
            self.assertTrue(allclose(x, x_sampled, atol=5e-1))

    def test_ghosting3d(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            intensity = zeros(shape[0])
            x_ghosted = core.ghosting3d(x, intensity)
            self.assertTrue(allclose(x, x_ghosted, atol=1e-7))
            x_ghosted = core.ghosting3d(x, intensity, dim=1)
            self.assertTrue(allclose(x, x_ghosted, atol=1e-7))
            x_ghosted = core.ghosting3d(x, intensity=0)
            self.assertTrue(allclose(x, x_ghosted, atol=1e-7))
            x_ghosted = core.ghosting3d(x, intensity=.1)
            self.assertFalse(allclose(x, x_ghosted, atol=1e-7))
            if shape[0] > 1:
                intensity[0] = .2
                x_ghosted = core.ghosting3d(x, intensity)
                self.assertFalse(equal(x_ghosted[0], x_ghosted[1]))

    def test_spike3d(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            intensity = zeros(shape[0])
            x_spiked = core.spike3d(x, intensity)
            self.assertTrue(allclose(x, x_spiked, atol=1e-7))
            x_spiked = core.spike3d(x, intensity=0)
            self.assertTrue(allclose(x, x_spiked, atol=1e-7))
            # x_spiked = core.spike3d(x, intensity=10)
            # print((x_spiked - x).abs().max(), (x_spiked - x).abs().mean())
            # self.assertFalse(allclose(x, x_spiked, atol=1e-7))
            if shape[0] > 1:
                intensity[0] = .2
                x_spiked = core.spike3d(x, intensity)
                self.assertFalse(equal(x_spiked[0], x_spiked[1]))

    def test_ringing3d(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            intensity = zeros(shape[0])
            x_ringed = core.ringing3d(x, intensity)
            self.assertTrue(allclose(x, x_ringed, atol=1e-7))
            x_ringed = core.ringing3d(x, intensity, dim=1)
            self.assertTrue(allclose(x, x_ringed, atol=1e-7))
            x_ringed = core.ringing3d(x, intensity=0)
            self.assertTrue(allclose(x, x_ringed, atol=1e-7))
            x_ringed = core.ringing3d(x, intensity=.1)
            self.assertFalse(allclose(x, x_ringed, atol=1e-7))
            if shape[0] > 1:
                intensity[0] = .2
                x_ringed = core.ringing3d(x, intensity)
                self.assertFalse(equal(x_ringed[0], x_ringed[1]))

    def test_motion3d(self):
        for shape in SHAPES:
            x = linspace(0, 1, tensor(shape).prod().item()).view(*shape)
            intensity = zeros(shape[0])
            x_moved = core.motion3d(x, intensity)
            self.assertTrue(allclose(x, x_moved, atol=1e-7))
            x_moved = core.motion3d(x, intensity=0)
            self.assertTrue(allclose(x, x_moved, atol=1e-7))
            x_moved = core.motion3d(x, intensity=.1, translate=.2)
            self.assertFalse(allclose(x, x_moved))
            if shape[0] > 1:
                intensity[0] = .2
                x_moved = core.motion3d(x, intensity)
                self.assertFalse(equal(x_moved[0], x_moved[1]))


if __name__ == '__main__':
    unittest.main()
