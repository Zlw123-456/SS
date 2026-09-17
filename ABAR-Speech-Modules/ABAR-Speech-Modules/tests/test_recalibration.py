import unittest

import torch

from abar_speech.recalibration import BandImportanceGate, BandRecalibration


class RecalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.module = BandRecalibration(8, 4)
        self.x = torch.randn(2, 7, 8, 11)

    def test_shape_and_broadcasting(self):
        y, gates = self.module(self.x, return_gate=True)
        self.assertEqual(y.shape, self.x.shape)
        self.assertEqual(tuple(gates.shape), (2, 7))
        self.assertTrue(bool(((gates > 0) & (gates < 1)).all()))
        torch.testing.assert_close(y, self.x * gates[:, :, None, None])
        torch.testing.assert_close(self.module(self.x), y)

    def test_zero_scorer_halves_features(self):
        with torch.no_grad():
            for parameter in self.module.parameters():
                parameter.zero_()
        y, gates = self.module(self.x, return_gate=True)
        torch.testing.assert_close(gates, torch.full((2, 7), 0.5))
        torch.testing.assert_close(y, 0.5 * self.x)

    def test_shared_network_is_band_permutation_equivariant(self):
        order = torch.tensor([4, 1, 6, 0, 2, 5, 3])
        torch.testing.assert_close(self.module(self.x[:, order]), self.module(self.x)[:, order])

    def test_temporal_pooling_is_time_permutation_invariant(self):
        order = torch.randperm(self.x.shape[-1])
        _, original = self.module(self.x, return_gate=True)
        _, permuted = self.module(self.x[..., order], return_gate=True)
        torch.testing.assert_close(original, permuted)

    def test_backward(self):
        x = self.x.clone().requires_grad_(True)
        self.module(x).square().mean().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        for parameter in self.module.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_parameter_count(self):
        gate = BandRecalibration(128, 64)
        self.assertEqual(sum(p.numel() for p in gate.parameters()), 8321)

    def test_standalone_state_dict(self):
        clone = BandImportanceGate(8, 4)
        clone.load_state_dict(self.module.state_dict(), strict=True)
        self.assertEqual(set(clone.state_dict()), {"scorer.0.weight", "scorer.0.bias", "scorer.2.weight", "scorer.2.bias"})
        torch.testing.assert_close(clone(self.x), self.module(self.x))

    def test_float64_and_single_frame(self):
        module = self.module.double()
        x = torch.randn(1, 1, 8, 1, dtype=torch.float64)
        y, gates = module(x, return_gate=True)
        self.assertEqual(y.dtype, torch.float64)
        self.assertEqual(gates.shape, (1, 1))

    def test_invalid_shape(self):
        for shape in ((2, 7, 8), (2, 7, 9, 11), (2, 7, 8, 0)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.module(torch.empty(shape))
        with self.assertRaises(TypeError):
            self.module(torch.ones(2, 7, 8, 11, dtype=torch.int64))

    def test_invalid_dimensions(self):
        for features, hidden in ((0, 4), (8, 0), (-1, 4)):
            with self.subTest(features=features, hidden=hidden), self.assertRaises(ValueError):
                BandRecalibration(features, hidden)


if __name__ == "__main__":
    unittest.main()
