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
        self.assertTrue(bool(((gates > 0) & (gates < 2)).all()))
        torch.testing.assert_close(y, self.x * gates[:, :, None, None])
        torch.testing.assert_close(self.module(self.x), y)

    def test_zero_scorer_preserves_features(self):
        with torch.no_grad():
            for parameter in self.module.parameters():
                parameter.zero_()
        y, gates = self.module(self.x, return_gate=True)
        torch.testing.assert_close(gates, torch.ones(2, 7))
        torch.testing.assert_close(y, self.x)

    def test_silu_and_scaled_sigmoid(self):
        with torch.no_grad():
            self.module.scorer[0].weight.fill_(0.1)
            self.module.scorer[0].bias.fill_(-1.0)
            self.module.scorer[2].weight.fill_(0.3)
            self.module.scorer[2].bias.fill_(0.2)
        pooled = self.x.mean(dim=-1)
        hidden = torch.nn.functional.silu(self.module.scorer[0](pooled))
        expected = 2.0 * torch.sigmoid(self.module.scorer[2](hidden))
        _, actual = self.module(self.x, return_gate=True)
        torch.testing.assert_close(actual, expected.squeeze(-1))
        self.assertIsInstance(self.module.scorer[1], torch.nn.SiLU)
        self.assertEqual(len(self.module.scorer), 3)

    def test_attenuation_and_amplification(self):
        with torch.no_grad():
            self.module.scorer[2].weight.zero_()
        for bias in (-2.0, 2.0):
            with self.subTest(bias=bias), torch.no_grad():
                self.module.scorer[2].bias.fill_(bias)
                y, gates = self.module(self.x, return_gate=True)
                expected = 2.0 * torch.sigmoid(torch.tensor(bias))
                torch.testing.assert_close(gates, expected.expand_as(gates))
                torch.testing.assert_close(y, expected * self.x)
                self.assertTrue(bool((gates < 1).all()) if bias < 0 else bool((gates > 1).all()))

    def test_initialization_matches_specification(self):
        torch.manual_seed(123)
        first = torch.nn.Linear(8, 4)
        second = torch.nn.Linear(4, 1)
        torch.nn.init.zeros_(first.bias)
        torch.nn.init.normal_(second.weight, mean=0.0, std=1e-3)
        torch.nn.init.zeros_(second.bias)
        torch.manual_seed(123)
        module = BandRecalibration(8, 4)
        for actual, expected in zip(module.parameters(), [*first.parameters(), *second.parameters()]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_small_initial_logits_give_near_identity(self):
        _, gates = self.module(self.x, return_gate=True)
        self.assertLess(float((gates - 1.0).abs().max().detach()), 0.01)
        self.assertTrue(bool((gates != 1.0).any()))

    def test_matching_state_keys_do_not_guarantee_old_gate_behavior(self):
        earlier_scorer = torch.nn.Sequential(
            torch.nn.Linear(8, 4), torch.nn.ReLU(),
            torch.nn.Linear(4, 1), torch.nn.Sigmoid(),
        )
        earlier_scorer.load_state_dict(self.module.scorer.state_dict(), strict=True)
        earlier_gates = earlier_scorer(self.x.mean(dim=-1)).squeeze(-1)
        _, current_gates = self.module(self.x, return_gate=True)
        self.assertFalse(torch.allclose(earlier_gates, current_gates))

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
