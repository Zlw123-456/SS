import unittest

import numpy as np

from abar_speech import band_slices, build_exp_band, build_hybrid_band


class AllocationTests(unittest.TestCase):
    def test_speech_partition(self):
        expected = [1] * 60 + [11, 15, 21, 30, 43, 59, 82]
        self.assertEqual(build_hybrid_band(321), expected)

    def test_exact_coverage_and_nonempty_bands(self):
        for bins in (1, 2, 7, 67, 321):
            for count in sorted({1, min(7, bins), bins}):
                for alpha in (0.01, 1.5, 2.0, 2.5, 8.0):
                    with self.subTest(bins=bins, count=count, alpha=alpha):
                        widths = build_exp_band(bins, count, alpha)
                        self.assertEqual(len(widths), count)
                        self.assertEqual(sum(widths), bins)
                        self.assertTrue(all(isinstance(w, int) and w > 0 for w in widths))

    def test_one_high_band(self):
        self.assertEqual(build_hybrid_band(10, 3, 4), [1, 1, 1, 7])

    def test_zero_low_bins(self):
        self.assertEqual(build_hybrid_band(321, 0, 67), build_exp_band(321, 67))

    def test_one_bin_per_band(self):
        self.assertEqual(build_hybrid_band(67, 60, 67), [1] * 67)

    def test_rounding_correction_order(self):
        self.assertEqual(build_exp_band(8, 3, 0.01), [2, 3, 3])
        self.assertEqual(build_exp_band(8, 6, 8.0), [1, 1, 1, 1, 1, 3])

    def test_slices(self):
        regions = band_slices(build_hybrid_band(321))
        self.assertEqual(regions[60], slice(60, 71))
        self.assertEqual(regions[-1], slice(239, 321))
        flattened = [index for region in regions for index in range(region.start, region.stop)]
        self.assertEqual(flattened, list(range(321)))

    def test_numpy_integer_inputs(self):
        self.assertEqual(build_exp_band(np.int64(5), np.int64(1)), [5])

    def test_invalid_counts(self):
        for bins, count in ((0, 1), (-1, 1), (5, 0), (5, 6)):
            with self.subTest(bins=bins, count=count), self.assertRaises(ValueError):
                build_exp_band(bins, count)
        for value in (1.5, True, "5"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                build_exp_band(value, 1)

    def test_invalid_alpha(self):
        for alpha in (0, -1, float("nan"), float("inf"), 1000):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                build_exp_band(10, 3, alpha)
        with self.assertRaises(TypeError):
            build_exp_band(10, 3, True)

    def test_invalid_hybrid_settings(self):
        for low, bands in ((-1, 5), (5, 5), (0, 11)):
            with self.subTest(low=low, bands=bands), self.assertRaises(ValueError):
                build_hybrid_band(10, low, bands)
        with self.assertRaises(TypeError):
            build_hybrid_band(10, 1.5, 5)

    def test_invalid_slices(self):
        for widths in ([], [1, 0], [1, -1]):
            with self.subTest(widths=widths), self.assertRaises(ValueError):
                band_slices(widths)


if __name__ == "__main__":
    unittest.main()
