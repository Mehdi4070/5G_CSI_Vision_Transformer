import unittest

import numpy as np

from src.noise import add_complex_awgn, measured_snr_db, normalized_complex_noise


class NoiseTests(unittest.TestCase):
    def test_adds_requested_per_user_snr(self) -> None:
        rng = np.random.default_rng(7)
        csi = (
            rng.normal(size=(4, 16, 32)) + 1j * rng.normal(size=(4, 16, 32))
        ).astype(np.complex64)
        unit_noise = normalized_complex_noise(csi.shape, rng)

        noisy = add_complex_awgn(csi, snr_db=10.0, unit_noise=unit_noise)

        np.testing.assert_allclose(measured_snr_db(csi, noisy), 10.0, atol=1e-4)

    def test_rejects_real_csi(self) -> None:
        with self.assertRaisesRegex(ValueError, "complex"):
            add_complex_awgn(
                np.ones((2, 3, 4), dtype=np.float32),
                snr_db=10.0,
                unit_noise=np.ones((2, 3, 4), dtype=np.complex64),
            )


if __name__ == "__main__":
    unittest.main()
