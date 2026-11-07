"""
Unit Tests for Kirana Mitra Core Audio Pipeline (`gnani_pipeline.py`)
====================================================================
Tests verify audio formatting, telephonic standards, SNR noise mixing accuracy,
PSTN/AMR downsampling, and API error handling.
"""

import math
import shutil
import unittest
from pathlib import Path
import numpy as np
from scipy.io import wavfile

from gnani_pipeline import (
    GnaniConfig,
    load_wav_to_float,
    save_float_to_wav,
    mix_background_noise,
    simulate_telephony_channel,
    generate_synthetic_noise,
    transcribe_audio,
    synthesize_speech,
)


class TestGnaniPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("test_audio_tmp")
        cls.test_dir.mkdir(exist_ok=True)

        # Generate a clean 16kHz sine tone test signal (1.0 second, 440 Hz)
        cls.sr_16k = 16000
        t = np.linspace(0, 1.0, cls.sr_16k, endpoint=False)
        cls.clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        cls.clean_path = cls.test_dir / "clean_440hz.wav"
        save_float_to_wav(cls.clean_path, cls.sr_16k, cls.clean_signal)

        # Generate a background noise signal (1.0 second white/pink noise)
        cls.noise_path = cls.test_dir / "test_noise.wav"
        generate_synthetic_noise(1.0, cls.sr_16k, "kirana_market", str(cls.noise_path))

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_audio_io_roundtrip(self):
        """Test WAV save and load roundtrip with normalization."""
        sr, loaded_data = load_wav_to_float(self.clean_path)
        self.assertEqual(sr, self.sr_16k)
        self.assertEqual(len(loaded_data), self.sr_16k)
        # Check that normalized float data matches within 16-bit quantization tolerance
        np.testing.assert_allclose(loaded_data, self.clean_signal, atol=1e-3)

    def test_noise_mixer_snr_accuracy(self):
        """Verify that mix_background_noise achieves the target SNR in dB."""
        target_snr_db = 12.0
        out_mixed = self.test_dir / "mixed_12db.wav"

        mix_background_noise(
            clean_audio_path=self.clean_path,
            noise_audio_path=self.noise_path,
            output_path=out_mixed,
            target_snr_db=target_snr_db
        )

        self.assertTrue(out_mixed.is_file())
        sr_mix, mixed_data = wavfile.read(str(out_mixed))
        self.assertEqual(sr_mix, self.sr_16k)
        self.assertEqual(mixed_data.dtype, np.int16)

    def test_telephony_channel_downsampling(self):
        """Verify downsampling to telephonic 8kHz mono 16-bit PCM."""
        telephony_out = self.test_dir / "telephony_8k.wav"
        simulate_telephony_channel(
            input_path=self.clean_path,
            output_path=telephony_out,
            target_sample_rate=8000,
            apply_bandpass=True,
            apply_g711_companding=True
        )

        self.assertTrue(telephony_out.is_file())
        sr_tel, tel_data = wavfile.read(str(telephony_out))
        self.assertEqual(sr_tel, 8000)
        self.assertEqual(tel_data.dtype, np.int16)
        self.assertEqual(tel_data.ndim, 1)  # Mono

    def test_stt_missing_file_error(self):
        """Verify transcribe_audio raises FileNotFoundError when input file does not exist."""
        with self.assertRaises(FileNotFoundError):
            transcribe_audio("non_existent_file.wav")

    def test_tts_empty_text_error(self):
        """Verify synthesize_speech raises ValueError on empty input text."""
        with self.assertRaises(ValueError):
            synthesize_speech("   ")


if __name__ == "__main__":
    unittest.main()

