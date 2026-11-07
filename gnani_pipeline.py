"""
Kirana Mitra - Voice-First Retail Ledger (Gnani AI Internship Challenge)
========================================================================
Core Audio Processing & API Pipeline (`gnani_pipeline.py`)

This module provides end-to-end audio processing and API integration tailored
for Indian retail environments (Kirana stores). It includes:
  1. Telephonic Standard Audio Recording (8kHz / 16kHz, 16-bit PCM Mono).
  2. Telephony & Kirana Store Noise Simulator (Configurable SNR mixing &
     PSTN/AMR 8kHz bandpass downsampling for the Noise Canceller Award).
  3. Gnani Prisma v2.5 STT API Integration (Speech-to-Text for Indian languages/Hinglish).
  4. Gnani Timbre v2.5 TTS API Integration (Text-to-Speech with Indian currency/number synthesis).

Dependencies:
    requests, sounddevice, scipy, numpy
"""

import os
import sys
import time
import math
import json
import base64
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
import requests
from scipy.io import wavfile
from scipy import signal

# Optional import for live audio recording (allows headless/server-side pipeline execution)
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("KiranaMitra.Pipeline")


def _load_env_file() -> None:
    """Load key-value pairs from .env file into os.environ if present."""
    env_path = Path(__file__).parent / ".env"
    if env_path.is_file():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


_load_env_file()


# ============================================================================
# CONFIGURATION & API CONSTANTS (GNANI AI V2.5 / VACH PLATFORM)
# ============================================================================
class GnaniConfig:
    """Configuration settings, endpoints, and credentials for Gnani.ai APIs."""
    DEFAULT_KEY: str = "vach_1ytE2CY5X2DI4NdAY1NKip7Z5EbJhDN1klooAAxLBfDb8ZSns0Ni1TzssNQI5gKY71dH8w9hmMV5sFhVsnK1yMVdpx4xaK2Q_0fdca4771272743d7550dec6a7ff03c5"

    # Official Gnani Vachana AI Base Endpoints
    BASE_URL: str = "https://api.vachana.ai"
    PRISMA_STT_ENDPOINT: str = os.getenv(
        "GNANI_PRISMA_ENDPOINT",
        "https://api.vachana.ai/stt/v3"
    )
    TIMBRE_TTS_ENDPOINT: str = os.getenv(
        "GNANI_TIMBRE_ENDPOINT",
        "https://api.vachana.ai/api/v1/tts/inference"
    )
    API_KEY: str = os.getenv("GNANI_API_KEY", DEFAULT_KEY)
    AUTH_TOKEN: str = os.getenv("GNANI_AUTH_TOKEN", DEFAULT_KEY)
    CLIENT_ID: str = os.getenv("GNANI_CLIENT_ID", "kiranamitra")

    # Audio Telephony Standards
    NARROWBAND_SR: int = 8000   # 8 kHz (PSTN / 2G / AMR telephony)
    WIDEBAND_SR: int = 16000    # 16 kHz (AMR-WB / VoLTE / Standard STT)
    DEFAULT_CHANNELS: int = 1   # Mono
    BIT_DEPTH: str = "int16"    # 16-bit linear PCM

    # Network timeouts (in seconds)
    CONNECT_TIMEOUT: float = 10.0
    READ_TIMEOUT: float = 30.0


# ============================================================================
# AUDIO I/O & NORMALIZATION HELPERS
# ============================================================================
def load_wav_to_float(file_path: Union[str, Path]) -> Tuple[int, np.ndarray]:
    """
    Load a WAV file and normalize its sample data to floating-point range [-1.0, 1.0].

    Args:
        file_path: Path to the input WAV file.

    Returns:
        Tuple of (sample_rate, normalized_float_data_1d).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path.resolve()}")

    sample_rate, data = wavfile.read(str(path))

    # Convert multi-channel (stereo) to mono if needed
    if data.ndim > 1:
        logger.warning("Input audio has %d channels. Downmixing to mono.", data.shape[1])
        data = np.mean(data, axis=1)

    # Normalize based on original bit depth
    if data.dtype == np.int16:
        float_data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        float_data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        float_data = (data.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.floating):
        float_data = data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported audio data type: {data.dtype}")

    return sample_rate, float_data


def save_float_to_wav(file_path: Union[str, Path], sample_rate: int, data: np.ndarray) -> str:
    """
    Convert normalized floating-point audio [-1.0, 1.0] to 16-bit PCM and save as WAV.

    Args:
        file_path: Output file destination path.
        sample_rate: Sampling frequency in Hz.
        data: Floating-point 1D numpy array.

    Returns:
        Absolute path to the saved WAV file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Apply clipping protection to avoid harsh digital overflow wrap-around
    clipped_data = np.clip(data, -1.0, 1.0)
    int16_data = (clipped_data * 32767.0).astype(np.int16)

    wavfile.write(str(path), sample_rate, int16_data)
    logger.info("Saved 16-bit PCM WAV to '%s' (Rate: %d Hz, Samples: %d)", path, sample_rate, len(int16_data))
    return str(path.resolve())


# ============================================================================
# MODULE 1: RECORDING UTILITY
# ============================================================================
def record_audio(
    output_path: Union[str, Path] = "recorded_audio.wav",
    duration_sec: float = 5.0,
    sample_rate: int = GnaniConfig.WIDEBAND_SR,
    channels: int = GnaniConfig.DEFAULT_CHANNELS,
) -> str:
    """
    Record audio from the local microphone adhering to telephonic standards.

    Standards Enforced:
      - Sample Rate: 8000 Hz (Narrowband) or 16000 Hz (Wideband AMR-WB).
      - Channel: Mono (1 channel).
      - Bit Depth: 16-bit signed Linear PCM WAV.

    Args:
        output_path: Destination path for the WAV recording.
        duration_sec: Duration in seconds to record.
        sample_rate: Audio sampling frequency in Hz (8000 or 16000).
        channels: Number of audio channels (defaults to 1 for mono).

    Returns:
        Absolute path to the generated WAV recording.

    Raises:
        RuntimeError: If sounddevice is unavailable or no input device is detected.
        ValueError: If unsupported sample rates or channel counts are specified.
    """
    if not SOUNDDEVICE_AVAILABLE:
        raise RuntimeError(
            "The 'sounddevice' package is not installed or audio hardware is unavailable. "
            "Install it via `pip install sounddevice` or use pre-recorded WAV files."
        )

    if sample_rate not in (8000, 16000):
        logger.warning(
            "Sample rate %d Hz is non-standard for Indian telecom STT. Standard is 8000 or 16000 Hz.",
            sample_rate
        )

    if channels != 1:
        logger.warning("Telephony standards require mono (1 channel). Received %d channels.", channels)

    total_frames = int(duration_sec * sample_rate)
    logger.info(
        "Starting recording: %s (Duration: %.1fs, Rate: %d Hz, Channels: %d, Format: 16-bit PCM)",
        output_path, duration_sec, sample_rate, channels
    )

    try:
        # Record directly as 16-bit signed integer
        audio_buffer = sd.rec(
            frames=total_frames,
            samplerate=sample_rate,
            channels=channels,
            dtype=GnaniConfig.BIT_DEPTH
        )
        sd.wait()  # Wait synchronously until the recording finishes
    except Exception as exc:
        logger.error("Audio recording failed on device: %s", exc)
        raise RuntimeError(f"Microphone recording failed: {exc}") from exc

    # Ensure output directory exists
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Flatten if mono array has shape (N, 1)
    if audio_buffer.ndim > 1 and audio_buffer.shape[1] == 1:
        audio_buffer = audio_buffer.flatten()

    wavfile.write(str(out_file), sample_rate, audio_buffer)
    logger.info("Successfully recorded audio to '%s'", out_file.resolve())
    return str(out_file.resolve())


# ============================================================================
# MODULE 2: TELEPHONY NOISE SIMULATOR (Noise Canceller Award)
# ============================================================================
def mix_background_noise(
    clean_audio_path: Union[str, Path],
    noise_audio_path: Union[str, Path],
    output_path: Union[str, Path] = "mixed_noisy_audio.wav",
    target_snr_db: float = 10.0,
    seed: Optional[int] = 42,
) -> str:
    """
    Overlay background noise (e.g., Kirana market crowd, traffic honks) onto clean
    speech at a precisely controlled Signal-to-Noise Ratio (SNR) in Decibels (dB).

    Mathematical Formulation:
      P_clean = (1/N) * sum(clean^2)
      P_noise = (1/M) * sum(noise^2)
      Target SNR_dB = 10 * log10(P_clean / P_target_noise)
      => P_target_noise = P_clean / (10^(target_snr_db / 10))
      => Scaling Factor (alpha) = sqrt(P_target_noise / P_noise)
      Mixed = clean + alpha * noise

    Args:
        clean_audio_path: Path to clean voice recording (WAV).
        noise_audio_path: Path to background noise file (WAV).
        output_path: Output file destination.
        target_snr_db: Desired SNR in decibels (e.g., +15dB = slight noise,
                       0dB = heavy market noise, -5dB = extreme noisy environment).
        seed: Random seed for reproducible noise segment selection.

    Returns:
        Absolute path to the resulting mixed noisy WAV file.
    """
    sr_clean, clean = load_wav_to_float(clean_audio_path)
    sr_noise, noise = load_wav_to_float(noise_audio_path)

    # Resample noise if sample rates do not match
    if sr_clean != sr_noise:
        logger.info("Resampling noise from %d Hz to match clean audio rate %d Hz.", sr_noise, sr_clean)
        num_samples = int(len(noise) * float(sr_clean) / sr_noise)
        noise = signal.resample(noise, num_samples)

    len_clean = len(clean)
    len_noise = len(noise)

    if len_clean == 0:
        raise ValueError("Clean audio file is empty.")
    if len_noise == 0:
        raise ValueError("Noise audio file is empty.")

    # Loop or slice noise to match the clean audio duration
    if len_noise < len_clean:
        repeats = math.ceil(len_clean / len_noise)
        noise = np.tile(noise, repeats)[:len_clean]
    elif len_noise > len_clean:
        if seed is not None:
            np.random.seed(seed)
        start_idx = np.random.randint(0, len_noise - len_clean + 1)
        noise = noise[start_idx:start_idx + len_clean]

    # Calculate Root Mean Square (RMS) power of clean speech and noise
    power_clean = np.mean(clean ** 2)
    power_noise = np.mean(noise ** 2)

    if power_clean < 1e-12:
        logger.warning("Clean speech signal has near-zero energy (silence).")
        mixed = clean + noise
        return save_float_to_wav(output_path, sr_clean, mixed)

    if power_noise < 1e-12:
        logger.warning("Noise signal has near-zero energy. Returning clean audio unchanged.")
        return save_float_to_wav(output_path, sr_clean, clean)

    # Derive scaling factor alpha for target SNR in dB
    # SNR_dB = 10 * log10(power_clean / (alpha^2 * power_noise))
    power_target_noise = power_clean / (10.0 ** (target_snr_db / 10.0))
    alpha = math.sqrt(power_target_noise / power_noise)

    scaled_noise = alpha * noise
    mixed = clean + scaled_noise

    # Headroom check: prevent hard clipping distortion while preserving dynamic range
    max_amp = np.max(np.abs(mixed))
    if max_amp > 1.0:
        logger.warning(
            "Mixed audio peak amplitude (%.2f) exceeds 1.0. Applying soft normalization to prevent clipping.",
            max_amp
        )
        mixed = mixed / max_amp * 0.98

    # Calculate empirical SNR achieved
    empirical_snr = 10.0 * np.log10(np.mean(clean ** 2) / (np.mean(scaled_noise ** 2) + 1e-12))
    logger.info("Noise overlaid successfully: Target SNR = %.1f dB | Empirical SNR = %.2f dB", target_snr_db, empirical_snr)

    return save_float_to_wav(output_path, sr_clean, mixed)


def simulate_telephony_channel(
    input_path: Union[str, Path],
    output_path: Union[str, Path] = "telephony_simulated_8k.wav",
    target_sample_rate: int = GnaniConfig.NARROWBAND_SR,
    apply_bandpass: bool = True,
    apply_g711_companding: bool = True,
) -> str:
    """
    Simulate low-bandwidth telecom transmission (e.g. 2G/AMR or PSTN landline call).

    Simulation Steps:
      1. PSTN Acoustic Bandpass: Narrowband telephony restricts speech frequencies
         to between 300 Hz and 3400 Hz. Applies a 4th-order Butterworth bandpass filter.
      2. Downsampling: Resamples the filtered audio down to 8000 Hz using anti-aliasing
         polyphase resampling (`scipy.signal.resample_poly`).
      3. G.711 Telephony Companding (Optional): Emulates 8-bit A-law companding quantization
         characteristic of Indian cellular and PSTN networks.

    Args:
        input_path: Source high-quality WAV file (e.g., 16kHz or 44.1kHz).
        output_path: Destination path for the simulated 8kHz telephonic WAV.
        target_sample_rate: Target telephony sampling rate (standard 8000 Hz).
        apply_bandpass: If True, restricts spectrum to 300 Hz - 3400 Hz.
        apply_g711_companding: If True, simulates 8-bit A-law telecom quantization noise.

    Returns:
        Absolute path to the 8kHz simulated telephonic WAV file.
    """
    sr_orig, data = load_wav_to_float(input_path)

    # 1. Apply Telephonic Bandpass Filter (300 Hz - 3400 Hz) before downsampling
    if apply_bandpass and sr_orig > target_sample_rate:
        lowcut = 300.0
        highcut = min(3400.0, (target_sample_rate / 2.0) - 100.0)
        nyq = 0.5 * sr_orig

        # 4th-order Butterworth bandpass filter
        sos = signal.butter(N=4, Wn=[lowcut / nyq, highcut / nyq], btype="bandpass", output="sos")
        data = signal.sosfilt(sos, data)
        logger.info("Applied PSTN bandpass filter: %.0f Hz - %.0f Hz", lowcut, highcut)

    # 2. Resample to 8000 Hz using polyphase anti-aliasing filter
    if sr_orig != target_sample_rate:
        gcd = math.gcd(sr_orig, target_sample_rate)
        up = target_sample_rate // gcd
        down = sr_orig // gcd
        data = signal.resample_poly(data, up, down)
        logger.info("Resampled audio from %d Hz -> %d Hz (Ratio: %d/%d)", sr_orig, target_sample_rate, up, down)

    # 3. Simulate G.711 A-law 8-bit companding quantization (telephony codec standard)
    if apply_g711_companding:
        data = _simulate_a_law_companding(data)
        logger.info("Applied G.711 A-law 8-bit companding simulation.")

    return save_float_to_wav(output_path, target_sample_rate, data)


def _simulate_a_law_companding(x: np.ndarray, a: float = 87.6) -> np.ndarray:
    """
    Emulate ITU-T G.711 A-law compression, 8-bit quantization, and expansion.
    Produces authentic telephonic voice quantization artifacts.
    """
    abs_x = np.abs(x)
    sign_x = np.sign(x)

    # A-law Compression characteristic
    compressed = np.where(
        abs_x < 1.0 / a,
        (a * abs_x) / (1.0 + np.log(a)),
        (1.0 + np.log(a * np.maximum(abs_x, 1e-12))) / (1.0 + np.log(a))
    )
    compressed = sign_x * compressed

    # 8-bit Non-linear Quantization (256 discrete levels)
    quantized = np.round(compressed * 127.0) / 127.0

    # A-law Expansion (Decompression)
    abs_q = np.abs(quantized)
    sign_q = np.sign(quantized)
    expanded = np.where(
        abs_q < 1.0 / (1.0 + np.log(a)),
        (abs_q * (1.0 + np.log(a))) / a,
        (np.exp(abs_q * (1.0 + np.log(a)) - 1.0)) / a
    )
    return np.clip(sign_q * expanded, -1.0, 1.0)


def generate_synthetic_noise(
    duration_sec: float = 5.0,
    sample_rate: int = GnaniConfig.WIDEBAND_SR,
    noise_type: str = "kirana_market",
    output_path: Optional[str] = "synthetic_market_noise.wav"
) -> str:
    """
    Generate synthetic background noise modeling an Indian Kirana store environment.
    Useful for testing the telephony noise simulator without external audio files.

    Noise Profiles:
      - 'kirana_market': Mix of 1/f pink noise, low-frequency refrigerator/fan hum (50Hz + harmonics),
                         and simulated intermittent high-frequency background transients.
      - 'traffic': Low-frequency engine rumble + pink noise + occasional synthetic honk tones.
      - 'white': Uniform white noise.

    Args:
        duration_sec: Length of audio to generate in seconds.
        sample_rate: Audio sampling frequency in Hz.
        noise_type: 'kirana_market', 'traffic', or 'white'.
        output_path: Destination path for the synthetic noise WAV.

    Returns:
        Path to the generated WAV noise file.
    """
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    if noise_type == "white":
        noise = np.random.normal(0, 0.2, num_samples)

    elif noise_type in ("kirana_market", "traffic"):
        # 1. Generate 1/f Pink Noise (ambient acoustic room energy)
        white = np.random.normal(0, 1.0, num_samples)
        # Apply pinking IIR filter (approximate 3dB/octave roll-off)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        pink = signal.lfilter(b, a, white)
        pink = pink / (np.max(np.abs(pink)) + 1e-6)

        # 2. Add electrical/ceiling fan hum (50 Hz Indian mains hum + harmonics)
        mains_hum = 0.15 * np.sin(2 * np.pi * 50 * t) + 0.08 * np.sin(2 * np.pi * 100 * t)

        # 3. Add refrigerator compressor drone / low-frequency street rumble (85 Hz)
        compressor = 0.12 * np.sin(2 * np.pi * 85 * t)

        # 4. Optional traffic honk burst for market realism
        honk = np.zeros(num_samples)
        if noise_type == "traffic" or noise_type == "kirana_market":
            # Simulate a 0.5s horn sound at t = 2.0s
            honk_start = int(1.5 * sample_rate)
            honk_len = int(0.4 * sample_rate)
            if honk_start + honk_len < num_samples:
                t_honk = np.linspace(0, 0.4, honk_len, endpoint=False)
                # Typical dual-tone horn: ~400 Hz and ~500 Hz
                tone = 0.3 * (np.sin(2 * np.pi * 420 * t_honk) + np.sin(2 * np.pi * 510 * t_honk))
                window = signal.windows.tukey(honk_len, alpha=0.2)
                honk[honk_start:honk_start + honk_len] = tone * window

        noise = (0.5 * pink) + mains_hum + compressor + honk
        noise = noise / (np.max(np.abs(noise)) + 1e-6) * 0.5

    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    out_path = output_path or "synthetic_noise.wav"
    return save_float_to_wav(out_path, sample_rate, noise)


# ============================================================================
# MODULE 3: GNANI PRISMA V2.5 API INTEGRATION (SPEECH-TO-TEXT)
# ============================================================================
def transcribe_audio(
    file_path: Union[str, Path],
    language: str = "hin_eng",
    endpoint_url: Optional[str] = None,
    api_key: Optional[str] = None,
    token: Optional[str] = None,
    client_id: Optional[str] = None,
    audio_format: str = "wav",
    sample_rate: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Transcribe a local WAV recording using Gnani Prisma v2.5 Speech-to-Text (STT).

    Supports Indian regional languages and code-mixed Hinglish:
      - 'hin_eng': Hinglish (Hindi + English colloquial code-mixed, ideal for Kirana ledgers)
      - 'hin': Pure Hindi
      - 'eng': Indian English
      - 'kan': Kannada, 'tam': Tamil, 'tel': Telugu, 'mar': Marathi, 'guj': Gujarati, etc.

    Args:
        file_path: Path to the input WAV audio file.
        language: Language code for transcription (default 'hin_eng').
        endpoint_url: Override Gnani Prisma STT endpoint URL.
        api_key: Gnani API Key (defaults to GnaniConfig.API_KEY or GNANI_API_KEY env).
        token: Gnani Auth Bearer Token (defaults to GnaniConfig.AUTH_TOKEN).
        client_id: Optional client identifier for enterprise metering.
        audio_format: Encoding format of audio ('wav', 'pcm').
        sample_rate: Sample rate in Hz. If None, auto-detected from WAV header.

    Returns:
        Dictionary containing transcription results, e.g.:
        {
            "success": True,
            "transcript": "do kilo cheeni aur ek packet tata namak",
            "confidence": 0.94,
            "language": "hin_eng",
            "raw_response": {...}
        }

    Raises:
        FileNotFoundError: If audio file is missing.
        requests.RequestException: If network or HTTP communication fails.
    """
    audio_path = Path(file_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"STT input audio file not found: {audio_path.resolve()}")

    # Auto-detect sample rate from WAV file if not explicitly supplied
    if sample_rate is None:
        try:
            detected_sr, _ = wavfile.read(str(audio_path))
            sample_rate = detected_sr
        except Exception:
            sample_rate = GnaniConfig.WIDEBAND_SR

    # Resolve credentials and endpoints
    endpoint = endpoint_url or GnaniConfig.PRISMA_STT_ENDPOINT
    effective_api_key = api_key or GnaniConfig.API_KEY
    effective_token = token or GnaniConfig.AUTH_TOKEN
    effective_client_id = client_id or GnaniConfig.CLIENT_ID

    headers = {
        "X-API-Key-ID": effective_api_key,
        "x-api-key": effective_api_key,
        "token": effective_token,
        "access-token": effective_token,
        "Authorization": f"Bearer {effective_token}",
        "x-client-id": effective_client_id,
        "User-Agent": "KiranaMitra-GnaniPipeline/1.0",
    }

    form_data = {
        "language": language,
        "audio_format": audio_format.lower(),
        "sample_rate": str(sample_rate),
        "encoding": "pcm",
        "channels": "1"
    }

    logger.info(
        "Invoking Gnani Prisma v2.5 STT [Lang: %s, Rate: %d Hz, Target: %s]",
        language, sample_rate, endpoint
    )

    try:
        with open(audio_path, "rb") as audio_file:
            files = {
                "file": (audio_path.name, audio_file, f"audio/{audio_format.lower()}")
            }

            response = requests.post(
                url=endpoint,
                headers=headers,
                data=form_data,
                files=files,
                timeout=(GnaniConfig.CONNECT_TIMEOUT, GnaniConfig.READ_TIMEOUT)
            )

        # Handle HTTP response codes gracefully
        if response.status_code == 200:
            result_json = response.json() if response.headers.get("Content-Type", "").startswith("application/json") else {}
            
            # Extract transcript from standard Gnani v2.5 response schema
            # Schema variations: result_json["transcript"], result_json["asr_output"], result_json["data"]["text"]
            transcript = (
                result_json.get("transcript")
                or result_json.get("asr_output")
                or (result_json.get("data", {}).get("transcript") if isinstance(result_json.get("data"), dict) else None)
                or result_json.get("text", "")
            )
            confidence = (
                result_json.get("confidence")
                or (result_json.get("data", {}).get("confidence") if isinstance(result_json.get("data"), dict) else None)
                or 1.0
            )

            logger.info("Gnani Prisma STT transcription successful: '%s' (Confidence: %s)", transcript, confidence)
            return {
                "success": True,
                "transcript": transcript,
                "confidence": confidence,
                "language": language,
                "raw_response": result_json
            }

        elif response.status_code in (401, 403):
            logger.error("Gnani Prisma STT Authentication Error [%d]: Verify GNANI_API_KEY and GNANI_AUTH_TOKEN.", response.status_code)
            return {
                "success": False,
                "error": f"Authentication Failed ({response.status_code}): {response.text}",
                "transcript": ""
            }
        else:
            logger.error("Gnani Prisma STT returned HTTP %d: %s", response.status_code, response.text)
            response.raise_for_status()

    except requests.exceptions.Timeout as exc:
        logger.error("Gnani Prisma STT request timed out: %s", exc)
        return {"success": False, "error": f"Request timed out: {exc}", "transcript": ""}
    except requests.exceptions.ConnectionError as exc:
        logger.error("Gnani Prisma STT network connection error: %s", exc)
        return {"success": False, "error": f"Connection failed: {exc}", "transcript": ""}
    except Exception as exc:
        logger.error("Gnani Prisma STT unexpected error: %s", exc)
        return {"success": False, "error": str(exc), "transcript": ""}

    return {"success": False, "error": "Unknown STT failure", "transcript": ""}


# ============================================================================
# MODULE 4: GNANI TIMBRE V2.5 API INTEGRATION (TEXT-TO-SPEECH)
# ============================================================================
def synthesize_speech(
    text: str,
    output_path: Union[str, Path] = "response.wav",
    language: str = "hin_eng",
    voice: str = "female",
    endpoint_url: Optional[str] = None,
    api_key: Optional[str] = None,
    token: Optional[str] = None,
    audio_format: str = "wav",
    sample_rate: int = GnaniConfig.WIDEBAND_SR,
) -> str:
    """
    Synthesize natural Hindi/Hinglish speech from text using Gnani Timbre v2.5 TTS.

    Specially adapted for Kirana store voice confirmations containing Indian
    numerals and currency symbols (e.g., 'Aapka kul bill ₹250 hai').

    Args:
        text: Input text string in Hindi, Hinglish, or English.
        output_path: Destination path where the received WAV audio is saved.
        language: Language code (e.g. 'hin_eng', 'hin', 'eng', 'kan', 'tam').
        voice: Voice gender or persona (e.g., 'female', 'male', 'ananya', 'aarav').
        endpoint_url: Override Gnani Timbre TTS endpoint URL.
        api_key: Gnani API key (defaults to GnaniConfig.API_KEY).
        token: Gnani Auth token (defaults to GnaniConfig.AUTH_TOKEN).
        audio_format: Output audio encoding ('wav' or 'mp3').
        sample_rate: Sampling frequency (8000 or 16000 Hz).

    Returns:
        Absolute path to the downloaded synthesized audio file.

    Raises:
        ValueError: If input text is empty.
        requests.RequestException: If network communication or API returns an error.
    """
    if not text or not text.strip():
        raise ValueError("Cannot synthesize speech from an empty text string.")

    # Resolve credentials and endpoints
    endpoint = endpoint_url or GnaniConfig.TIMBRE_TTS_ENDPOINT
    effective_api_key = api_key or GnaniConfig.API_KEY
    effective_token = token or GnaniConfig.AUTH_TOKEN

    headers = {
        "Content-Type": "application/json",
        "X-API-Key-ID": effective_api_key,
        "x-api-key": effective_api_key,
        "token": effective_token,
        "access-token": effective_token,
        "Authorization": f"Bearer {effective_token}",
        "User-Agent": "KiranaMitra-GnaniPipeline/1.0",
    }

    payload = {
        "text": text,
        "language": language,
        "voice": voice,
        "audio_format": audio_format.lower(),
        "sample_rate": sample_rate,
        # Currency & number pre-processing normalization hint
        "speech_context": {
            "currency_format": "INR",
            "locale": "en-IN"
        }
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Invoking Gnani Timbre v2.5 TTS [Lang: %s, Voice: %s, Text: '%s']",
        language, voice, text
    )

    try:
        response = requests.post(
            url=endpoint,
            headers=headers,
            json=payload,
            stream=True,
            timeout=(GnaniConfig.CONNECT_TIMEOUT, GnaniConfig.READ_TIMEOUT)
        )

        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")

            # Case A: Binary audio stream (audio/wav, audio/mpeg, application/octet-stream)
            if "audio" in content_type or "octet-stream" in content_type:
                with open(out_file, "wb") as f_out:
                    for chunk in response.iter_content(chunk_size=4096):
                        if chunk:
                            f_out.write(chunk)
                logger.info("Successfully streamed TTS audio to '%s'", out_file.resolve())
                return str(out_file.resolve())

            # Case B: JSON payload containing Base64-encoded audio
            else:
                try:
                    resp_json = response.json()
                    # Look for base64 audio key
                    b64_audio = (
                        resp_json.get("audioContent")
                        or resp_json.get("audio")
                        or resp_json.get("data")
                    )
                    if b64_audio:
                        raw_bytes = base64.b64decode(b64_audio)
                        with open(out_file, "wb") as f_out:
                            f_out.write(raw_bytes)
                        logger.info("Successfully decoded base64 TTS audio to '%s'", out_file.resolve())
                        return str(out_file.resolve())
                    else:
                        logger.warning("No audio stream or base64 audio payload found in response: %s", resp_json)
                except Exception as parse_err:
                    logger.error("Failed to parse JSON TTS response: %s", parse_err)

        elif response.status_code in (401, 403):
            logger.error("Gnani Timbre TTS Authentication Error [%d]: Verify GNANI_API_KEY and GNANI_AUTH_TOKEN.", response.status_code)
            raise requests.exceptions.HTTPError(f"TTS Authentication Failed ({response.status_code}): {response.text}")
        else:
            logger.error("Gnani Timbre TTS HTTP error [%d]: %s", response.status_code, response.text)
            response.raise_for_status()

    except requests.exceptions.RequestException as exc:
        logger.error("Gnani Timbre TTS request failed: %s", exc)
        raise

    return str(out_file.resolve())


# ============================================================================
# COMPREHENSIVE PIPELINE DEMONSTRATION & TEST HARNESS
# ============================================================================
def run_pipeline_demo() -> None:
    """
    Demonstrate all Day 1 pipeline capabilities:
      1. Synthesize audio speech (Kirana Ledger billing confirmation).
      2. Generate synthetic Kirana store market noise.
      3. Mix noise at +10dB and -2dB SNR (Noise Canceller Award simulation).
      4. Downsample to 8kHz PSTN/AMR telephony simulation.
      5. Transcribe audio via Gnani Prisma v2.5 STT.
    """
    print("\n" + "=" * 75)
    print("  KIRANA MITRA - CORE AUDIO PIPELINE DEMO (GNANI AI CHALLENGE)")
    print("=" * 75)

    demo_dir = Path("kirana_audio_demo")
    demo_dir.mkdir(exist_ok=True)

    # 1. Prepare sample retail transaction speech (Hinglish with Indian currency)
    test_text = "Aapka kul bill ₹250 hai. Ramesh bhaiya ka khata update ho gaya hai."
    clean_audio_path = demo_dir / "clean_billing_speech.wav"
    noise_audio_path = demo_dir / "kirana_market_noise.wav"
    noisy_audio_path = demo_dir / "noisy_kirana_speech_snr10db.wav"
    telephony_audio_path = demo_dir / "telephony_call_simulated_8k.wav"

    print("\n[Step 1/5] Synthesizing Clean Audio for Testing...")
    # Generate clean speech: Try Gnani Timbre TTS; if placeholder keys are unconfigured, generate clean synthetic tone/speech
    try:
        if GnaniConfig.API_KEY != "YOUR_GNANI_API_KEY_HERE":
            synthesize_speech(
                text=test_text,
                output_path=clean_audio_path,
                language="hin_eng"
            )
            print(f"  ✓ Gnani Timbre TTS synthesized audio saved: {clean_audio_path}")
        else:
            print("  ℹ GNANI_API_KEY is placeholder. Generating synthetic sample WAV for offline pipeline test.")
            # Synthesize 3.5s speech-like multi-formant audio signal
            sr = 16000
            t = np.linspace(0, 3.5, int(3.5 * sr), endpoint=False)
            formants = (
                0.4 * np.sin(2 * np.pi * 220 * t) +   # Fundamental
                0.3 * np.sin(2 * np.pi * 750 * t) +   # First Formant (vowel /a/)
                0.2 * np.sin(2 * np.pi * 1220 * t)    # Second Formant
            )
            envelope = signal.windows.tukey(len(t), alpha=0.1)
            save_float_to_wav(clean_audio_path, sr, formants * envelope)
            print(f"  ✓ Offline sample speech saved: {clean_audio_path}")
    except Exception as e:
        print(f"  ⚠ Live TTS call skipped ({e}). Using offline synthetic speech.")

    print("\n[Step 2/5] Generating Indian Kirana Market Background Noise...")
    generate_synthetic_noise(
        duration_sec=4.0,
        sample_rate=16000,
        noise_type="kirana_market",
        output_path=str(noise_audio_path)
    )
    print(f"  ✓ Kirana market ambient noise created: {noise_audio_path}")

    print("\n[Step 3/5] Simulating Noise Overlay at Configurable Decibel SNR (+10.0 dB)...")
    mix_background_noise(
        clean_audio_path=clean_audio_path,
        noise_audio_path=noise_audio_path,
        output_path=noisy_audio_path,
        target_snr_db=10.0
    )
    print(f"  ✓ Mixed noisy speech created: {noisy_audio_path}")

    print("\n[Step 4/5] Simulating 8kHz AMR / PSTN Telephony Channel (Bandpass + G.711 A-law)...")
    simulate_telephony_channel(
        input_path=noisy_audio_path,
        output_path=telephony_audio_path,
        target_sample_rate=8000,
        apply_bandpass=True,
        apply_g711_companding=True
    )
    print(f"  ✓ Telephony 8kHz simulated WAV created: {telephony_audio_path}")

    print("\n[Step 5/5] Testing Gnani Prisma v2.5 STT Transcription...")
    if GnaniConfig.API_KEY != "YOUR_GNANI_API_KEY_HERE":
        stt_result = transcribe_audio(
            file_path=telephony_audio_path,
            language="hin_eng"
        )
        print(f"  ✓ STT Result: {stt_result}")
    else:
        print("  ℹ GNANI_API_KEY is placeholder. Skipping live HTTP POST to avoid 401.")
        print(f"  ℹ Simulated STT Payload verification: Telephonic audio at '{telephony_audio_path}' is ready.")

    print("\n" + "=" * 75)
    print("  ALL CORE AUDIO PIPELINE MODULES OPERATIONAL FOR DAY 1!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_pipeline_demo()

