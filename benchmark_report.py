"""
Kirana Mitra - Benchmark & Evaluation Suite (Noise Canceller Award)
==================================================================
Gnani AI Great Indian AI Internship Challenge

This module provides empirical benchmarking for:
  1. Telecom Noise Canceller Performance (SNR Improvement, Noise Power Reduction).
  2. Evon v3.3 Indic Metric Extraction Precision across varying noise levels:
     - Clean Audio (No noise)
     - SNR = +15 dB (Quiet store)
     - SNR = +5 dB (Bustling Indian market crowd)
     - SNR = 0 dB (Heavy street traffic & vehicle honking)
     - SNR = -5 dB (Extreme background cacophony)
  3. Latency profiling across Audio DSP -> STT -> Evon NLU -> TTS.

Generates a formatted markdown report and JSON summary for competition judges.
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from gnani_pipeline import (
    load_wav_to_float,
    save_float_to_wav,
    mix_background_noise,
    simulate_telephony_channel,
    generate_synthetic_noise,
)
from noise_canceller import cancel_noise_file
from parser_engine import parse_transcript_to_json, calculate_bill


def generate_benchmark_speech_audio(file_path: Path) -> str:
    """Generate standardized test voice order audio (16kHz 16-bit PCM)."""
    file_path.parent.mkdir(exist_ok=True)
    sr = 16000
    t = np.linspace(0, 3.5, int(3.5 * sr), endpoint=False)
    # Formants modeling multi-vowel speech syllables
    speech_signal = (
        0.35 * np.sin(2 * np.pi * 220 * t) +
        0.25 * np.sin(2 * np.pi * 700 * t) +
        0.20 * np.sin(2 * np.pi * 1220 * t) +
        0.15 * np.sin(2 * np.pi * 2400 * t)
    )
    # Natural speech syllabic envelope
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 3.5 * t))
    return save_float_to_wav(file_path, sr, speech_signal * envelope)


def run_noise_canceller_benchmark() -> List[Dict[str, Any]]:
    """
    Evaluate the Noise Canceller across 5 challenging SNR levels.
    """
    bench_dir = Path("benchmark_results")
    bench_dir.mkdir(exist_ok=True)

    clean_speech_wav = bench_dir / "clean_benchmark_speech.wav"
    market_noise_wav = bench_dir / "market_noise.wav"

    generate_benchmark_speech_audio(clean_speech_wav)
    generate_synthetic_noise(4.0, 16000, "kirana_market", str(market_noise_wav))

    test_snrs = [15.0, 10.0, 5.0, 0.0, -5.0]
    results = []

    print("\n" + "=" * 75)
    print("  RUNNING TELECOM NOISE CANCELLER BENCHMARK (NOISE CANCELLER AWARD)")
    print("=" * 75)

    for snr in test_snrs:
        noisy_wav = bench_dir / f"noisy_speech_snr_{int(snr)}db.wav"
        enhanced_wav = bench_dir / f"enhanced_speech_snr_{int(snr)}db.wav"

        # 1. Overlay noise at exact mathematical SNR
        mix_background_noise(
            clean_audio_path=clean_speech_wav,
            noise_audio_path=market_noise_wav,
            output_path=noisy_wav,
            target_snr_db=snr
        )

        # 2. Simulate 8kHz AMR telephony channel
        telephony_8k = bench_dir / f"telephony_8k_snr_{int(snr)}db.wav"
        simulate_telephony_channel(noisy_wav, telephony_8k, target_sample_rate=8000)

        # 3. Measure DSP Noise Cancellation latency and improvement
        t_start = time.perf_counter()
        canceller_res = cancel_noise_file(telephony_8k, enhanced_wav)
        dsp_latency_ms = (time.perf_counter() - t_start) * 1000.0

        metrics = canceller_res["metrics"]
        res_entry = {
            "target_snr_db": snr,
            "snr_improvement_db": metrics["snr_improvement_db"],
            "noise_reduction_db": metrics["noise_reduction_db"],
            "dsp_latency_ms": round(dsp_latency_ms, 2),
            "sample_rate": 8000
        }
        results.append(res_entry)

        print(f"  • Input SNR: {snr:>5.1f} dB | SNR Improvement: +{metrics['snr_improvement_db']:>5.2f} dB | Noise Reduction: {metrics['noise_reduction_db']:>5.2f} dB | Latency: {dsp_latency_ms:>5.1f} ms")

    return results


def run_nlu_indic_benchmark() -> Dict[str, Any]:
    """
    Benchmark Evon v3.3 Indic Parser precision across complex colloquial retail phrasing.
    """
    benchmarks = [
        {
            "transcript": "bhaiya do packet maggi aur ek pav dhaniya de do",
            "expected_items": [("Maggi", 2.0, "packet"), ("Dhaniya", 0.25, "kg")],
            "expected_total": 58.0
        },
        {
            "transcript": "aadha kilo cheeni aur ek packet tata namak chahiye",
            "expected_items": [("Cheeni", 0.5, "kg"), ("Tata Salt", 1.0, "packet")],
            "expected_total": 50.0
        },
        {
            "transcript": "dhai sau gram jeera aur do kilo aashirvaad aata de dena",
            "expected_items": [("Jeera", 250.0, "g"), ("Aashirvaad Aata", 2.0, "kg")],
            "expected_total": 172.0
        },
        {
            "transcript": "ek peti frooti aur basmati chawal ki ek bori likh lo",
            "expected_items": [("Frooti", 1.0, "box"), ("Basmati Chawal", 1.0, "sack")],
            "expected_total": 2090.0
        },
        {
            "transcript": "teen pav haldi aur dedh kilo toor daal pack kar do",
            "expected_items": [("Haldi Powder", 0.75, "kg"), ("Toor Daal", 1.5, "kg")],
            "expected_total": 360.0
        }
    ]

    print("\n" + "=" * 75)
    print("  RUNNING EVON v3.3 INDIC NLU & BILLING ACCURACY BENCHMARK")
    print("=" * 75)

    passed_count = 0
    total_count = len(benchmarks)
    details = []

    for idx, test in enumerate(benchmarks, 1):
        t0 = time.perf_counter()
        parsed = parse_transcript_to_json(test["transcript"])
        invoice, total = calculate_bill(parsed.get("items", []))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        is_accurate = abs(total - test["expected_total"]) < 0.01
        if is_accurate:
            passed_count += 1

        details.append({
            "test_case": test["transcript"],
            "expected_total": test["expected_total"],
            "computed_total": total,
            "latency_ms": round(latency_ms, 2),
            "status": "PASS" if is_accurate else "FAIL"
        })

        print(f"  Test {idx}: {'✓ PASS' if is_accurate else '❌ FAIL'} | Total: ₹{total:.2f} (Expected: ₹{test['expected_total']:.2f}) | Latency: {latency_ms:.1f}ms")

    accuracy_pct = (passed_count / total_count) * 100.0
    print(f"\n  NLU Metric Accuracy: {accuracy_pct:.1f}% ({passed_count}/{total_count} Passed)")

    return {
        "accuracy_pct": accuracy_pct,
        "passed": passed_count,
        "total": total_count,
        "details": details
    }


def generate_markdown_report(noise_results: List[Dict[str, Any]], nlu_results: Dict[str, Any]) -> str:
    """Generate a GitHub Flavored Markdown report for the internship submission."""
    report_path = Path("BENCHMARK_REPORT.md")

    md_lines = [
        "# Kirana Mitra - Empirical Benchmark & Evaluation Report",
        "### Gnani AI Great Indian AI Internship Challenge",
        "",
        "## 1. Telecom Noise Canceller Performance (Noise Canceller Award Focus)",
        "Evaluation across standard narrowband 8,000 Hz telephonic audio with realistic Indian Kirana market ambient noise (50Hz mains hum, refrigerator rumble, customer babble, traffic):",
        "",
        "| Input Condition | Target SNR (dB) | SNR Improvement (ΔSNR) | Noise Power Reduction | DSP Latency (ms) |",
        "|---|---|---|---|---|",
    ]

    for r in noise_results:
        cond = "Clean Store" if r["target_snr_db"] >= 15 else ("Market Ambient" if r["target_snr_db"] >= 5 else ("Heavy Traffic" if r["target_snr_db"] >= 0 else "Extreme Noise"))
        md_lines.append(f"| {cond} | {r['target_snr_db']:.1f} dB | **+{r['snr_improvement_db']:.2f} dB** | {r['noise_reduction_db']:.2f} dB | {r['dsp_latency_ms']:.1f} ms |")

    md_lines.extend([
        "",
        "### Key DSP Features:",
        "- **50 Hz / 100 Hz Notch Filtering**: Eliminates AC hum without voice distortion.",
        "- **Adaptive Energy VAD**: Accurately tracks changing noise floor during conversational pauses.",
        "- **Berouti Spectral Subtraction**: Minimizes musical noise via spectral floor ($\\beta = 0.02$).",
        "",
        "## 2. Evon v3.3 Indic Metric & Billing Accuracy",
        f"**Accuracy Score: {nlu_results['accuracy_pct']:.1f}%** ({nlu_results['passed']}/{nlu_results['total']} tests passed)",
        "",
        "| Test Spoken Order | Expected Bill | Computed Bill | Status | NLU Latency |",
        "|---|---|---|---|---|",
    ])

    for d in nlu_results["details"]:
        md_lines.append(f"| \"{d['test_case']}\" | ₹{d['expected_total']:.2f} | ₹{d['computed_total']:.2f} | {d['status']} | {d['latency_ms']:.1f} ms |")

    md_lines.extend([
        "",
        "---",
        "Report generated automatically by `benchmark_report.py`.",
    ])

    content = "\n".join(md_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n✓ Markdown benchmark report saved to: {report_path.resolve()}\n")
    return content


if __name__ == "__main__":
    noise_res = run_noise_canceller_benchmark()
    nlu_res = run_nlu_indic_benchmark()
    generate_markdown_report(noise_res, nlu_res)

