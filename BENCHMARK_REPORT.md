# Kirana Mitra - Empirical Benchmark & Evaluation Report
### Gnani AI Great Indian AI Internship Challenge

## 1. Telecom Noise Canceller Performance (Noise Canceller Award Focus)
Evaluation across standard narrowband 8,000 Hz telephonic audio with realistic Indian Kirana market ambient noise (50Hz mains hum, refrigerator rumble, customer babble, traffic):

| Input Condition | Target SNR (dB) | SNR Improvement (ΔSNR) | Noise Power Reduction | DSP Latency (ms) |
|---|---|---|---|---|
| Clean Store | 15.0 dB | **+16.86 dB** | 17.83 dB | 31.8 ms |
| Market Ambient | 10.0 dB | **+15.06 dB** | 16.16 dB | 24.8 ms |
| Market Ambient | 5.0 dB | **+12.20 dB** | 13.61 dB | 29.9 ms |
| Heavy Traffic | 0.0 dB | **+8.06 dB** | 10.15 dB | 30.4 ms |
| Extreme Noise | -5.0 dB | **+5.92 dB** | 8.21 dB | 22.8 ms |

### Key DSP Features:
- **50 Hz / 100 Hz Notch Filtering**: Eliminates AC hum without voice distortion.
- **Adaptive Energy VAD**: Accurately tracks changing noise floor during conversational pauses.
- **Berouti Spectral Subtraction**: Minimizes musical noise via spectral floor ($\beta = 0.02$).

## 2. Evon v3.3 Indic Metric & Billing Accuracy
**Accuracy Score: 100.0%** (5/5 tests passed)

| Test Spoken Order | Expected Bill | Computed Bill | Status | NLU Latency |
|---|---|---|---|---|
| "bhaiya do packet maggi aur ek pav dhaniya de do" | ₹58.00 | ₹58.00 | PASS | 37.0 ms |
| "aadha kilo cheeni aur ek packet tata namak chahiye" | ₹50.00 | ₹50.00 | PASS | 12.4 ms |
| "dhai sau gram jeera aur do kilo aashirvaad aata de dena" | ₹172.00 | ₹172.00 | PASS | 10.0 ms |
| "ek peti frooti aur basmati chawal ki ek bori likh lo" | ₹2090.00 | ₹2090.00 | PASS | 11.3 ms |
| "teen pav haldi aur dedh kilo toor daal pack kar do" | ₹360.00 | ₹360.00 | PASS | 11.8 ms |

---
Report generated automatically by `benchmark_report.py`.