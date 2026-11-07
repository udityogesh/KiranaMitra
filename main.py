"""
Kirana Mitra - Voice-First Retail Ledger (Unified Application)
==============================================================
Gnani AI Great Indian AI Internship Challenge

Unified console interface bringing together:
  - Day 1: Telephonic Audio Pipeline & DSP Noise Canceller
  - Day 2: Evon v3.3 Indic Parser & Inventory Billing Engine
  - Day 3: Persistent SQLite Khata (Credit Ledger) & WhatsApp UPI Receipts

Usage:
  python main.py
"""

import os
import sys
import time
from pathlib import Path

# Core modules
from gnani_pipeline import (
    GnaniConfig,
    record_audio,
    mix_background_noise,
    simulate_telephony_channel,
    generate_synthetic_noise,
    SOUNDDEVICE_AVAILABLE
)
from noise_canceller import cancel_noise_file
from parser_engine import process_voice_order, KiranaInventoryDB
from khata_ledger import KiranaKhataLedger, process_khata_voice_command


def banner() -> None:
    print("=" * 75)
    print("        🛒  KIRANA MITRA - VOICE-FIRST RETAIL LEDGER  🛒")
    print("      Gnani AI Great Indian AI Internship Challenge Solution")
    print("=" * 75)
    key_preview = GnaniConfig.API_KEY[:15] + "..." if GnaniConfig.API_KEY else "Not Set"
    print(f"  [API Status] Gnani Vach Credentials: {key_preview}")
    print(f"  [Mic Status] Hardware Recording: {'Available' if SOUNDDEVICE_AVAILABLE else 'Pre-recorded files only'}")
    print("=" * 75)


def menu() -> None:
    print("\nSelect an action:")
    print("  1. 🎙️  Process Voice Grocery Order (STT -> Evon v3.3 -> Bill -> TTS)")
    print("  2. 📒  Voice Khata / Udhaar Management (Add Credit, Record Payment)")
    print("  3. 🔍  Check Customer Khata Balance & Statement")
    print("  4. 🔇  Noise Canceller Award Demonstration (DSP Speech Enhancer)")
    print("  5. 📦  Inspect Store Inventory & Prices")
    print("  6. 🧪  Run All Automated Test Suites")
    print("  0. ❌  Exit")


def action_voice_order(ledger: KiranaKhataLedger) -> None:
    print("\n--- [Option 1: Process Voice Grocery Order] ---")
    demo_audio = Path("kirana_audio_demo") / "clean_billing_speech.wav"

    if SOUNDDEVICE_AVAILABLE:
        choice = input("Record live from microphone? (y/N): ").strip().lower()
        if choice == "y":
            rec_path = Path("kirana_audio_demo") / "live_recording.wav"
            print("Recording for 5 seconds... Please speak your grocery order in Hindi/Hinglish:")
            print("  (e.g., 'Bhaiya do packet maggi aur ek pav dhaniya de do')")
            record_audio(output_path=rec_path, duration_sec=5.0)
            demo_audio = rec_path

    cust_name = input("Enter Customer Name (Default: Ramesh Bhaiya): ").strip() or "Ramesh Bhaiya"

    print("\nProcessing voice order through Kirana Mitra...")
    result = process_voice_order(
        audio_file_path=demo_audio,
        output_speech_path="kirana_audio_demo/order_confirmation.wav",
        customer_name=cust_name,
        apply_noise_cleaning=True
    )

    print("\n" + "─" * 50)
    print("🧾 DIGITAL INVOICE GENERATED:")
    print("─" * 50)
    for line in result["invoice"]:
        print(f" • {line['matched_name']} ({line['ordered_quantity']} {line['ordered_unit']}) @ ₹{line['rate']} -> ₹{line['subtotal']:.2f} [{line['status']}]")
    print("─" * 50)
    print(f"💰 GRAND TOTAL: ₹{result['grand_total_inr']:.2f}")
    print(f"🔊 Spoken Hindi Confirmation: \"{result['spoken_confirmation']}\"")
    print(f"🎵 Response Audio Saved: {result['confirmation_audio_path']}")

    # Ask if this should be added to Khata credit
    khata_choice = input("\nAdd this invoice to customer's Khata credit (Udhaar)? (y/N): ").strip().lower()
    if khata_choice == "y":
        rec = ledger.record_invoice(
            customer_name=cust_name,
            invoice_lines=result["invoice"],
            total_amount=result["grand_total_inr"],
            payment_status="KHATA_CREDIT",
            voice_transcript=result["raw_transcript"]
        )
        print("\n📲 WhatsApp Receipt Generated:")
        print(rec["whatsapp_receipt"])


def action_voice_khata(ledger: KiranaKhataLedger) -> None:
    print("\n--- [Option 2: Voice Khata Management] ---")
    print("Enter or speak a colloquial Khata command, for example:")
    print("  • 'Ramesh bhaiya ke khate me 250 rupaye likh do'  (Add Credit)")
    print("  • 'Suresh ne 100 rupaye jama kiye'               (Record Payment)")
    print("  • 'Gupta ji ka kitna baaki hai?'                 (Balance Inquiry)")

    cmd = input("\nEnter voice transcript: ").strip()
    if not cmd:
        cmd = "Ramesh bhaiya ke khate me 250 rupaye likh do"
        print(f"Using default benchmark command: \"{cmd}\"")

    res = process_khata_voice_command(cmd, ledger=ledger)
    print(f"\n[Detected Intent]: {res['intent']}")
    print(f"[Spoken Response]: \"{res['spoken_confirmation']}\"")
    print(f"[Audio Saved]: {res['audio_path']}")


def action_check_balance(ledger: KiranaKhataLedger) -> None:
    print("\n--- [Option 3: Check Customer Khata Balance] ---")
    name = input("Enter customer name (e.g. Ramesh Bhaiya, Suresh Kumar, Gupta Ji): ").strip() or "Ramesh Bhaiya"
    data = ledger.get_customer_balance(name)

    print("\n" + "─" * 45)
    print(f"👤 Customer: {data['customer_name']} (Ph: {data['phone']})")
    print(f"💳 Current Khata Balance: ₹{data['current_balance']:.2f}")
    print(f"🔒 Credit Limit: ₹{data['credit_limit']:.2f}")
    print("─" * 45)
    if data["recent_entries"]:
        print("Recent Transactions:")
        for entry in data["recent_entries"]:
            print(f"  • {entry['timestamp'][:16]} | {entry['entry_type']:<6} | ₹{entry['amount']:<6.2f} | Balance: ₹{entry['balance_after']:.2f} ({entry['description']})")
    else:
        print("  (No recent transactions found)")
    print("─" * 45)


def action_noise_canceller_demo() -> None:
    print("\n--- [Option 4: Noise Canceller Award Demonstration] ---")
    demo_dir = Path("kirana_audio_demo")
    clean_audio = demo_dir / "clean_billing_speech.wav"
    noise_audio = demo_dir / "kirana_market_noise.wav"
    noisy_audio = demo_dir / "telecom_noisy_sim.wav"
    enhanced_audio = demo_dir / "telecom_enhanced_clean.wav"

    print("Step 1: Generating background Indian market crowd babble and traffic noise...")
    generate_synthetic_noise(4.0, 16000, "kirana_market", str(noise_audio))

    print("Step 2: Mixing market noise with clean speech at challenging +6.0 dB SNR...")
    mix_background_noise(clean_audio, noise_audio, noisy_audio, target_snr_db=6.0)

    print("Step 3: Downsampling to 8kHz AMR / PSTN Telephony standard (300Hz-3400Hz)...")
    simulated_8k = demo_dir / "noisy_phone_call_8k.wav"
    simulate_telephony_channel(noisy_audio, simulated_8k, target_sample_rate=8000)

    print("Step 4: Running DSP Telephony Noise Canceller (Berouti Spectral Subtraction + 50Hz/100Hz Notch)...")
    result = cancel_noise_file(simulated_8k, enhanced_audio)

    metrics = result["metrics"]
    print("\n" + "=" * 55)
    print("  🏆 NOISE CANCELLER AWARD - OBJECTIVE METRICS 🏆")
    print("=" * 55)
    print(f"  • Input Audio:           {simulated_8k.name} (8,000 Hz Narrowband Telephony)")
    print(f"  • Enhanced Audio:        {Path(enhanced_audio).name}")
    print(f"  • SNR Improvement:       +{metrics['snr_improvement_db']} dB")
    print(f"  • Noise Power Reduction: {metrics['noise_reduction_db']} dB")
    print("=" * 55)


def action_view_inventory() -> None:
    print("\n--- [Option 5: Store Inventory & Prices] ---")
    db = KiranaInventoryDB("inventory_db.json")
    print(f"{'Item Name':<32} | {'Unit':<8} | {'Price (INR)':<12} | {'Stock Qty'}")
    print("─" * 68)
    for item in db.inventory:
        print(f"{item['canonical_name']:<32} | {item['unit']:<8} | ₹{item['price_per_unit']:<11.2f} | {item.get('stock_qty', 'N/A')}")
    print("─" * 68)


def action_run_tests() -> None:
    print("\n--- [Option 6: Running All Automated Test Suites] ---")
    test_files = ["test_pipeline.py", "test_parser_engine.py", "test_khata_ledger.py"]
    for tf in test_files:
        print(f"\n>> Running {tf}...")
        code = os.system(f'"{sys.executable}" {tf}')
        if code == 0:
            print(f"  ✓ {tf} PASSED")
        else:
            print(f"  ❌ {tf} FAILED")


def main() -> None:
    ledger = KiranaKhataLedger()
    banner()

    while True:
        menu()
        choice = input("\nEnter choice (0-6): ").strip()
        if choice == "1":
            action_voice_order(ledger)
        elif choice == "2":
            action_voice_khata(ledger)
        elif choice == "3":
            action_check_balance(ledger)
        elif choice == "4":
            action_noise_canceller_demo()
        elif choice == "5":
            action_view_inventory()
        elif choice == "6":
            action_run_tests()
        elif choice == "0":
            print("\nDhanyawad! Exiting Kirana Mitra.\n")
            break
        else:
            print("Invalid option. Please enter a number between 0 and 6.")


if __name__ == "__main__":
    main()

