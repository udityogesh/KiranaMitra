"""
Unit Tests for Kirana Mitra Evon v3.3 Parser & Billing Engine (`parser_engine.py`)
==================================================================================
Tests JSON cleaning, Indic metric conversions, inventory fuzzy matching,
unit conversion calculations, and end-to-end voice order processing.
"""

import unittest
from pathlib import Path

from parser_engine import (
    _clean_and_extract_json,
    parse_transcript_to_json,
    calculate_bill,
    process_voice_order,
)


class TestParserEngine(unittest.TestCase):

    def test_json_cleaner_with_markdown_fences(self):
        """Test that _clean_and_extract_json strips markdown backticks and conversational text."""
        raw_llm_response = """
        Here is the parsed billing information for the customer:
        ```json
        {
          "items": [
            {"item_name": "Maggi", "quantity": 2.0, "unit": "packet"},
            {"item_name": "Dhaniya", "quantity": 0.25, "unit": "kg"}
          ],
          "confirmation_speech_hi": "Aapka 2 packet Maggi aur ek pav dhaniya note kar liya gaya hai."
        }
        ```
        Hope this helps!
        """
        cleaned = _clean_and_extract_json(raw_llm_response)
        self.assertIn("items", cleaned)
        self.assertEqual(len(cleaned["items"]), 2)
        self.assertEqual(cleaned["items"][0]["item_name"], "Maggi")
        self.assertEqual(cleaned["items"][0]["quantity"], 2.0)
        self.assertEqual(cleaned["items"][1]["quantity"], 0.25)
        self.assertEqual(cleaned["items"][1]["unit"], "kg")

    def test_colloquial_indic_metrics_extraction(self):
        """Test Indic metric extraction rules (aadha kilo, ek pav, dhai sau gram, peti, bori)."""
        # 1. "ek pav dhaniya" -> 0.25 kg
        res1 = parse_transcript_to_json("bhaiya ek pav dhaniya de do")
        self.assertEqual(len(res1["items"]), 1)
        self.assertEqual(res1["items"][0]["quantity"], 0.25)
        self.assertEqual(res1["items"][0]["unit"], "kg")

        # 2. "aadha kilo cheeni" -> 0.5 kg
        res2 = parse_transcript_to_json("aadha kilo cheeni pack karo")
        self.assertEqual(len(res2["items"]), 1)
        self.assertEqual(res2["items"][0]["quantity"], 0.5)
        self.assertEqual(res2["items"][0]["unit"], "kg")

        # 3. "dhai sau gram jeera" -> 250 g
        res3 = parse_transcript_to_json("dhai sau gram jeera dena")
        self.assertEqual(len(res3["items"]), 1)
        self.assertEqual(res3["items"][0]["quantity"], 250.0)
        self.assertEqual(res3["items"][0]["unit"], "g")

        # 4. "ek peti frooti" -> 1 box
        res4 = parse_transcript_to_json("ek peti frooti de do")
        self.assertEqual(len(res4["items"]), 1)
        self.assertEqual(res4["items"][0]["quantity"], 1.0)
        self.assertEqual(res4["items"][0]["unit"], "box")

        # 5. "ek bori chawal" -> 1 sack
        res5 = parse_transcript_to_json("ek bori basmati chawal")
        self.assertEqual(len(res5["items"]), 1)
        self.assertEqual(res5["items"][0]["quantity"], 1.0)
        self.assertEqual(res5["items"][0]["unit"], "sack")

    def test_inventory_matching_and_bill_calculation(self):
        """Test inventory lookup, fuzzy matching, and metric unit conversion math."""
        items = [
            {"item_name": "Maggi", "quantity": 2.0, "unit": "packet"},     # 2 * 14 = 28
            {"item_name": "Dhaniya", "quantity": 0.25, "unit": "kg"},       # 0.25 * 120 = 30
            {"item_name": "Jeera", "quantity": 250.0, "unit": "g"}         # 250g / 1000 * 320 = 80
        ]
        invoice, grand_total = calculate_bill(items)

        self.assertEqual(len(invoice), 3)
        self.assertEqual(invoice[0]["subtotal"], 28.0)
        self.assertEqual(invoice[1]["subtotal"], 30.0)
        self.assertEqual(invoice[2]["subtotal"], 80.0)
        self.assertEqual(grand_total, 138.0)
        self.assertTrue(all(line["status"] == "IN_STOCK" for line in invoice))

    def test_process_voice_order_end_to_end(self):
        """Verify process_voice_order runs and returns a valid invoice and confirmation."""
        demo_audio = Path("kirana_audio_demo") / "clean_billing_speech.wav"
        out_speech = Path("kirana_audio_demo") / "test_confirmation.wav"

        result = process_voice_order(
            audio_file_path=demo_audio,
            output_speech_path=out_speech,
            customer_name="Suresh Kumar",
            apply_noise_cleaning=False
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["customer"], "Suresh Kumar")
        self.assertGreater(result["grand_total_inr"], 0.0)
        self.assertTrue(out_speech.is_file())


if __name__ == "__main__":
    unittest.main()

