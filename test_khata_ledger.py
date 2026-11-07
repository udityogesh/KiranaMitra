"""
Unit Tests for Kirana Mitra Khata (Credit Ledger) & Voice Engine (`khata_ledger.py`)
===================================================================================
Tests customer creation, credit (udhaar), debit (jama), balance inquiry,
and WhatsApp receipt generation with UPI deep links.
"""

import unittest
from pathlib import Path

from khata_ledger import KiranaKhataLedger, process_khata_voice_command


class TestKhataLedger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_db = Path("test_kirana_ledger_suite.db")
        if cls.test_db.exists():
            try:
                cls.test_db.unlink()
            except PermissionError:
                pass
        cls.ledger = KiranaKhataLedger(cls.test_db)

    @classmethod
    def tearDownClass(cls):
        if cls.test_db.exists():
            try:
                cls.test_db.unlink()
            except PermissionError:
                pass

    def test_1_record_khata_credit_and_balance(self):
        """Test recording udhaar (credit) and verifying balance increase."""
        res = self.ledger.record_khata_credit(
            customer_name="Ramesh Bhaiya",
            amount=250.0,
            description="Test Udhaar"
        )
        self.assertTrue(res["success"])
        # Default seed balance for Ramesh Bhaiya is 450.0; after +250.0 it should be 700.0
        self.assertEqual(res["new_balance"], 700.0)

        bal = self.ledger.get_customer_balance("Ramesh Bhaiya")
        self.assertEqual(bal["current_balance"], 700.0)

    def test_2_record_khata_payment(self):
        """Test recording jama (payment) and verifying balance decrease."""
        res = self.ledger.record_khata_payment(
            customer_name="Ramesh Bhaiya",
            amount=200.0,
            description="Partial payment"
        )
        self.assertTrue(res["success"])
        # 700 - 200 = 500
        self.assertEqual(res["remaining_balance"], 500.0)

    def test_3_voice_khata_command_credit(self):
        """Test voice command dispatch for adding credit."""
        transcript = "Ramesh bhaiya ke khate me 150 rupaye likh lo"
        res = process_khata_voice_command(transcript, ledger=self.ledger)
        self.assertEqual(res["intent"], "ADD_CREDIT")
        self.assertEqual(res["details"]["amount"], 150.0)
        self.assertIn("likh diye gaye hain", res["spoken_confirmation"])

    def test_4_voice_khata_command_payment(self):
        """Test voice command dispatch for recording payment."""
        transcript = "Suresh ne 100 rupaye jama kiye"
        res = process_khata_voice_command(transcript, ledger=self.ledger)
        self.assertEqual(res["intent"], "RECORD_PAYMENT")
        self.assertEqual(res["details"]["amount"], 100.0)
        self.assertIn("jama kar liya gaya hai", res["spoken_confirmation"])

    def test_5_whatsapp_receipt_generation(self):
        """Test formatting of WhatsApp invoice with UPI deep link."""
        invoice_lines = [
            {"matched_name": "Maggi 2-Minute Noodles", "ordered_quantity": 2.0, "ordered_unit": "packet", "subtotal": 28.0},
            {"matched_name": "Fresh Dhaniya (Coriander)", "ordered_quantity": 0.25, "ordered_unit": "kg", "subtotal": 30.0}
        ]
        rec = self.ledger.record_invoice(
            customer_name="Ramesh Bhaiya",
            invoice_lines=invoice_lines,
            total_amount=58.0,
            payment_status="KHATA_CREDIT"
        )
        wa = rec["whatsapp_receipt"]
        self.assertIn("GUPTA KIRANA STORE", wa)
        self.assertIn("upi://pay", wa)
        self.assertIn("₹58.00", wa)


if __name__ == "__main__":
    unittest.main()
