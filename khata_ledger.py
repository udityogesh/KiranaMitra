"""
Kirana Mitra - Persistent Khata (Credit Ledger) & Invoice Engine
================================================================
Gnani AI Great Indian AI Internship Challenge - Day 3

In Indian retail, over 70% of mom-and-pop grocery transactions run on credit
(Udhaar / Khata). This module provides an automated, voice-driven Khata ledger
backed by SQLite, complete with UPI QR payment links and WhatsApp receipts.

Core Capabilities:
  1. Persistent SQLite Database (`kirana_ledger.db`):
     - `customers`: Customer registry with contact details and running balances.
     - `ledger_entries`: Double-entry audit trail for all credits, debits, and payments.
     - `invoices`: Digital invoice records linking voice transcripts and line items.
  2. Voice Khata Command Dispatcher (`process_khata_voice_command`):
     - 'ADD_CREDIT' (Udhaar): "Ramesh bhaiya ke khate me 250 rupaye likh do"
     - 'RECORD_PAYMENT' (Jama): "Ramesh bhaiya ne 500 rupaye jama kar diye"
     - 'BALANCE_INQUIRY' (Baaki): "Ramesh ka kitna baaki hai?"
     - 'VOICE_BILLING' (Bikri): "Do packet maggi aur ek pav dhaniya"
  3. WhatsApp Digital Receipt & UPI Deep Link Generator:
     Generates professional formatted receipts with `upi://pay` links for instant settlement.

Dependencies:
    sqlite3 (standard library), typing, json, pathlib
"""

import re
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union

# Import NLU and Audio pipelines
from parser_engine import parse_transcript_to_json, calculate_bill
from gnani_pipeline import GnaniConfig, synthesize_speech

logger = logging.getLogger("KiranaMitra.KhataLedger")


class KiranaKhataLedger:
    """
    Persistent voice-enabled Khata ledger and billing store.
    """

    def __init__(self, db_path: Union[str, Path] = "kirana_ledger.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create relational schema if not already present."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Customers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    phone TEXT,
                    current_balance REAL DEFAULT 0.0,
                    credit_limit REAL DEFAULT 5000.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Ledger audit log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ledger_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id INTEGER NOT NULL,
                    entry_type TEXT NOT NULL, -- 'CREDIT' (Udhaar), 'DEBIT' (Paid/Jama)
                    amount REAL NOT NULL,
                    balance_after REAL NOT NULL,
                    description TEXT,
                    voice_transcript TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(customer_id) REFERENCES customers(id)
                )
            """)

            # Invoices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS invoices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    invoice_number TEXT UNIQUE NOT NULL,
                    customer_id INTEGER,
                    total_amount REAL NOT NULL,
                    items_json TEXT NOT NULL,
                    payment_status TEXT NOT NULL, -- 'PAID', 'KHATA_CREDIT'
                    voice_transcript TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(customer_id) REFERENCES customers(id)
                )
            """)

            # Seed common default customers if table is empty
            cursor.execute("SELECT COUNT(*) FROM customers")
            if cursor.fetchone()[0] == 0:
                initial_customers = [
                    ("Ramesh Bhaiya", "9876543210", 450.0),
                    ("Suresh Kumar", "9812345678", 120.0),
                    ("Gupta Ji", "9899887766", 0.0),
                    ("Mishra Ji", "9711223344", 850.0),
                ]
                for name, phone, bal in initial_customers:
                    cursor.execute(
                        "INSERT INTO customers (name, phone, current_balance) VALUES (?, ?, ?)",
                        (name, phone, bal)
                    )
            conn.commit()

    def get_or_create_customer(self, name: str, phone: Optional[str] = None) -> Dict[str, Any]:
        """Lookup customer by name or register a new customer."""
        clean_name = name.strip().title()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM customers WHERE LOWER(name) = LOWER(?)", (clean_name,))
            row = cursor.fetchone()
            if row:
                return dict(row)

            # Insert new customer
            cursor.execute(
                "INSERT INTO customers (name, phone, current_balance) VALUES (?, ?, 0.0)",
                (clean_name, phone or "9800000000")
            )
            conn.commit()
            cursor.execute("SELECT * FROM customers WHERE id = ?", (cursor.lastrowid,))
            return dict(cursor.fetchone())

    def record_khata_credit(
        self,
        customer_name: str,
        amount: float,
        description: str = "Voice Khata Entry",
        voice_transcript: str = ""
    ) -> Dict[str, Any]:
        """
        Record credit (Udhaar added to customer balance).
        Customer balance increases (they owe more money to the Kirana).
        """
        customer = self.get_or_create_customer(customer_name)
        new_balance = round(customer["current_balance"] + amount, 2)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE customers SET current_balance = ? WHERE id = ?",
                (new_balance, customer["id"])
            )
            cursor.execute("""
                INSERT INTO ledger_entries (customer_id, entry_type, amount, balance_after, description, voice_transcript)
                VALUES (?, 'CREDIT', ?, ?, ?, ?)
            """, (customer["id"], amount, new_balance, description, voice_transcript))
            conn.commit()

        logger.info(
            "Khata Credit: %s +₹%.2f | New Balance: ₹%.2f",
            customer["name"], amount, new_balance
        )
        return {
            "success": True,
            "customer_name": customer["name"],
            "transaction_type": "CREDIT",
            "amount": amount,
            "previous_balance": customer["current_balance"],
            "new_balance": new_balance,
            "description": description
        }

    def record_khata_payment(
        self,
        customer_name: str,
        amount: float,
        description: str = "Voice Payment Settlement",
        voice_transcript: str = ""
    ) -> Dict[str, Any]:
        """
        Record payment received (Jama / settling udhaar).
        Customer balance decreases.
        """
        customer = self.get_or_create_customer(customer_name)
        new_balance = round(max(0.0, customer["current_balance"] - amount), 2)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE customers SET current_balance = ? WHERE id = ?",
                (new_balance, customer["id"])
            )
            cursor.execute("""
                INSERT INTO ledger_entries (customer_id, entry_type, amount, balance_after, description, voice_transcript)
                VALUES (?, 'DEBIT', ?, ?, ?, ?)
            """, (customer["id"], amount, new_balance, description, voice_transcript))
            conn.commit()

        logger.info(
            "Khata Payment: %s -₹%.2f | Remaining Balance: ₹%.2f",
            customer["name"], amount, new_balance
        )
        return {
            "success": True,
            "customer_name": customer["name"],
            "transaction_type": "PAYMENT",
            "amount": amount,
            "previous_balance": customer["current_balance"],
            "remaining_balance": new_balance,
            "description": description
        }

    def get_customer_balance(self, customer_name: str) -> Dict[str, Any]:
        """Retrieve current outstanding khata balance for a customer."""
        customer = self.get_or_create_customer(customer_name)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM ledger_entries WHERE customer_id = ? ORDER BY timestamp DESC LIMIT 5
            """, (customer["id"],))
            recent_entries = [dict(r) for r in cursor.fetchall()]

        return {
            "customer_name": customer["name"],
            "phone": customer["phone"],
            "current_balance": customer["current_balance"],
            "credit_limit": customer["credit_limit"],
            "recent_entries": recent_entries
        }

    def record_invoice(
        self,
        customer_name: str,
        invoice_lines: List[Dict[str, Any]],
        total_amount: float,
        payment_status: str = "PAID",
        voice_transcript: str = ""
    ) -> Dict[str, Any]:
        """Record an itemized digital retail invoice."""
        customer = self.get_or_create_customer(customer_name)
        invoice_num = f"KM-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO invoices (invoice_number, customer_id, total_amount, items_json, payment_status, voice_transcript)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (invoice_num, customer["id"], total_amount, json.dumps(invoice_lines), payment_status, voice_transcript))
            conn.commit()

        # If sold on khata credit, automatically update ledger balance
        if payment_status == "KHATA_CREDIT":
            self.record_khata_credit(
                customer_name=customer_name,
                amount=total_amount,
                description=f"Invoice #{invoice_num}",
                voice_transcript=voice_transcript
            )

        whatsapp_receipt = self.generate_whatsapp_receipt(
            invoice_num=invoice_num,
            customer_name=customer["name"],
            invoice_lines=invoice_lines,
            total_amount=total_amount,
            payment_status=payment_status
        )

        return {
            "invoice_number": invoice_num,
            "customer_name": customer["name"],
            "total_amount": total_amount,
            "payment_status": payment_status,
            "invoice_lines": invoice_lines,
            "whatsapp_receipt": whatsapp_receipt
        }

    def generate_whatsapp_receipt(
        self,
        invoice_num: str,
        customer_name: str,
        invoice_lines: List[Dict[str, Any]],
        total_amount: float,
        payment_status: str,
        store_name: str = "Gupta Kirana Store",
        upi_id: str = "guptakirana@okaxis"
    ) -> str:
        """
        Generate a WhatsApp-ready digital receipt with instant UPI payment deep link.
        """
        date_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
        lines = [
            f"🛒 *{store_name.upper()}*",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"🧾 *Bill No:* `{invoice_num}`",
            f"📅 *Date:* {date_str}",
            f"👤 *Customer:* {customer_name}",
            f"💳 *Status:* {'✅ PAID' if payment_status == 'PAID' else '⏳ KHATA (UDHAAR)'}",
            "──────────────────────",
            "*ITEMS ORDERED:*"
        ]

        for item in invoice_lines:
            name = item["matched_name"]
            qty = item["ordered_quantity"]
            unit = item["ordered_unit"]
            sub = item["subtotal"]
            lines.append(f"• {name} ({qty} {unit}) : ₹{sub:.2f}")

        lines.extend([
            "──────────────────────",
            f"💰 *TOTAL AMOUNT:* ₹{total_amount:.2f}",
        ])

        # If sold on credit, include current total outstanding khata balance
        cust_bal = self.get_customer_balance(customer_name)["current_balance"]
        if payment_status == "KHATA_CREDIT":
            lines.append(f"📌 *Kul Baaki (Khata Balance):* ₹{cust_bal:.2f}")

        # UPI Deep Link for 1-click customer payment
        encoded_store = store_name.replace(" ", "%20")
        upi_link = f"upi://pay?pa={upi_id}&pn={encoded_store}&am={total_amount:.2f}&cu=INR"
        lines.extend([
            "━━━━━━━━━━━━━━━━━━━━━━",
            "📲 *Pay Online via UPI:*",
            f"{upi_link}",
            "\n🙏 _Dhanyawad! Phir Padhaariye._"
        ])

        return "\n".join(lines)


# ============================================================================
# VOICE KHATA COMMAND DISPATCHER & NLU INTERPRETER
# ============================================================================
def process_khata_voice_command(
    transcript: str,
    ledger: Optional[KiranaKhataLedger] = None,
    output_speech_path: str = "khata_speech_response.wav"
) -> Dict[str, Any]:
    """
    Intelligently interpret colloquial Hindi/Hinglish ledger voice commands:
      - 'ADD_CREDIT': "Ramesh bhaiya ke khate me 250 rupaye likh lo"
      - 'RECORD_PAYMENT': "Ramesh ne 500 rupaye jama kar diye"
      - 'BALANCE_INQUIRY': "Ramesh ka kitna baaki hai?"
      - 'VOICE_ORDER': "Do packet maggi aur ek pav dhaniya de do"

    Args:
        transcript: Transcribed speech from Gnani Prisma STT.
        ledger: KiranaKhataLedger instance.
        output_speech_path: Where to save the spoken confirmation audio.

    Returns:
        Structured result dictionary.
    """
    ledger = ledger or KiranaKhataLedger()
    clean_t = transcript.strip().lower()

    # Rule 1: Extract customer name if mentioned (e.g. "Ramesh", "Suresh", "Gupta ji")
    known_customers = ["ramesh bhaiya", "ramesh", "suresh kumar", "suresh", "gupta ji", "mishra ji"]
    matched_customer = "Grahak"
    for kc in known_customers:
        if kc in clean_t:
            matched_customer = kc.title()
            break

    # Rule 2: Extract rupee amount mentioned in the transcript
    amount_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:rupaye|rs|rupya|rupee|inr)", clean_t)
    amount = float(amount_match.group(1)) if amount_match else None

    # Fallback number words if digits not used: "dhai sau" -> 250, "paanch sau" -> 500, "sau" -> 100
    if amount is None:
        if "dhai sau" in clean_t:
            amount = 250.0
        elif "dedh sau" in clean_t:
            amount = 150.0
        elif "paanch sau" in clean_t:
            amount = 500.0
        elif "hazaar" in clean_t or "hazar" in clean_t:
            amount = 1000.0

    # -------------------------------------------------------------
    # INTENT A: RECORD PAYMENT (Jama / Settling Udhaar)
    # Keywords: "jama", "paid", "de diye", "wapas kiye", "bhugtan"
    # -------------------------------------------------------------
    if any(k in clean_t for k in ("jama", "paid", "bhugtan", "de diye")) and amount:
        res = ledger.record_khata_payment(
            customer_name=matched_customer,
            amount=amount,
            description="Voice Khata Payment",
            voice_transcript=transcript
        )
        spoken_hi = (
            f"{matched_customer} ji ka {int(amount)} rupaye jama kar liya gaya hai. "
            f"Ab bacha hua baaki khata {int(res['remaining_balance'])} rupaye hai."
        )
        _generate_tts_confirmation(spoken_hi, output_speech_path)
        return {
            "intent": "RECORD_PAYMENT",
            "details": res,
            "spoken_confirmation": spoken_hi,
            "audio_path": output_speech_path
        }

    # -------------------------------------------------------------
    # INTENT B: ADD CREDIT (Udhaar Entry)
    # Keywords: "khate me", "khata", "udhaar", "likh lo", "likh do", "jodh do"
    # -------------------------------------------------------------
    elif any(k in clean_t for k in ("khate me", "khata", "udhaar", "likh lo", "likh do", "jodh do")) and amount:
        res = ledger.record_khata_credit(
            customer_name=matched_customer,
            amount=amount,
            description="Voice Udhaar Entry",
            voice_transcript=transcript
        )
        spoken_hi = (
            f"{matched_customer} ji ke khate me {int(amount)} rupaye likh diye gaye hain. "
            f"Ab kul baaki {int(res['new_balance'])} rupaye ho gaya hai."
        )
        _generate_tts_confirmation(spoken_hi, output_speech_path)
        return {
            "intent": "ADD_CREDIT",
            "details": res,
            "spoken_confirmation": spoken_hi,
            "audio_path": output_speech_path
        }

    # -------------------------------------------------------------
    # INTENT C: BALANCE INQUIRY (Kitna Baaki Hai?)
    # Keywords: "kitna baaki", "balance", "hisaab", "kitna hua", "udhari"
    # -------------------------------------------------------------
    elif any(k in clean_t for k in ("kitna baaki", "balance", "hisaab", "kitna hua")):
        bal_res = ledger.get_customer_balance(matched_customer)
        curr_bal = bal_res["current_balance"]
        spoken_hi = (
            f"{matched_customer} ji ka kul baaki khata {int(curr_bal)} rupaye hai."
        )
        _generate_tts_confirmation(spoken_hi, output_speech_path)
        return {
            "intent": "BALANCE_INQUIRY",
            "details": bal_res,
            "spoken_confirmation": spoken_hi,
            "audio_path": output_speech_path
        }

    # -------------------------------------------------------------
    # INTENT D: RETAIL VOICE ORDER BILLING (Default)
    # -------------------------------------------------------------
    else:
        parsed = parse_transcript_to_json(transcript)
        invoice_lines, grand_total = calculate_bill(parsed.get("items", []))
        invoice_record = ledger.record_invoice(
            customer_name=matched_customer,
            invoice_lines=invoice_lines,
            total_amount=grand_total,
            payment_status="PAID",
            voice_transcript=transcript
        )
        spoken_hi = (
            f"{matched_customer} ji, aapka bill ban gaya hai. Kul rashi {int(grand_total)} rupaye hai."
        )
        _generate_tts_confirmation(spoken_hi, output_speech_path)
        return {
            "intent": "VOICE_BILLING",
            "details": invoice_record,
            "spoken_confirmation": spoken_hi,
            "audio_path": output_speech_path
        }


def _generate_tts_confirmation(text: str, output_path: str) -> str:
    """Generate spoken audio confirmation using Gnani Timbre TTS."""
    try:
        return synthesize_speech(text=text, output_path=output_path, language="hin_eng")
    except Exception as e:
        logger.warning("TTS generation skipped: %s", e)
        return output_path

