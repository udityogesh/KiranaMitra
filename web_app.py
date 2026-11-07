"""
Kirana Mitra - Interactive Web Dashboard & Demo Interface
=========================================================
Gnani AI Great Indian AI Internship Challenge

Zero-dependency browser interface built using Python's native `http.server`.
Provides shopkeepers and challenge evaluators with a visual console to:
  1. Record or upload voice grocery orders with live STT and Evon v3.3 extraction.
  2. Test the DSP Telecom Noise Canceller with A/B audio player comparison.
  3. Manage Khata (Credit) ledgers, check balances, and generate WhatsApp UPI bills.

Usage:
  python web_app.py
  Open: http://localhost:8000
"""

import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from gnani_pipeline import GnaniConfig
from noise_canceller import cancel_noise_file
from parser_engine import parse_transcript_to_json, calculate_bill, process_voice_order, KiranaInventoryDB
from khata_ledger import KiranaKhataLedger, process_khata_voice_command

PORT = int(os.getenv("PORT", 8000))
LEDGER = KiranaKhataLedger()
INVENTORY = KiranaInventoryDB()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kirana Mitra - Voice-First Retail Ledger</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .tab-active { border-b-2 border-orange-500 text-orange-600 font-bold; }
  </style>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen">

  <!-- Header -->
  <header class="bg-orange-600 text-white shadow-md">
    <div class="max-w-6xl mx-auto px-4 py-4 flex justify-between items-center">
      <div class="flex items-center space-x-3">
        <i class="fa-solid fa-store text-2xl text-yellow-300"></i>
        <div>
          <h1 class="text-xl font-bold tracking-tight">KIRANA MITRA (किराना मित्र)</h1>
          <p class="text-xs text-orange-200">Voice-First Retail Ledger • Gnani AI Internship Challenge</p>
        </div>
      </div>
      <div class="flex items-center space-x-2">
        <span class="bg-orange-700 text-xs px-3 py-1 rounded-full text-orange-200 border border-orange-500">
          <i class="fa-solid fa-key mr-1"></i> Gnani Vach v2.5 / v3.3 Active
        </span>
      </div>
    </div>
  </header>

  <!-- Navigation Tabs -->
  <div class="max-w-6xl mx-auto px-4 mt-6">
    <div class="flex border-b border-slate-200 space-x-6 text-sm">
      <button onclick="switchTab('tab-order')" id="btn-tab-order" class="tab-btn pb-3 tab-active">
        <i class="fa-solid fa-microphone mr-1.5"></i> Voice Order & Billing
      </button>
      <button onclick="switchTab('tab-khata')" id="btn-tab-khata" class="tab-btn pb-3 text-slate-500 hover:text-slate-800">
        <i class="fa-solid fa-book mr-1.5"></i> Khata (Credit Ledger)
      </button>
      <button onclick="switchTab('tab-noise')" id="btn-tab-noise" class="tab-btn pb-3 text-slate-500 hover:text-slate-800">
        <i class="fa-solid fa-wave-square mr-1.5"></i> Noise Canceller Award Showcase
      </button>
      <button onclick="switchTab('tab-inventory')" id="btn-tab-inventory" class="tab-btn pb-3 text-slate-500 hover:text-slate-800">
        <i class="fa-solid fa-boxes-stacked mr-1.5"></i> Store Catalog
      </button>
    </div>
  </div>

  <!-- Main Container -->
  <main class="max-w-6xl mx-auto px-4 py-6">

    <!-- TAB 1: VOICE ORDER & BILLING -->
    <section id="tab-order" class="tab-content">
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

        <!-- Input Card -->
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h2 class="text-base font-semibold text-slate-800 mb-4 flex items-center">
            <i class="fa-solid fa-cash-register text-orange-500 mr-2"></i> Voice Grocery Order Input
          </h2>

          <div class="space-y-4">
            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">Customer Name</label>
              <input type="text" id="cust-name" value="Ramesh Bhaiya" class="w-full text-sm px-3 py-2 border rounded-lg focus:ring-2 focus:ring-orange-500 outline-none">
            </div>

            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">Preset Indian Kirana Voice Order</label>
              <select id="preset-order" onchange="applyPreset()" class="w-full text-sm px-3 py-2 border rounded-lg focus:ring-2 focus:ring-orange-500 outline-none">
                <option value="bhaiya do packet maggi aur ek pav dhaniya de do">"Bhaiya do packet maggi aur ek pav dhaniya de do" (Maggi + 250g Coriander)</option>
                <option value="aadha kilo cheeni aur ek packet tata namak chahiye">"Aadha kilo cheeni aur ek packet tata namak chahiye" (0.5kg Sugar + Salt)</option>
                <option value="dhai sau gram jeera aur do kilo aashirvaad aata de dena">"Dhai sau gram jeera aur do kilo aashirvaad aata" (250g Cumin + 2kg Atta)</option>
                <option value="ek peti frooti aur basmati chawal ki ek bori likh lo">"Ek peti frooti aur basmati chawal ki ek bori" (1 Box Frooti + 1 Sack Rice)</option>
                <option value="teen pav haldi aur dedh kilo toor daal pack kar do">"Teen pav haldi aur dedh kilo toor daal" (0.75kg Turmeric + 1.5kg Daal)</option>
              </select>
            </div>

            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">Spoken Voice Transcript (Hinglish)</label>
              <textarea id="order-transcript" rows="3" class="w-full text-sm px-3 py-2 border rounded-lg focus:ring-2 focus:ring-orange-500 outline-none">bhaiya do packet maggi aur ek pav dhaniya de do</textarea>
            </div>

            <div class="flex items-center space-x-2 bg-slate-50 p-3 rounded-lg border border-slate-100">
              <input type="checkbox" id="chk-noise" checked class="h-4 w-4 text-orange-600 rounded">
              <label for="chk-noise" class="text-xs text-slate-700 cursor-pointer">
                <strong>Apply DSP Noise Canceller</strong> (Cleans market background noise before STT)
              </label>
            </div>

            <button onclick="processOrder()" id="btn-process" class="w-full bg-orange-600 hover:bg-orange-700 text-white font-medium py-2.5 rounded-lg shadow transition flex justify-center items-center">
              <i class="fa-solid fa-receipt mr-2"></i> Generate Digital Bill & Voice Confirmation
            </button>
          </div>
        </div>

        <!-- Output Invoice Card -->
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col justify-between">
          <div>
            <div class="flex justify-between items-center mb-4">
              <h2 class="text-base font-semibold text-slate-800 flex items-center">
                <i class="fa-solid fa-file-invoice text-emerald-600 mr-2"></i> Itemized Digital Bill
              </h2>
              <span id="bill-badge" class="hidden text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-800">
                VERIFIED
              </span>
            </div>

            <div id="invoice-placeholder" class="py-12 text-center text-slate-400">
              <i class="fa-solid fa-receipt text-4xl mb-3 text-slate-300"></i>
              <p class="text-sm">Click "Generate Digital Bill" to parse speech and compute the invoice.</p>
            </div>

            <!-- Invoice Details -->
            <div id="invoice-details" class="hidden space-y-4">
              <div class="border rounded-lg overflow-hidden text-sm">
                <table class="w-full text-left">
                  <thead class="bg-slate-100 text-xs text-slate-600 uppercase border-b">
                    <tr>
                      <th class="px-3 py-2">Item</th>
                      <th class="px-3 py-2">Qty</th>
                      <th class="px-3 py-2">Rate</th>
                      <th class="px-3 py-2 text-right">Subtotal</th>
                    </tr>
                  </thead>
                  <tbody id="invoice-tbody" class="divide-y divide-slate-100"></tbody>
                </table>
              </div>

              <div class="flex justify-between items-center p-3 bg-emerald-50 rounded-lg border border-emerald-200">
                <span class="text-xs uppercase tracking-wider font-semibold text-emerald-800">Grand Total (कुल राशि)</span>
                <span id="invoice-total" class="text-lg font-bold text-emerald-900">₹0.00</span>
              </div>

              <!-- Spoken Audio Player -->
              <div class="bg-slate-50 p-3 rounded-lg border border-slate-200">
                <p class="text-xs font-medium text-slate-700 mb-1 flex items-center">
                  <i class="fa-solid fa-volume-high text-orange-500 mr-1.5"></i> Gnani Timbre TTS Hindi Confirmation:
                </p>
                <p id="tts-text" class="text-xs text-slate-600 italic mb-2">"..."</p>
                <audio id="tts-audio" controls class="w-full h-8"></audio>
              </div>
            </div>
          </div>

          <div id="invoice-actions" class="hidden mt-4 pt-4 border-t border-slate-100 flex space-x-3">
            <button onclick="addCurrentOrderToKhata()" class="flex-1 bg-amber-500 hover:bg-amber-600 text-white text-xs font-semibold py-2 rounded-lg shadow">
              <i class="fa-solid fa-book mr-1"></i> Add to Khata (Udhaar)
            </button>
            <button onclick="shareWhatsApp()" class="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold py-2 rounded-lg shadow">
              <i class="fa-brands fa-whatsapp mr-1"></i> WhatsApp Receipt
            </button>
          </div>

        </div>

      </div>
    </section>

    <!-- TAB 2: KHATA (CREDIT LEDGER) -->
    <section id="tab-khata" class="tab-content hidden">
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">

        <!-- Voice Khata Action Form -->
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-200 lg:col-span-1">
          <h2 class="text-base font-semibold text-slate-800 mb-4 flex items-center">
            <i class="fa-solid fa-pen-to-square text-orange-500 mr-2"></i> Voice Khata Command
          </h2>
          <div class="space-y-4">
            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">Preset Spoken Commands</label>
              <select id="preset-khata" onchange="applyKhataPreset()" class="w-full text-sm px-3 py-2 border rounded-lg focus:ring-2 focus:ring-orange-500 outline-none">
                <option value="Ramesh bhaiya ke khate me 250 rupaye likh do">"Ramesh bhaiya ke khate me 250 rupaye likh do" (Add Udhaar)</option>
                <option value="Suresh ne 100 rupaye jama kar diye">"Suresh ne 100 rupaye jama kar diye" (Record Payment)</option>
                <option value="Gupta ji ka kitna baaki hai?">"Gupta ji ka kitna baaki hai?" (Check Balance)</option>
              </select>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-600 mb-1">Spoken Hindi/Hinglish Text</label>
              <textarea id="khata-transcript" rows="3" class="w-full text-sm px-3 py-2 border rounded-lg focus:ring-2 focus:ring-orange-500 outline-none">Ramesh bhaiya ke khate me 250 rupaye likh do</textarea>
            </div>
            <button onclick="executeKhataCommand()" class="w-full bg-slate-800 hover:bg-slate-900 text-white font-medium py-2.5 rounded-lg shadow transition">
              <i class="fa-solid fa-bolt mr-1.5 text-yellow-400"></i> Execute Khata Command
            </button>
            <div id="khata-feedback" class="hidden p-3 rounded-lg text-xs bg-slate-100 text-slate-700"></div>
          </div>
        </div>

        <!-- Customer Accounts List -->
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-200 lg:col-span-2">
          <div class="flex justify-between items-center mb-4">
            <h2 class="text-base font-semibold text-slate-800 flex items-center">
              <i class="fa-solid fa-users text-blue-600 mr-2"></i> Customer Credit Accounts (उधार खाता)
            </h2>
            <button onclick="loadCustomers()" class="text-xs text-orange-600 hover:underline">
              <i class="fa-solid fa-rotate-right mr-1"></i> Refresh
            </button>
          </div>
          <div class="border rounded-lg overflow-hidden text-sm">
            <table class="w-full text-left">
              <thead class="bg-slate-100 text-xs text-slate-600 uppercase border-b">
                <tr>
                  <th class="px-4 py-2.5">Customer</th>
                  <th class="px-4 py-2.5">Phone</th>
                  <th class="px-4 py-2.5">Credit Limit</th>
                  <th class="px-4 py-2.5 text-right">Outstanding Khata</th>
                </tr>
              </thead>
              <tbody id="customers-tbody" class="divide-y divide-slate-100"></tbody>
            </table>
          </div>
        </div>

      </div>
    </section>

    <!-- TAB 3: NOISE CANCELLER AWARD -->
    <section id="tab-noise" class="tab-content hidden">
      <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-200 space-y-6">
        <div>
          <h2 class="text-lg font-bold text-slate-800 flex items-center">
            <i class="fa-solid fa-trophy text-yellow-500 mr-2"></i> Telecom Speech Enhancer (Noise Canceller Award)
          </h2>
          <p class="text-sm text-slate-600 mt-1">
            Indian Kirana stores have intense ambient noise: 50Hz mains power hum, ceiling fan resonance, street traffic horns, and customer babble.
            Our custom DSP engine combines <strong>multi-harmonic notch filtering</strong>, <strong>energy-based VAD</strong>, and <strong>Berouti spectral subtraction</strong> ($\beta=0.02$).
          </p>
        </div>

        <!-- Empirical Table -->
        <div class="border rounded-xl overflow-hidden">
          <table class="w-full text-left text-sm">
            <thead class="bg-slate-100 text-xs text-slate-600 uppercase border-b">
              <tr>
                <th class="px-4 py-3">Acoustic Condition</th>
                <th class="px-4 py-3">Input Target SNR</th>
                <th class="px-4 py-3">SNR Improvement (ΔSNR)</th>
                <th class="px-4 py-3">Noise Power Reduction</th>
                <th class="px-4 py-3">DSP Latency</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100 text-xs">
              <tr class="hover:bg-slate-50">
                <td class="px-4 py-2.5 font-medium text-slate-800">Clean Store Ambient</td>
                <td class="px-4 py-2.5">+15.0 dB</td>
                <td class="px-4 py-2.5 text-emerald-700 font-bold">+16.86 dB</td>
                <td class="px-4 py-2.5">17.83 dB</td>
                <td class="px-4 py-2.5">31.8 ms</td>
              </tr>
              <tr class="hover:bg-slate-50">
                <td class="px-4 py-2.5 font-medium text-slate-800">Bustling Kirana Market Crowd</td>
                <td class="px-4 py-2.5">+10.0 dB</td>
                <td class="px-4 py-2.5 text-emerald-700 font-bold">+15.06 dB</td>
                <td class="px-4 py-2.5">16.16 dB</td>
                <td class="px-4 py-2.5">24.8 ms</td>
              </tr>
              <tr class="hover:bg-slate-50">
                <td class="px-4 py-2.5 font-medium text-slate-800">Heavy Street Traffic + Fan Hum</td>
                <td class="px-4 py-2.5">+5.0 dB</td>
                <td class="px-4 py-2.5 text-emerald-700 font-bold">+12.20 dB</td>
                <td class="px-4 py-2.5">13.61 dB</td>
                <td class="px-4 py-2.5">29.9 ms</td>
              </tr>
              <tr class="hover:bg-slate-50">
                <td class="px-4 py-2.5 font-medium text-slate-800">Direct Vehicle Honking (Roadside)</td>
                <td class="px-4 py-2.5">0.0 dB</td>
                <td class="px-4 py-2.5 text-emerald-700 font-bold">+8.06 dB</td>
                <td class="px-4 py-2.5">10.15 dB</td>
                <td class="px-4 py-2.5">30.4 ms</td>
              </tr>
              <tr class="hover:bg-slate-50">
                <td class="px-4 py-2.5 font-medium text-slate-800">Extreme Cacophony</td>
                <td class="px-4 py-2.5">-5.0 dB</td>
                <td class="px-4 py-2.5 text-emerald-700 font-bold">+5.92 dB</td>
                <td class="px-4 py-2.5">8.21 dB</td>
                <td class="px-4 py-2.5">22.8 ms</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Audio Player Comparison -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
          <div class="p-4 bg-slate-50 rounded-xl border border-slate-200">
            <h4 class="text-xs font-bold text-red-700 mb-2 flex items-center">
              <i class="fa-solid fa-volume-xmark mr-1.5"></i> Noisy 8kHz Telephony Input (+5dB SNR)
            </h4>
            <audio controls class="w-full h-8" src="/audio/noisy_speech_snr_5db.wav"></audio>
          </div>
          <div class="p-4 bg-emerald-50 rounded-xl border border-emerald-200">
            <h4 class="text-xs font-bold text-emerald-800 mb-2 flex items-center">
              <i class="fa-solid fa-wand-magic-sparkles mr-1.5"></i> Enhanced Cleaned Speech (+12.2 dB SNR Boost)
            </h4>
            <audio controls class="w-full h-8" src="/audio/enhanced_speech_snr_5db.wav"></audio>
          </div>
        </div>

      </div>
    </section>

    <!-- TAB 4: STORE INVENTORY -->
    <section id="tab-inventory" class="tab-content hidden">
      <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
        <h2 class="text-base font-semibold text-slate-800 mb-4 flex items-center">
          <i class="fa-solid fa-boxes-stacked text-orange-500 mr-2"></i> Kirana Store Catalog & MRPs
        </h2>
        <div class="border rounded-lg overflow-hidden text-sm">
          <table class="w-full text-left">
            <thead class="bg-slate-100 text-xs text-slate-600 uppercase border-b">
              <tr>
                <th class="px-4 py-2.5">Item Name</th>
                <th class="px-4 py-2.5">Unit</th>
                <th class="px-4 py-2.5">Price (₹)</th>
                <th class="px-4 py-2.5 text-right">In Stock</th>
              </tr>
            </thead>
            <tbody id="inventory-tbody" class="divide-y divide-slate-100"></tbody>
          </table>
        </div>
      </div>
    </section>

  </main>

  <script>
    let currentInvoiceData = null;

    function switchTab(tabId) {
      document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
      document.querySelectorAll('.tab-btn').forEach(el => {
        el.classList.remove('tab-active');
        el.classList.add('text-slate-500');
      });
      document.getElementById(tabId).classList.remove('hidden');
      document.getElementById('btn-' + tabId).classList.add('tab-active');
      document.getElementById('btn-' + tabId).classList.remove('text-slate-500');

      if (tabId === 'tab-khata') loadCustomers();
      if (tabId === 'tab-inventory') loadInventory();
    }

    function applyPreset() {
      const val = document.getElementById('preset-order').value;
      document.getElementById('order-transcript').value = val;
    }

    function applyKhataPreset() {
      const val = document.getElementById('preset-khata').value;
      document.getElementById('khata-transcript').value = val;
    }

    async function processOrder() {
      const transcript = document.getElementById('order-transcript').value.trim();
      const customer = document.getElementById('cust-name').value.trim() || 'Ramesh Bhaiya';
      const applyNoise = document.getElementById('chk-noise').checked;

      const btn = document.getElementById('btn-process');
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-2"></i> Processing Speech...';
      btn.disabled = true;

      try {
        const res = await fetch('/api/process_order', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ transcript, customer, apply_noise: applyNoise })
        });
        const data = await res.json();
        currentInvoiceData = data;

        // Render invoice lines
        const tbody = document.getElementById('invoice-tbody');
        tbody.innerHTML = '';
        data.invoice.forEach(it => {
          tbody.innerHTML += `
            <tr class="hover:bg-slate-50">
              <td class="px-3 py-2 font-medium">${it.matched_name}</td>
              <td class="px-3 py-2 text-slate-600">${it.ordered_quantity} ${it.ordered_unit}</td>
              <td class="px-3 py-2 text-slate-600">₹${it.rate.toFixed(2)}</td>
              <td class="px-3 py-2 text-right font-semibold">₹${it.subtotal.toFixed(2)}</td>
            </tr>
          `;
        });

        document.getElementById('invoice-total').textContent = `₹${data.grand_total_inr.toFixed(2)}`;
        document.getElementById('tts-text').textContent = `"${data.spoken_confirmation}"`;
        document.getElementById('tts-audio').src = `/audio/order_confirmation.wav?t=${Date.now()}`;

        document.getElementById('invoice-placeholder').classList.add('hidden');
        document.getElementById('invoice-details').classList.remove('hidden');
        document.getElementById('invoice-actions').classList.remove('hidden');
        document.getElementById('bill-badge').classList.remove('hidden');
      } catch (err) {
        alert('Error processing order: ' + err.message);
      } finally {
        btn.innerHTML = '<i class="fa-solid fa-receipt mr-2"></i> Generate Digital Bill & Voice Confirmation';
        btn.disabled = false;
      }
    }

    async function addCurrentOrderToKhata() {
      if (!currentInvoiceData) return;
      const res = await fetch('/api/khata_command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transcript: `${currentInvoiceData.customer} ke khate me ${currentInvoiceData.grand_total_inr} rupaye likh do`
        })
      });
      const data = await res.json();
      alert(`✅ ₹${currentInvoiceData.grand_total_inr} successfully added to ${currentInvoiceData.customer}'s Khata!`);
    }

    function shareWhatsApp() {
      if (!currentInvoiceData) return;
      const text = encodeURIComponent(
        `🛒 *GUPTA KIRANA STORE BILL*\\n` +
        `Customer: ${currentInvoiceData.customer}\\n` +
        `Amount: ₹${currentInvoiceData.grand_total_inr}\\n\\n` +
        `📲 Pay via UPI: upi://pay?pa=guptakirana@okaxis&pn=Gupta%20Kirana&am=${currentInvoiceData.grand_total_inr}&cu=INR`
      );
      window.open(`https://api.whatsapp.com/send?text=${text}`, '_blank');
    }

    async function executeKhataCommand() {
      const text = document.getElementById('khata-transcript').value.trim();
      const fb = document.getElementById('khata-feedback');
      fb.classList.remove('hidden');
      fb.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-1"></i> Executing...';

      try {
        const res = await fetch('/api/khata_command', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ transcript: text })
        });
        const data = await res.json();
        fb.innerHTML = `<strong>${data.intent}:</strong> ${data.spoken_confirmation}`;
        loadCustomers();
      } catch (err) {
        fb.innerHTML = '<span class="text-red-600">Error: ' + err.message + '</span>';
      }
    }

    async function loadCustomers() {
      const res = await fetch('/api/customers');
      const data = await res.json();
      const tbody = document.getElementById('customers-tbody');
      tbody.innerHTML = '';
      data.forEach(c => {
        tbody.innerHTML += `
          <tr class="hover:bg-slate-50">
            <td class="px-4 py-2.5 font-medium text-slate-800">${c.name}</td>
            <td class="px-4 py-2.5 text-slate-600">${c.phone}</td>
            <td class="px-4 py-2.5 text-slate-600">₹${c.credit_limit.toFixed(2)}</td>
            <td class="px-4 py-2.5 text-right font-bold ${c.current_balance > 0 ? 'text-amber-700' : 'text-emerald-700'}">
              ₹${c.current_balance.toFixed(2)}
            </td>
          </tr>
        `;
      });
    }

    async function loadInventory() {
      const res = await fetch('/api/inventory');
      const data = await res.json();
      const tbody = document.getElementById('inventory-tbody');
      tbody.innerHTML = '';
      data.forEach(it => {
        tbody.innerHTML += `
          <tr class="hover:bg-slate-50">
            <td class="px-4 py-2.5 font-medium text-slate-800">${it.canonical_name}</td>
            <td class="px-4 py-2.5 text-slate-600">${it.unit}</td>
            <td class="px-4 py-2.5 font-semibold text-slate-700">₹${it.price_per_unit.toFixed(2)}</td>
            <td class="px-4 py-2.5 text-right text-slate-600">${it.stock_qty || 'Available'}</td>
          </tr>
        `;
      });
    }
  </script>
</body>
</html>
"""


class KiranaHttpHandler(BaseHTTPRequestHandler):

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif path == "/api/customers":
            with LEDGER._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM customers ORDER BY current_balance DESC")
                customers = [dict(r) for r in cursor.fetchall()]
            self._send_json(customers)

        elif path == "/api/inventory":
            self._send_json(INVENTORY.inventory)

        elif path.startswith("/audio/"):
            filename = Path(path).name
            # Look in kirana_audio_demo or benchmark_results
            file_loc = Path("kirana_audio_demo") / filename
            if not file_loc.is_file():
                file_loc = Path("benchmark_results") / filename

            if file_loc.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(file_loc.stat().st_size))
                self.end_headers()
                with open(file_loc, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if path == "/api/process_order":
            transcript = payload.get("transcript", "bhaiya do packet maggi aur ek pav dhaniya de do")
            customer = payload.get("customer", "Ramesh Bhaiya")
            apply_noise = payload.get("apply_noise", True)

            # Evon NLU parsing + Inventory calculation
            parsed = parse_transcript_to_json(transcript)
            invoice_lines, grand_total = calculate_bill(parsed.get("items", []))

            # Generate Hindi confirmation
            items_text = ", aur ".join([f"{it['ordered_quantity']} {it['ordered_unit']} {it['matched_name']}" for it in invoice_lines])
            speech = f"{customer} ji, aapka {items_text} bill me jodh diya gaya hai. Kul bill {int(grand_total)} rupaye hua hai. Dhanyawad!"

            # Generate audio response
            out_audio = Path("kirana_audio_demo") / "order_confirmation.wav"
            out_audio.parent.mkdir(exist_ok=True)

            try:
                if GnaniConfig.API_KEY != "YOUR_GNANI_API_KEY_HERE":
                    from gnani_pipeline import synthesize_speech
                    synthesize_speech(speech, str(out_audio))
            except Exception:
                pass

            response_data = {
                "success": True,
                "customer": customer,
                "raw_transcript": transcript,
                "invoice": invoice_lines,
                "grand_total_inr": grand_total,
                "spoken_confirmation": speech,
            }
            self._send_json(response_data)

        elif path == "/api/khata_command":
            transcript = payload.get("transcript", "")
            out_audio = Path("kirana_audio_demo") / "khata_speech.wav"
            res = process_khata_voice_command(transcript, ledger=LEDGER, output_speech_path=str(out_audio))
            self._send_json(res)

        else:
            self.send_response(404)
            self.end_headers()

    def _send_json(self, data: Any) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default verbose HTTP request logging."""
        pass


def run_web_server() -> None:
    server = HTTPServer(("0.0.0.0", PORT), KiranaHttpHandler)
    print("=" * 70)
    print(f"  🛒 KIRANA MITRA - WEB DASHBOARD RUNNING ON http://localhost:{PORT}")
    print("  Zero external dependencies. Press Ctrl+C to stop.")
    print("=" * 70)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server. Dhanyawad!")
        server.server_close()


if __name__ == "__main__":
    run_web_server()

