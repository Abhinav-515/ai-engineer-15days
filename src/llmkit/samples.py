"""Sample invoice texts (clean, multilingual/messy, missing fields) with expected values."""

SAMPLES: list[dict] = [
    {
        "name": "clean",
        "expected": {"invoice_number": "INV-2026-0042", "total": 671.0, "currency": "USD"},
        "text": """INVOICE #INV-2026-0042
Acme Cloud Services Ltd.
Date: 2026-03-14    Due: 2026-04-13
Bill to: Globex Corp

Description              Qty   Unit Price   Amount
API credits (1M calls)    3     120.00      360.00
Premium support           1     250.00      250.00

Subtotal: 610.00
Tax (10%): 61.00
Total: USD 671.00
Status: PAID
""",
    },
    {
        "name": "messy-eur",
        "expected": {"invoice_number": "R-7781", "total": 593.81, "currency": "EUR"},
        "text": """Rechnung Nr. R-7781 / Invoice
Müller Büromaterial GmbH
Rechnungsdatum: 02 May 2026
Zahlbar bis: 16 May 2026

2 x Ergonomic chair @ EUR 189.50 = 379.00
5 x Desk lamp @ EUR 24.00 = 120.00
Nettobetrag: 499.00
MwSt 19%: 94.81
Gesamtbetrag: EUR 593.81
Noch nicht bezahlt
""",
    },
    {
        "name": "no-due-date",
        "expected": {"invoice_number": "5521", "total": 1980.0, "currency": "USD"},
        "text": """Invoice 5521 - Bright Studio
Issued Jan 9, 2026
Logo design package - 1 - $1,800.00
Hosting (12 months) - 12 x $15.00 - $180.00
Total due: $1,980.00 (no tax)
OVERDUE
""",
    },
]
