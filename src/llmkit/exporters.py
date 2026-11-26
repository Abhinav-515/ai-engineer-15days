"""Export validated invoices to Pandas DataFrames and SQLite."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from llmkit.schemas import Invoice

INVOICE_COLUMNS = [
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "due_date",
    "currency",
    "status",
    "subtotal",
    "tax",
    "total",
]
LINE_ITEM_COLUMNS = [
    "vendor_name",
    "invoice_number",
    "line_no",
    "description",
    "quantity",
    "unit_price",
    "amount",
]

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS invoices (
    vendor_name TEXT NOT NULL,
    invoice_number TEXT NOT NULL,
    invoice_date TEXT NOT NULL,
    due_date TEXT,
    currency TEXT NOT NULL,
    status TEXT NOT NULL,
    subtotal REAL NOT NULL,
    tax REAL NOT NULL,
    total REAL NOT NULL,
    UNIQUE (vendor_name, invoice_number)
);
CREATE TABLE IF NOT EXISTS line_items (
    vendor_name TEXT NOT NULL,
    invoice_number TEXT NOT NULL,
    line_no INTEGER NOT NULL,
    description TEXT NOT NULL,
    quantity REAL NOT NULL,
    unit_price REAL NOT NULL,
    amount REAL NOT NULL,
    UNIQUE (vendor_name, invoice_number, line_no),
    FOREIGN KEY (vendor_name, invoice_number)
        REFERENCES invoices (vendor_name, invoice_number)
);
"""


def invoices_to_frames(invoices: Iterable[Invoice]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten invoices into an invoices table and a line_items table."""
    invoice_rows: list[dict] = []
    item_rows: list[dict] = []
    for inv in invoices:
        invoice_rows.append(
            {
                "vendor_name": inv.vendor_name,
                "invoice_number": inv.invoice_number,
                "invoice_date": inv.invoice_date.isoformat(),
                "due_date": inv.due_date.isoformat() if inv.due_date else None,
                "currency": inv.currency.value,
                "status": inv.status.value,
                "subtotal": round(inv.subtotal, 2),
                "tax": round(inv.tax, 2),
                "total": round(inv.total, 2),
            }
        )
        for line_no, item in enumerate(inv.line_items, start=1):
            item_rows.append(
                {
                    "vendor_name": inv.vendor_name,
                    "invoice_number": inv.invoice_number,
                    "line_no": line_no,
                    "description": item.description,
                    "quantity": item.quantity,
                    "unit_price": round(item.unit_price, 2),
                    "amount": round(item.amount, 2),
                }
            )
    return (
        pd.DataFrame(invoice_rows, columns=INVOICE_COLUMNS),
        pd.DataFrame(item_rows, columns=LINE_ITEM_COLUMNS),
    )


def write_sqlite(invoices: Iterable[Invoice], db_path: str | Path) -> dict[str, int]:
    """Insert invoices idempotently; return counts of inserted and skipped invoices."""
    inserted = 0
    skipped = 0
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_SQL)
        with conn:
            for invoice in invoices:
                invoice_df, items_df = invoices_to_frames([invoice])
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO invoices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    tuple(invoice_df.iloc[0]),
                )
                if cursor.rowcount == 1:
                    inserted += 1
                    conn.executemany(
                        "INSERT INTO line_items VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [tuple(item) for item in items_df.itertuples(index=False)],
                    )
                else:
                    skipped += 1
    finally:
        conn.close()
    return {"inserted": inserted, "skipped": skipped}
