import sqlite3
from datetime import date

from llmkit.exporters import invoices_to_frames, write_sqlite
from llmkit.schemas import Currency, Invoice, LineItem, PaymentStatus


def make_invoice(number="A-1", vendor="Acme", due=None, amounts=(10.0, 5.0)):
    items = [
        LineItem(description=f"item {i}", quantity=1, unit_price=a, amount=a)
        for i, a in enumerate(amounts, start=1)
    ]
    subtotal = sum(amounts)
    return Invoice(
        invoice_number=number,
        vendor_name=vendor,
        invoice_date=date(2026, 3, 14),
        due_date=due,
        currency=Currency.USD,
        status=PaymentStatus.PAID,
        line_items=items,
        subtotal=subtotal,
        tax=0.0,
        total=subtotal,
    )


def test_frames_have_expected_shape_and_columns():
    invoices_df, items_df = invoices_to_frames([make_invoice(), make_invoice("A-2")])
    assert len(invoices_df) == 2
    assert len(items_df) == 4
    assert list(items_df["line_no"]) == [1, 2, 1, 2]
    assert invoices_df.loc[0, "currency"] == "USD"
    assert invoices_df.loc[0, "status"] == "paid"


def test_frames_serialize_dates_and_missing_due_date():
    invoices_df, _ = invoices_to_frames([make_invoice(due=date(2026, 4, 13)), make_invoice("A-2")])
    assert invoices_df.loc[0, "invoice_date"] == "2026-03-14"
    assert invoices_df.loc[0, "due_date"] == "2026-04-13"
    assert invoices_df["due_date"].isna().tolist() == [False, True]


def test_frames_round_money_to_two_decimals():
    invoice = make_invoice(amounts=(0.1, 0.2))
    invoices_df, items_df = invoices_to_frames([invoice])
    assert invoices_df.loc[0, "subtotal"] == 0.3
    assert list(items_df["amount"]) == [0.1, 0.2]


def test_empty_input_gives_empty_frames_with_columns():
    invoices_df, items_df = invoices_to_frames([])
    assert invoices_df.empty and items_df.empty
    assert "invoice_number" in invoices_df.columns
    assert "line_no" in items_df.columns


def test_write_sqlite_inserts_rows(tmp_path):
    db = tmp_path / "out.db"
    counts = write_sqlite([make_invoice(), make_invoice("A-2")], db)
    assert counts == {"inserted": 2, "skipped": 0}
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM invoices").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM line_items").fetchone()[0] == 4
    conn.close()


def test_write_sqlite_is_idempotent(tmp_path):
    db = tmp_path / "out.db"
    write_sqlite([make_invoice()], db)
    counts = write_sqlite([make_invoice()], db)
    assert counts == {"inserted": 0, "skipped": 1}
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM invoices").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM line_items").fetchone()[0] == 2
    conn.close()


def test_duplicate_within_one_batch_is_skipped(tmp_path):
    db = tmp_path / "out.db"
    counts = write_sqlite([make_invoice(), make_invoice()], db)
    assert counts == {"inserted": 1, "skipped": 1}


def test_same_number_from_different_vendors_is_kept(tmp_path):
    db = tmp_path / "out.db"
    counts = write_sqlite([make_invoice(vendor="Acme"), make_invoice(vendor="Globex")], db)
    assert counts == {"inserted": 2, "skipped": 0}


def test_null_due_date_is_stored_as_null(tmp_path):
    db = tmp_path / "out.db"
    write_sqlite([make_invoice()], db)
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT due_date FROM invoices").fetchone()[0] is None
    conn.close()
