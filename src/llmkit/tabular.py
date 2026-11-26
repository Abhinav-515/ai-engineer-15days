"""Deterministic parsing of flat line-item tables into Invoice objects."""

from __future__ import annotations

import math

from pydantic import ValidationError

from llmkit.schemas import TOLERANCE, Invoice

HEADER_FIELDS = (
    "invoice_number",
    "vendor_name",
    "invoice_date",
    "due_date",
    "currency",
    "status",
    "tax",
)
ITEM_FIELDS = ("description", "quantity", "unit_price", "amount")
REQUIRED_COLUMNS = HEADER_FIELDS + ITEM_FIELDS


class TableParseError(ValueError):
    """The table is not a clean flat invoice table; callers may fall back to the LLM."""


def _normalize(name: str) -> str:
    return "_".join(name.strip().lower().replace("-", " ").split())


def _number(value: str, column: str) -> float:
    try:
        number = float(value.replace(",", "").replace(" ", ""))
    except ValueError as error:
        raise TableParseError(f"column '{column}': '{value}' is not a number") from error
    if not math.isfinite(number):
        raise TableParseError(f"column '{column}': '{value}' is not a number")
    return number


def parse_invoice_table(rows: list[list[str]]) -> Invoice:
    """Turn a header row plus data rows into one validated Invoice."""
    if len(rows) < 2:
        raise TableParseError("table has no data rows")

    header = [_normalize(cell) for cell in rows[0]]
    missing = [name for name in REQUIRED_COLUMNS if name not in header]
    if missing:
        raise TableParseError(f"missing columns: {', '.join(missing)}")
    index = {name: header.index(name) for name in REQUIRED_COLUMNS}

    records: list[dict[str, str]] = []
    for line_no, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) != len(header):
            raise TableParseError(f"row {line_no} has {len(row)} cells, expected {len(header)}")
        records.append({name: row[i].strip() for name, i in index.items()})
    if not records:
        raise TableParseError("table has no data rows")

    for name in HEADER_FIELDS:
        values = {record[name] for record in records}
        if len(values) > 1:
            raise TableParseError(f"inconsistent values in column '{name}': {sorted(values)}")

    items: list[dict] = []
    for n, record in enumerate(records, start=1):
        quantity = _number(record["quantity"], "quantity")
        unit_price = _number(record["unit_price"], "unit_price")
        amount = _number(record["amount"], "amount")
        if abs(quantity * unit_price - amount) > TOLERANCE:
            raise TableParseError(
                f"row {n}: quantity x unit_price = {quantity * unit_price:.2f}, "
                f"amount is {amount:.2f}"
            )
        items.append(
            {
                "description": record["description"],
                "quantity": quantity,
                "unit_price": unit_price,
                "amount": amount,
            }
        )

    first = records[0]
    tax = _number(first["tax"], "tax")
    subtotal = round(sum(item["amount"] for item in items), 2)
    try:
        return Invoice(
            invoice_number=first["invoice_number"],
            vendor_name=first["vendor_name"],
            invoice_date=first["invoice_date"],
            due_date=first["due_date"] or None,
            currency=first["currency"].upper(),
            status=first["status"].lower(),
            line_items=items,
            subtotal=subtotal,
            tax=tax,
            total=round(subtotal + tax, 2),
        )
    except ValidationError as error:
        raise TableParseError(f"validation failed: {error}") from error
