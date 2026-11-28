"""Pydantic models for invoice extraction: types, enums and business-rule validation."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

TOLERANCE = 0.01  # money is float here for schema simplicity; compare with a tolerance


class Currency(StrEnum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    INR = "INR"


class PaymentStatus(StrEnum):
    PAID = "paid"
    UNPAID = "unpaid"
    OVERDUE = "overdue"
    UNKNOWN = "unknown"


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    amount: float = Field(ge=0)


class Invoice(BaseModel):
    # extra="forbid" -> "additionalProperties": false in the JSON Schema.
    # No defaults: every field is required (nullable where data may be absent),
    # which is what strict structured-output modes expect.
    model_config = ConfigDict(extra="forbid")

    invoice_number: str = Field(min_length=1)
    vendor_name: str = Field(min_length=1)
    invoice_date: date
    due_date: date | None
    currency: Currency
    status: PaymentStatus
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: float = Field(ge=0)
    tax: float = Field(ge=0)
    total: float = Field(ge=0)

    @model_validator(mode="after")
    def check_business_rules(self) -> Self:
        items_sum = sum(i.amount for i in self.line_items)
        if abs(items_sum - self.subtotal) > TOLERANCE:
            raise ValueError(f"line items sum to {items_sum:.2f}, subtotal is {self.subtotal:.2f}")
        if abs(self.subtotal + self.tax - self.total) > TOLERANCE:
            raise ValueError(
                f"subtotal {self.subtotal:.2f} + tax {self.tax:.2f} != total {self.total:.2f}"
            )
        if self.due_date and self.due_date < self.invoice_date:
            raise ValueError("due_date is before invoice_date")
        return self
