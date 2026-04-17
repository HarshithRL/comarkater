"""Pydantic validators for parsed sub-agent response items.

LEAF module — only depends on pydantic + re.
Validates TextItem, TableItem, and provides ParsedOutput
for aggregate validation with dedup and error collection.

Ported from legacy: lines 1812-1931.
"""

import re
from pydantic import BaseModel, field_validator, model_validator


class TextItem(BaseModel):
    """Validated text item from sub-agent response."""

    type: str = "text"
    id: str
    value: str

    @field_validator("value")
    @classmethod
    def value_not_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only text values."""
        if not v or not v.strip():
            raise ValueError("Text item value cannot be empty")
        return v

    @field_validator("value")
    @classmethod
    def no_raw_table_markup(cls, v: str) -> str:
        """Reject separator rows that leaked into text items."""
        stripped = v.strip()
        if stripped.startswith('|') and stripped.endswith('|') and stripped.count('|') >= 3:
            if re.match(r'^\|[\s:-]+(\|[\s:-]+)+\|$', stripped):
                raise ValueError(f"Raw separator leaked into text: {stripped[:50]}")
        return v


class TableValue(BaseModel):
    """Validated table value with headers, data rows, and column alignment."""

    tableHeaders: list[str]
    data: list[list[str]]
    alignment: list[str]

    @model_validator(mode="after")
    def validate_structure(self) -> "TableValue":
        """Ensure alignment/data dimensions match header count."""
        n = len(self.tableHeaders)
        if n == 0:
            raise ValueError("Table has no headers")
        if len(self.alignment) != n:
            raise ValueError(f"Alignment ({len(self.alignment)}) != headers ({n})")
        for i, row in enumerate(self.data):
            if len(row) != n:
                raise ValueError(f"Row {i}: {len(row)} cells, expected {n}")
        for a in self.alignment:
            if a not in ("left", "right", "center"):
                raise ValueError(f"Invalid alignment: {a}")
        return self


class TableItem(BaseModel):
    """Validated table item from sub-agent response."""

    type: str = "table"
    id: str
    value: TableValue


class ParsedOutput(BaseModel):
    """Aggregate validation result from SubAgentResponseParser.

    Usage:
        items = SubAgentResponseParser.parse(raw_content)
        validated = ParsedOutput.validate_items(items)
        validated.to_items_list()  # back to dict format for frontend
    """

    items: list = []
    has_table: bool = False
    has_text: bool = False
    table_count: int = 0
    text_count: int = 0
    total_rows: int = 0
    errors: list[str] = []

    @classmethod
    def validate_items(cls, raw_items: list[dict]) -> "ParsedOutput":
        """Validate and enrich parser output. Invalid items are dropped with errors logged."""
        if not raw_items:
            return cls()

        validated = []
        errors = []
        seen_ids: set[str] = set()

        for item in raw_items:
            item_type = item.get("type")
            item_id = item.get("id", "")

            if item_id in seen_ids:
                errors.append(f"Duplicate ID dropped: {item_id[:20]}")
                continue
            seen_ids.add(item_id)

            try:
                if item_type == "text":
                    validated.append(TextItem(**item))
                elif item_type == "table":
                    validated.append(TableItem(**item))
                else:
                    errors.append(f"Unknown item type: {item_type}")
            except Exception as e:
                errors.append(f"{item_type} validation failed: {str(e)[:120]}")

        table_items = [i for i in validated if isinstance(i, TableItem)]
        text_items = [i for i in validated if isinstance(i, TextItem)]

        return cls(
            items=validated,
            has_table=len(table_items) > 0,
            has_text=len(text_items) > 0,
            table_count=len(table_items),
            text_count=len(text_items),
            total_rows=sum(len(t.value.data) for t in table_items),
            errors=errors,
        )

    def to_items_list(self) -> list[dict]:
        """Convert back to dict format for the frontend."""
        return [item.model_dump() for item in self.items]
