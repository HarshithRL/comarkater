"""AST-based markdown parser for sub-agent responses.

Uses markdown-it-py for proper table tokenization instead of regex.
3-phase algorithm:
  1. Split lines into (text, table) tuples, handling edge cases
  2. Group consecutive same-type lines into segments
  3. Parse each segment via AST (tables) or passthrough (text)

Handles edge cases:
  - Header row merged with preceding narrative text
  - Data rows with trailing narrative (no closing |)
  - Pandas index column leaking into markdown
  - Multiple tables with interleaved text
  - <name>...</name> routing tags

Ported from legacy: lines 1601-1805.
"""

import re
import uuid
from typing import Optional

from parsers.validators import ParsedOutput

# Pre-compiled regex (compiled once at module load)
_RE_NAME_TAG = re.compile(r'\s*<name>[^<]+</name>\s*')
_RE_NUMERIC_CELL = re.compile(r'^[\d,.%₹$€£\-]+$')


class SubAgentResponseParser:
    """Parse raw sub-agent markdown into structured [{type, id, value}] items."""

    _md = None  # Lazy-init singleton

    @classmethod
    def _get_md(cls):
        """Lazy-init markdown-it parser with table extension."""
        if cls._md is None:
            from markdown_it import MarkdownIt
            cls._md = MarkdownIt().enable('table')
        return cls._md

    @classmethod
    def parse(cls, raw_content: str) -> list[dict]:
        """Parse raw sub-agent content into structured items.

        Returns:
            List of dicts: [{"type": "text"|"table", "id": uuid, "value": ...}]
        """
        if not raw_content:
            return []
        cleaned = _RE_NAME_TAG.sub(' ', raw_content).strip()
        if not cleaned:
            return []

        # Handle escaped newlines
        cleaned = cleaned.replace('\\n', '\n')
        lines = cleaned.split('\n')

        # Phase 1: Split lines into text vs table, handling edge cases
        expanded = []
        for line in lines:
            stripped = line.strip()
            if '|' in stripped:
                pipe_idx = stripped.find('|')
                if pipe_idx > 0:
                    # Text before first pipe: "I'll analyze... | col1 | col2 |"
                    text_part = stripped[:pipe_idx].strip()
                    table_part = stripped[pipe_idx:].strip()
                    if text_part:
                        expanded.append(('text', text_part))
                    if table_part:
                        if table_part.startswith('|') and not table_part.endswith('|'):
                            last_pipe = table_part.rfind('|')
                            if last_pipe > 0:
                                tbl = table_part[:last_pipe + 1]
                                trail = table_part[last_pipe + 1:].strip()
                                expanded.append(('table', tbl))
                                if trail and len(trail) > 10 and ' ' in trail:
                                    expanded.append(('text', trail))
                            else:
                                expanded.append(('table', table_part))
                        else:
                            expanded.append(('table', table_part))
                else:
                    # Starts with | — check for trailing narrative
                    if not stripped.endswith('|') and stripped.count('|') >= 2:
                        last_pipe = stripped.rfind('|')
                        if last_pipe > 0:
                            tbl = stripped[:last_pipe + 1]
                            trail = stripped[last_pipe + 1:].strip()
                            expanded.append(('table', tbl))
                            if trail and len(trail) > 10 and ' ' in trail:
                                expanded.append(('text', trail))
                        else:
                            expanded.append(('table', stripped))
                    else:
                        expanded.append(('table', stripped))
            else:
                if stripped:
                    expanded.append(('text', stripped))

        # Phase 2: Group consecutive table/text lines
        segments: list[tuple[str, str]] = []
        current_type: Optional[str] = None
        current_lines: list[str] = []

        for tag, content in expanded:
            if tag != current_type:
                if current_lines:
                    segments.append((current_type, '\n'.join(current_lines)))
                current_type = tag
                current_lines = [content]
            else:
                current_lines.append(content)
        if current_lines:
            segments.append((current_type, '\n'.join(current_lines)))

        # Phase 3: Parse each segment into items
        items = []
        md = cls._get_md()

        for seg_type, seg_content in segments:
            if seg_type == 'text':
                text = seg_content.strip()
                if text:
                    items.append({"type": "text", "id": str(uuid.uuid4()), "value": text})

            elif seg_type == 'table':
                table_item = cls._parse_table_ast(md, seg_content)
                if table_item:
                    items.append(table_item)
                else:
                    # Fallback: AST couldn't parse, treat as text
                    if seg_content.strip():
                        items.append({"type": "text", "id": str(uuid.uuid4()), "value": seg_content.strip()})

        if not items:
            items.append({"type": "text", "id": str(uuid.uuid4()), "value": cleaned})

        return items

    @classmethod
    def _parse_table_ast(cls, md, table_markdown: str) -> Optional[dict]:
        """Parse markdown table using markdown-it AST tokenizer.

        Returns:
            Item dict with type=table, or None if parsing fails.
        """
        tokens = md.parse(table_markdown)

        headers: list[str] = []
        rows: list[list[str]] = []
        current_row: list[str] = []
        in_header = False
        in_body = False

        for token in tokens:
            if token.type == 'thead_open':
                in_header = True
            elif token.type == 'thead_close':
                in_header = False
            elif token.type == 'tbody_open':
                in_body = True
            elif token.type == 'tbody_close':
                in_body = False
            elif token.type == 'tr_open':
                current_row = []
            elif token.type == 'tr_close':
                if in_header and current_row:
                    headers = current_row
                elif in_body and current_row:
                    rows.append(current_row)
                current_row = []
            elif token.type == 'inline' and (in_header or in_body):
                current_row.append(token.content.strip() if token.content else '')

        # Clean empty headers (from unnamed index columns)
        headers = [h for h in headers if h.strip()] if headers else []

        if not headers or not rows:
            return None

        # Handle index column: if row has 1 extra cell, strip first cell
        cleaned_rows = []
        for row in rows:
            if len(row) == len(headers) + 1:
                row = row[1:]
            if len(row) < len(headers):
                row.extend([''] * (len(headers) - len(row)))
            elif len(row) > len(headers):
                row = row[:len(headers)]
            cleaned_rows.append(row)

        if not cleaned_rows:
            return None

        # Compute alignment based on numeric content
        alignment = []
        for i in range(len(headers)):
            numeric_count = sum(
                1 for row in cleaned_rows
                if i < len(row) and row[i] and _RE_NUMERIC_CELL.match(row[i].strip())
            )
            alignment.append("right" if numeric_count > len(cleaned_rows) / 2 else "left")

        return {
            "type": "table",
            "id": str(uuid.uuid4()),
            "value": {
                "tableHeaders": headers,
                "data": cleaned_rows,
                "alignment": alignment,
            }
        }
