"""Response formatting functions for sub-agent and supervisor output.

Converts raw sub-agent responses into validated structured JSON,
and formats supervisor JSON output for client consumption
(numeric formatting, alignment defaults).

Ported from legacy: lines 1160-1170, 2128-2252.
"""

import logging
import re
import uuid

from parsers.subagent_parser import SubAgentResponseParser
from parsers.validators import ParsedOutput

logger = logging.getLogger(__name__)


def build_custom_outputs(custom_inputs, agent_id: str, output_type: str) -> dict:
    """Build custom_outputs metadata dict (single source of truth).

    Args:
        custom_inputs: CustomInputs instance with user context.
        agent_id: Agent identifier string.
        output_type: "observation" or "RATIONALE".

    Returns:
        Dict with user_name, user_id, thread_id, conversation_id, task_type, agent_id, type.
    """
    return {
        "user_name": getattr(custom_inputs, "user_name", "User"),
        "user_id": getattr(custom_inputs, "user_id", ""),
        "thread_id": getattr(custom_inputs, "thread_id", ""),
        "conversation_id": getattr(custom_inputs, "conversation_id", ""),
        "task_type": getattr(custom_inputs, "task_type", "general"),
        "agent_id": agent_id,
        "type": output_type,
    }


def pipe_table_to_2d(pipe_md: str) -> dict:
    """Convert pipe-delimited markdown table to 2D array format.

    Input:  "| Campaign | Open Rate |\\n|---|---|\\n| D3_Plants | 61.24 |"
    Output: {"tableHeaders": ["Campaign","Open Rate"], "data": [["D3_Plants",61.24]], "alignment": ["left","left"]}
    """
    lines = [l.strip() for l in pipe_md.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return {"tableHeaders": [], "data": [], "alignment": []}

    def parse_row(line: str) -> list[str]:
        return [c.strip() for c in line.split("|") if c.strip() != ""]

    headers = parse_row(lines[0])

    # Skip separator row (dashes)
    data_start = 1
    if len(lines) > 1 and re.match(r"^[\s|:\-]+$", lines[1]):
        data_start = 2

    data = []
    for line in lines[data_start:]:
        cells = parse_row(line)
        typed_cells = []
        for cell in cells:
            clean = cell.replace(",", "").replace("%", "").strip()
            try:
                if "." in clean:
                    typed_cells.append(float(clean))
                else:
                    typed_cells.append(int(clean))
            except ValueError:
                typed_cells.append(cell)
        data.append(typed_cells)

    alignment = ["left"] * len(headers)
    return {"tableHeaders": headers, "data": data, "alignment": alignment}


def format_subagent_response(raw_content: str, agent_name: str, custom_inputs) -> dict:
    """Convert raw sub-agent response into validated structured JSON.

    Strips error messages, parses markdown via AST, validates via Pydantic.
    Invalid items are dropped with warnings. Falls back to raw items if all fail.

    Returns:
        Dict with "items" (list of item lists) and "custom_outputs".
    """
    if not raw_content or not raw_content.strip():
        logger.warning(f"FORMAT-SUBAGENT: Empty content from {agent_name}")
        return {
            "items": [[{"type": "text", "id": str(uuid.uuid4()), "value": "No data returned from analytics agent."}]],
            "custom_outputs": build_custom_outputs(custom_inputs, "ZECDLGGP3J", "observation"),
        }

    # Strip sub-agent error messages from content
    error_patterns = [
        "Error: peer closed connection",
        "incomplete chunked read",
        "Error: Connection reset",
        "Error: timeout",
        "HTTPError",
        "ServiceUnavailable",
    ]
    clean_content = raw_content
    detected_errors = []
    for pattern in error_patterns:
        if pattern.lower() in clean_content.lower():
            detected_errors.append(pattern)
            lines = clean_content.split('\n')
            clean_content = '\n'.join(
                line for line in lines
                if pattern.lower() not in line.lower()
            )

    if detected_errors:
        logger.warning(f"FORMAT-SUBAGENT: Stripped errors from {agent_name}: {detected_errors}")

    clean_content = clean_content.strip()
    if not clean_content:
        error_msg = f"The analytics agent encountered an error ({', '.join(detected_errors)}). Please try refining your query."
        return {
            "items": [[{"type": "text", "id": str(uuid.uuid4()), "value": error_msg}]],
            "custom_outputs": build_custom_outputs(custom_inputs, "ZECDLGGP3J", "observation"),
        }

    # Parse and validate
    raw_items = SubAgentResponseParser.parse(clean_content)
    validated = ParsedOutput.validate_items(raw_items)

    if validated.errors:
        for err in validated.errors:
            logger.warning(f"FORMAT-SUBAGENT: Validation issue in {agent_name}: {err}")

    items = validated.to_items_list() if validated.items else raw_items

    # Post-validation: strip empty text items
    items = [item for item in items if not (item.get("type") == "text" and not item.get("value", "").strip())]

    if not items:
        items = [{"type": "text", "id": str(uuid.uuid4()), "value": "Data received but could not be formatted. Please try a more specific query."}]

    # Append error warning if we stripped errors but still have data
    if detected_errors and len(items) > 0:
        items.append({
            "type": "text",
            "id": str(uuid.uuid4()),
            "value": f"Note: Partial data returned due to: {', '.join(detected_errors)}. Results may be incomplete.",
        })

    logger.info(
        f"FORMAT-SUBAGENT: {agent_name} -> "
        f"{validated.text_count} text + {validated.table_count} table "
        f"({validated.total_rows} rows), errors={len(validated.errors)}, "
        f"stream_errors={len(detected_errors)}"
    )
    return {
        "items": [items],
        "custom_outputs": build_custom_outputs(custom_inputs, "ZECDLGGP3J", "observation"),
    }


def format_for_client(supervisor_json: dict) -> dict:
    """Format supervisor JSON output for client consumption.

    Applies numeric formatting:
    - Integers >999 get comma separation (e.g., 3,345)
    - Floats get 2 decimal places with comma separation for >=100
    - None values become empty strings

    Returns:
        Dict with "items" key containing formatted item list.
    """
    items = supervisor_json.get("items", [])
    formatted_items = []

    for item in items:
        formatted_item = {"type": item["type"], "id": item["id"]}
        if "name" in item:
            formatted_item["name"] = item["name"]

        if item["type"] == "text":
            formatted_item["value"] = item["value"]
        elif item["type"] == "table":
            table_value = item["value"]
            formatted_data = []
            for row in table_value["data"]:
                formatted_row = []
                for cell in row:
                    if isinstance(cell, int):
                        formatted_row.append(f"{cell:,}" if cell > 999 else str(cell))
                    elif isinstance(cell, float):
                        formatted_row.append(f"{cell:,.2f}" if cell >= 100 else f"{cell:.2f}")
                    elif cell is None:
                        formatted_row.append("")
                    else:
                        formatted_row.append(str(cell))
                formatted_data.append(formatted_row)
            formatted_item["value"] = {
                "tableHeaders": table_value["tableHeaders"],
                "data": formatted_data,
                "alignment": table_value.get("alignment", ["left"] * len(table_value["tableHeaders"])),
            }
        elif item["type"] == "chart":
            formatted_item["value"] = item["value"]
        elif item["type"] == "collapsedText":
            formatted_item["value"] = item["value"]
            if "hidden" in item:
                formatted_item["hidden"] = item["hidden"]

        formatted_items.append(formatted_item)

    return {"items": [formatted_items]}
