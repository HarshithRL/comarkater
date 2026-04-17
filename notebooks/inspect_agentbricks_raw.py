"""Inspect raw AgentBricks response for all 3 clients.

Calls AgentBricks endpoint DIRECTLY (bypassing our agent) using each client's
SP token. Dumps the full response object so we can see exactly what AgentBricks returns.

Usage:
    cd comarketer
    python notebooks/inspect_agentbricks_raw.py
"""

import json
import time
import base64
import httpx
from openai import OpenAI

# ── Config ──
DATABRICKS_HOST = "https://dbc-540c0e05-7c19.cloud.databricks.com"
AGENTBRICKS_ENDPOINT = "mas-0c49109d-endpoint"
SECRETS_SCOPE = "agent-sp-credentials"

QUESTION = "Show me top 5 email campaigns by open rate last month"

CLIENTS = {
    "IGP": {
        "client_id_key": "igp-sp-client-id",
        "secret_key": "igp-sp-client-secret",
    },
    "Pepe Jeans": {
        "client_id_key": "pepe-sp-client-id",
        "secret_key": "pepe-sp-client-secret",
    },
    "Crocs": {
        "client_id_key": "shopcrocsmt-sp-client-id",
        "secret_key": "shopcrocsmt-sp-client-secret",
    },
}


def get_workspace_client():
    """Init WorkspaceClient with browser auth."""
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient(
        host=DATABRICKS_HOST,
        auth_type="external-browser",
    )


def read_secret(ws, key: str) -> str:
    """Read a secret from the scope."""
    resp = ws.secrets.get_secret(scope=SECRETS_SCOPE, key=key)
    return base64.b64decode(resp.value).decode("utf-8")


def get_sp_token(client_id: str, client_secret: str) -> str:
    """Get SP OAuth token via client_credentials flow."""
    resp = httpx.post(
        f"{DATABRICKS_HOST}/oidc/v1/token",
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "all-apis",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15.0,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def dump_response(response) -> dict:
    """Recursively convert the OpenAI response object to a serializable dict."""
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "__dict__"):
        result = {}
        for k, v in response.__dict__.items():
            if k.startswith("_"):
                continue
            result[k] = dump_response(v)
        return result
    if isinstance(response, list):
        return [dump_response(item) for item in response]
    if isinstance(response, dict):
        return {k: dump_response(v) for k, v in response.items()}
    return response


def call_agentbricks(token: str, question: str) -> object:
    """Call AgentBricks endpoint directly."""
    client = OpenAI(
        api_key=token,
        base_url=f"{DATABRICKS_HOST}/serving-endpoints",
    )
    return client.responses.create(
        model=AGENTBRICKS_ENDPOINT,
        input=[{"role": "user", "content": question}],
        extra_body={"databricks_options": {"return_trace": True}},
    )


def inspect_trace_spans(raw: dict):
    """Pull out key info from databricks_output trace spans."""
    db_output = raw.get("databricks_output")
    if not db_output:
        print("    [No databricks_output]")
        return

    spans = db_output.get("trace", {}).get("data", {}).get("spans", [])
    print(f"    Trace spans: {len(spans)}")

    for i, span in enumerate(spans):
        name = span.get("name", "?")
        attrs = span.get("attributes", {})
        span_type = attrs.get("mlflow.spanType", "?")

        has_sql = "sql_query" in attrs
        has_inputs = "mlflow.spanInputs" in attrs
        has_outputs = "mlflow.spanOutputs" in attrs

        attr_keys = list(attrs.keys())

        print(f"    Span[{i}]: name={name} | type={span_type}")
        print(f"      has_sql={has_sql} | has_inputs={has_inputs} | has_outputs={has_outputs}")
        print(f"      attr_keys: {attr_keys}")

        if has_sql:
            sql = attrs["sql_query"]
            if isinstance(sql, str) and sql.startswith('"'):
                try:
                    sql = json.loads(sql)
                except Exception:
                    pass
            print(f"      SQL: {str(sql)[:300]}")

        if has_outputs and "AGENT" in str(span_type):
            outputs_raw = attrs["mlflow.spanOutputs"]
            try:
                outputs = json.loads(outputs_raw) if isinstance(outputs_raw, str) else outputs_raw
                outputs_str = json.dumps(outputs) if isinstance(outputs, (dict, list)) else str(outputs)
                print(f"      AGENT outputs: {outputs_str[:500]}")
            except Exception:
                print(f"      AGENT outputs (raw): {str(outputs_raw)[:500]}")

        if has_inputs and "AGENT" in str(span_type):
            inputs_raw = attrs["mlflow.spanInputs"]
            try:
                inputs = json.loads(inputs_raw) if isinstance(inputs_raw, str) else inputs_raw
                inputs_str = json.dumps(inputs) if isinstance(inputs, (dict, list)) else str(inputs)
                print(f"      AGENT inputs: {inputs_str[:500]}")
            except Exception:
                print(f"      AGENT inputs (raw): {str(inputs_raw)[:500]}")


def main():
    print("=" * 70)
    print("  AgentBricks Raw Response Inspector")
    print(f"  Question: {QUESTION}")
    print("=" * 70)

    ws = get_workspace_client()
    print(f"Authenticated as: {ws.current_user.me().user_name}\n")

    for client_name, keys in CLIENTS.items():
        print(f"\n{'=' * 70}")
        print(f"  CLIENT: {client_name}")
        print(f"{'=' * 70}")

        # Get credentials
        sp_client_id = read_secret(ws, keys["client_id_key"])
        sp_client_secret = read_secret(ws, keys["secret_key"])
        print(f"  SP client_id: {sp_client_id[:8]}...")

        # Get token
        token = get_sp_token(sp_client_id, sp_client_secret)
        print(f"  Token len: {len(token)}")

        # Call AgentBricks
        print(f"  Calling AgentBricks...")
        t0 = time.time()
        try:
            response = call_agentbricks(token, QUESTION)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        latency = time.time() - t0
        print(f"  Latency: {latency:.1f}s")

        # Dump full response
        raw = dump_response(response)

        # ── 1. Top-level keys ──
        print(f"\n  TOP-LEVEL KEYS: {list(raw.keys())}")

        # ── 2. Output structure ──
        output = raw.get("output", [])
        print(f"\n  OUTPUT: {len(output)} items")
        all_text_blocks = []

        for i, item in enumerate(output):
            item_type = item.get("type", "?")

            # function_call items
            if item_type == "function_call":
                fn_name = item.get("name", "?")
                fn_args = str(item.get("arguments", ""))[:300]
                call_id = str(item.get("call_id", ""))[:20]
                print(f"    [{i}] function_call: name={fn_name} | call_id={call_id}...")
                print(f"        args: {fn_args}")
                continue

            # function_call_output items
            if item_type == "function_call_output":
                call_id = str(item.get("call_id", ""))[:20]
                fn_out_str = str(item.get("output", ""))
                print(f"    [{i}] function_call_output: call_id={call_id}... | len={len(fn_out_str)}")
                print(f"        preview: {fn_out_str[:500]}")
                # Check for pipe tables
                pipe_lines = [l for l in fn_out_str.split("\n") if l.strip().startswith("|") and l.strip().endswith("|")]
                if pipe_lines:
                    print(f"        PIPE TABLE: {len(pipe_lines)} lines")
                    for pl in pipe_lines[:10]:
                        print(f"          {pl.strip()}")
                    all_text_blocks.append(fn_out_str)
                continue

            # message items
            content = item.get("content") or []
            item_role = item.get("role", "?")
            print(f"    [{i}] message: role={item_role} | content_blocks={len(content)}")
            for j, block in enumerate(content):
                if isinstance(block, dict):
                    block_type = block.get("type", "?")
                    text = block.get("text", "") or ""
                else:
                    block_type = "?"
                    text = str(block)
                print(f"        [{j}] type={block_type} | len={len(text)}")
                if text:
                    all_text_blocks.append(text)
                    print(f"        TEXT:\n          {text[:800]}")
                    pipe_lines = [l for l in text.split("\n") if l.strip().startswith("|") and l.strip().endswith("|")]
                    if pipe_lines:
                        print(f"        PIPE TABLE: {len(pipe_lines)} lines")
                        for pl in pipe_lines[:10]:
                            print(f"          {pl.strip()}")

        # Summary
        if all_text_blocks:
            total_text = "\n".join(all_text_blocks)
            all_pipes = [l for l in total_text.split("\n") if l.strip().startswith("|") and l.strip().endswith("|")]
            print(f"\n  TEXT SUMMARY: {len(all_text_blocks)} blocks | {len(total_text)} chars | {len(all_pipes)} pipe-table lines")
        else:
            print(f"\n  TEXT SUMMARY: no text blocks found")

        # ── 3. databricks_output / trace spans ──
        print(f"\n  DATABRICKS_OUTPUT:")
        has_db = raw.get("databricks_output") is not None
        print(f"    present: {has_db}")
        if has_db:
            inspect_trace_spans(raw)

        # ── 4. Save full raw JSON ──
        filename = f"notebooks/raw_response_{client_name.lower().replace(' ', '_')}.json"
        with open(filename, "w") as f:
            json.dump(raw, f, indent=2, default=str)
        print(f"\n  Saved: {filename}")

    print(f"\n{'=' * 70}")
    print("  DONE - check the raw_response_*.json files for full details")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
