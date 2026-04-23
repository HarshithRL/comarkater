"""External client — authenticates with Databricks SDK (SP OAuth M2M) and
streams the CoMarketer /invocations endpoint.

Demonstrates the integration pattern an external (non-Databricks-App) backend
would use: SP client_id/secret → bearer token → SSE streaming /invocations,
optional reconnect via GET /responses/{id}.

Usage:
    python agent_server/tests/test_external_sdk_client.py \
        --query "how is my email CTR last week"

    # resume the stream via GET /responses/{id} after the POST closes
    python agent_server/tests/test_external_sdk_client.py --reconnect --query "..."

App URL and SP creds have dev defaults baked in; override via env
(COMARKETER_ENDPOINT, DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET) or CLI
flag. Rotate the SP secret if this file is ever pushed to a public repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Iterable, Iterator

import httpx
from databricks.sdk import WorkspaceClient

DATABRICKS_HOST = os.environ.get(
    "DATABRICKS_HOST", "https://dbc-540c0e05-7c19.cloud.databricks.com"
)
SP_CLIENT_ID = os.environ.get(
    "DATABRICKS_CLIENT_ID", "130eea19-be15-4008-b06c-9a5dc6bdfe09"
)
SP_CLIENT_SECRET = os.environ.get(
    "DATABRICKS_CLIENT_SECRET", "dose78874886c6cd25a02ad906630d91c5f3"
)

COMARKETER_ENDPOINT = os.environ.get(
    "COMARKETER_ENDPOINT",
    "https://comarketer-2276245144129479.aws.databricksapps.com",
)

DEFAULT_SP_ID = "sp-igp-001"
DEFAULT_USER_NAME = "IGP Test"
DEFAULT_TIMEOUT_S = 900.0  # matches server task_timeout_seconds (start_server.py:54)


def get_auth_headers() -> dict[str, str]:
    """Authenticate the SP via Databricks SDK and return bearer headers.

    ``config.authenticate()`` returns ``{"Authorization": "Bearer <token>"}``
    — the SDK handles token exchange + caching + refresh internally.
    """
    w = WorkspaceClient(
        host=DATABRICKS_HOST,
        client_id=SP_CLIENT_ID,
        client_secret=SP_CLIENT_SECRET,
        auth_type="oauth-m2m",
    )
    headers = w.config.authenticate()
    if "Authorization" not in headers:
        raise RuntimeError(f"Auth failed — no bearer in headers: {list(headers)}")
    return headers


def build_payload(
    query: str,
    sp_id: str = DEFAULT_SP_ID,
    user_name: str = DEFAULT_USER_NAME,
    stream: bool = True,
    background: bool = True,
) -> dict[str, Any]:
    return {
        "input": [{"role": "user", "content": query}],
        "stream": stream,
        "background": background,
        "custom_inputs": {"sp_id": sp_id, "user_name": user_name},
    }


def parse_sse_lines(lines: Iterable[str]) -> Iterator[dict]:
    """Parse SSE text lines into event dicts; skips ``:`` keepalive comments."""
    for raw in lines:
        if not raw:
            continue
        if raw.startswith(":"):
            continue
        if not raw.startswith("data:"):
            continue
        payload = raw[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            print(f"non-JSON SSE frame: {payload!r}", file=sys.stderr)


def print_event(event: dict, idx: int) -> None:
    etype = event.get("type") or event.get("event") or "event"
    preview = json.dumps(event, default=str)
    if len(preview) > 300:
        preview = preview[:300] + "..."
    print(f"[{idx:03d}] {etype:<22} {preview}")


def stream_invocation(
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float = DEFAULT_TIMEOUT_S,
) -> tuple[str | None, int]:
    """POST /invocations and stream SSE to stdout. Returns (response_id, last_seq)."""
    url = endpoint.rstrip("/") + "/invocations"
    req_headers = {
        **headers,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    response_id: str | None = None
    last_seq = 0
    idx = 0

    print(f"-> POST {url}")
    print(f"-> payload: {json.dumps(payload)}")

    with httpx.stream(
        "POST", url, headers=req_headers, json=payload, timeout=timeout
    ) as r:
        print(f"<- HTTP {r.status_code}  content-type={r.headers.get('content-type')}")
        if r.status_code >= 400:
            body = r.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {r.status_code}: {body}")

        for event in parse_sse_lines(r.iter_lines()):
            idx += 1
            if response_id is None and event.get("response_id"):
                response_id = event["response_id"]
                print(f"    captured response_id={response_id}")
            seq = event.get("sequence_number") or event.get("seq")
            if isinstance(seq, int):
                last_seq = max(last_seq, seq)
            print_event(event, idx)

    return response_id, last_seq


def stream_reconnect(
    endpoint: str,
    headers: dict[str, str],
    response_id: str,
    starting_after: int,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> None:
    """GET /responses/{id}?stream=true&starting_after=N — resume after disconnect."""
    url = (
        endpoint.rstrip("/") + f"/responses/{response_id}"
        f"?stream=true&starting_after={starting_after}"
    )
    req_headers = {**headers, "Accept": "text/event-stream"}
    print(f"\n-> GET (reconnect) {url}")

    with httpx.stream("GET", url, headers=req_headers, timeout=timeout) as r:
        print(f"<- HTTP {r.status_code}")
        if r.status_code >= 400:
            body = r.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Reconnect HTTP {r.status_code}: {body}")
        for idx, event in enumerate(parse_sse_lines(r.iter_lines()), start=1):
            print_event(event, idx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--endpoint",
        default=COMARKETER_ENDPOINT,
        help=f"Databricks App URL (default: {COMARKETER_ENDPOINT})",
    )
    ap.add_argument("--query", default="how is my email CTR last week")
    ap.add_argument("--sp-id", default=DEFAULT_SP_ID)
    ap.add_argument("--user-name", default=DEFAULT_USER_NAME)
    ap.add_argument("--no-background", action="store_true", help="Disable background mode")
    ap.add_argument(
        "--reconnect",
        action="store_true",
        help="After the POST stream closes, resume via GET /responses/{id}",
    )
    args = ap.parse_args()

    print("-- Authenticating with Databricks SDK (OAuth M2M) --")
    headers = get_auth_headers()
    print(f"    got bearer token (len={len(headers['Authorization'])})")

    payload = build_payload(
        args.query,
        sp_id=args.sp_id,
        user_name=args.user_name,
        stream=True,
        background=not args.no_background,
    )

    t0 = time.time()
    response_id, last_seq = stream_invocation(args.endpoint, headers, payload)
    elapsed = time.time() - t0
    print(
        f"\n-- stream closed | response_id={response_id} "
        f"last_seq={last_seq} elapsed={elapsed:.1f}s --"
    )

    if args.reconnect and response_id:
        stream_reconnect(args.endpoint, headers, response_id, last_seq)


if __name__ == "__main__":
    main()
