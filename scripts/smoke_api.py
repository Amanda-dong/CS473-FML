from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _request(
    method: str, url: str, payload: Dict[str, Any] | None = None
) -> Tuple[int, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = Request(url=url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body)
    except HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
        return exc.code, parsed
    except URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test core API endpoints.")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000", help="API base URL"
    )
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    # (method, url, payload, required_keys)
    # required_keys: dot-path strings; "query.train_window" means body["query"]["train_window"]
    checks = [
        ("GET", f"{base}/health", None, []),
        ("GET", f"{base}/datasets", None, []),
        (
            "POST",
            f"{base}/predict/cmf",
            {
                "concept_subtype": "ramen",
                "limit": 3,
                "risk_tolerance": "balanced",
                "price_tier": "mid",
            },
            ["query.train_window", "query.model_version", "recommendations"],
        ),
        (
            "POST",
            f"{base}/predict/trajectory",
            {"concept_subtype": "ramen"},
            ["train_window", "model_version", "trajectory_cluster"],
        ),
    ]

    failed = False
    for method, url, payload, required_keys in checks:
        status, body = _request(method, url, payload)
        ok = 200 <= status < 300
        print(f"[{'OK' if ok else 'FAIL'}] {method} {url} -> {status}")
        if not ok:
            print(f"  Body: {body}")
            failed = True
            continue
        for key_path in required_keys:
            parts = key_path.split(".")
            node = body
            missing = False
            for part in parts:
                if not isinstance(node, dict) or part not in node:
                    missing = True
                    break
                node = node[part]
            if missing:
                print(f"  [FAIL] missing field: {key_path}")
                failed = True

    if failed:
        return 1

    print("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
