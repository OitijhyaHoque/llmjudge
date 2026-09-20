#!/usr/bin/env python3
"""A stand-in for the vLLM OpenAI-compatible server, for testing layer 3 offline.

It answers `/v1/models` and `/v1/chat/completions` with the same shapes vLLM
does, so the real client is exercised end to end -- transport, structured-output
request, parsing, caching, policy -- without a GPU or a model download.

Its "judgement" is a fixed rule, not a model: it reads back the base rate the
prompt supplied and calls a finding `plausible` when that rate is above 0.1%.
That is enough to prove the plumbing and the policy layer. It proves nothing
about a real judge's accuracy, which needs the blind gold set.

    python3 tests/mock_vllm.py --port 8009
"""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL = "mock/judge-1"
BASE_RATE = re.compile(r"fires on ([\d.]+)% of REAL rows")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):            # keep the test output readable
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._json(200, {"object": "list",
                                    "data": [{"id": MODEL, "object": "model"}]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        prompt = req["messages"][-1]["content"]

        rates = [float(m) for m in BASE_RATE.findall(prompt)]
        top = max(rates) if rates else 0.0
        if top > 0.1:
            verdict = "plausible"
            reason = (f"the flagged pattern occurs in {top:.3f}% of real rows, so a "
                      f"real patient can present this way")
        elif top > 0.0:
            verdict = "unsure"
            reason = (f"the pattern occurs in real rows but only at {top:.3f}%; the "
                      f"evidence supplied does not settle it")
        else:
            verdict = "implausible"
            reason = ("the pattern does not occur in the real table and the "
                      "knowledge rows contradict it")

        cols = re.findall(r"^ {2}(\S+) — ", prompt, re.M)
        content = json.dumps({"verdict": verdict, "implicated_columns": cols[:6],
                              "reason": reason,
                              "confidence": 0.8 if verdict != "unsure" else 0.3})
        self._json(200, {
            "id": "mock-1", "object": "chat.completion", "model": MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": len(prompt) // 4,
                      "completion_tokens": len(content) // 4},
        })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8009)
    args = ap.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"mock vLLM on http://127.0.0.1:{args.port}/v1 serving {MODEL}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
