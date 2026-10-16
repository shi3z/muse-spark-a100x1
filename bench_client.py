#!/usr/bin/env python3
"""Measure llama-server prefill / decode speed for Muse-Spark: uses /completion timings (prompt_ms, predicted_ms)."""
import json, sys, time, urllib.request
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8020
N_PREDICT = int(sys.argv[2]) if len(sys.argv) > 2 else 128
prompts = {
    "short": "Explain how CRISPR gene editing works, in about 200 words.",
    "code": "Write a thread-safe bounded blocking queue in Rust with a short explanation.",
    "long": "Summarize the following policy in three sentences.\n" + " ".join(f"Clause {i}: the committee shall review item {i} of the annual procedure and record the outcome in register {i % 7}." for i in range(1, 120)),
}
for name, p in prompts.items():
    req = {"messages": [{"role": "user", "content": p}], "max_tokens": N_PREDICT, "temperature": 0, "stream": False}
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}), timeout=3600)
    j = json.loads(r.read()); dt = time.time() - t0
    t = j.get("timings", {}); u = j.get("usage", {})
    spec = ""
    if t.get("draft_n"):
        spec = f", draft {t['draft_n_accepted']}/{t['draft_n']} accepted ({100*t['draft_n_accepted']/t['draft_n']:.0f}%)"
    print(f"[{name}] wall {dt:.1f}s prompt {u.get('prompt_tokens')} tok @ {t.get('prompt_per_second', 0):.1f} tok/s, decode {u.get('completion_tokens')} tok @ {t.get('predicted_per_second', 0):.2f} tok/s{spec} | {(j['choices'][0]['message'].get('content') or j['choices'][0]['message'].get('reasoning_content') or '')[:90]!r}", flush=True)
