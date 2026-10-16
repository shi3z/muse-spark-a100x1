#!/usr/bin/env python3
"""Drive the server through a set of training prompts (disjoint from bench_client.py's) so the kernel-side histogram
(GGML_MMID_HIST) reflects decode-time routing. usage: collect_decode_prompts.py PORT N_PREDICT"""
import json, sys, urllib.request
PORT = int(sys.argv[1]); N = int(sys.argv[2])
prompts = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain the difference between TCP and UDP to a junior developer.",
    "Implement binary search in Python and explain its complexity.",
    "日本の四季について、それぞれの季節の特徴を説明してください。",
    "What are the main causes of inflation, and how do central banks respond?",
    "Give me a 5-day itinerary for a first visit to Kyoto.",
    "Prove that the square root of 2 is irrational.",
    "Write a SQL query that finds the top 3 customers by total order value per month, and explain it.",
    "Summarize the plot of Hamlet in one paragraph and list its major themes.",
    "How does a transformer language model generate text? Keep it under 150 words.",
]
for i, p in enumerate(prompts):
    req = {"messages": [{"role": "user", "content": p}], "max_tokens": N, "temperature": 0.7, "stream": False}
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}), timeout=3600)
    j = json.loads(r.read()); t = j.get("timings", {})
    print(f"[{i}] {t.get('predicted_n')} tok @ {t.get('predicted_per_second', 0):.2f} tok/s", flush=True)
