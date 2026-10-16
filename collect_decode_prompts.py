#!/usr/bin/env python3
"""Drive the server through a set of training prompts (disjoint from bench_client.py's) so the kernel-side histogram
(GGML_MMID_HIST) reflects decode-time routing. usage: collect_decode_prompts.py PORT N_PREDICT"""
import json, sys, urllib.request
PORT = int(sys.argv[1]); N = int(sys.argv[2]); SET = int(sys.argv[3]) if len(sys.argv) > 3 else 1
prompts_set2 = [
    "Draft a polite email declining a job offer while keeping the door open.",
    "Explain quantum entanglement to a curious 12-year-old.",
    "Write a Go function that reads a CSV file and computes per-column averages, with error handling.",
    "機械学習における過学習とは何か、どう防ぐかを説明してください。",
    "Compare the economic policies of Keynes and Hayek in a short essay.",
    "Design a REST API for a library lending system: list the endpoints and their payloads.",
    "What happened during the Cambrian explosion and why does it matter?",
    "Write a haiku sequence (five haiku) about autumn in a city.",
    "Debug this: a Python script that reads a 10 GB file with readlines() runs out of memory. What should change?",
    "Explain how public-key cryptography lets two strangers communicate securely.",
    "Give a recipe for a vegetarian ramen broth and explain the role of each ingredient.",
    "Describe the main differences between PostgreSQL and MongoDB and when to use each.",
    "Translate the following into natural Japanese: 'The meeting has been moved to Thursday; please update your calendars.'",
    "Write a bash one-liner that finds the 10 largest files under a directory, and explain it.",
    "What is the Monty Hall problem and why is switching better?",
    "Outline a 10-minute talk introducing Rust's ownership model.",
    "Explain how vaccines train the immune system.",
    "Write a product description for noise-cancelling headphones aimed at commuters.",
    "How does garbage collection work in Java? Mention generational collection.",
    "Plan a weekly training schedule for a beginner preparing for a 10 km run in 8 weeks.",
]
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
if SET == 2:
    prompts = prompts_set2
for i, p in enumerate(prompts):
    req = {"messages": [{"role": "user", "content": p}], "max_tokens": N, "temperature": 0.7, "stream": False}
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}), timeout=3600)
    j = json.loads(r.read()); t = j.get("timings", {})
    print(f"[{i}] {t.get('predicted_n')} tok @ {t.get('predicted_per_second', 0):.2f} tok/s", flush=True)
