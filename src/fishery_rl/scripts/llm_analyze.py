from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


def _call_openai_compatible(base_url: str, model: str, api_key: str, prompt: str, timeout_sec: int = 120) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert RL researcher for safety-critical control."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=base_url.rstrip("/") + "/chat/completions",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def main():
    p = argparse.ArgumentParser(description="Send benchmark results to an LLM and save a deep analysis report.")
    p.add_argument("--summary", type=str, required=True, help="Path to benchmark summary.json")
    p.add_argument("--out", type=str, default=None, help="Output path for markdown report. Defaults next to summary.")
    p.add_argument("--llm-base-url", type=str, default="https://openrouter.ai/api/v1")
    p.add_argument("--llm-model", type=str, default="deepseek/deepseek-chat-v3-0324:free")
    p.add_argument("--llm-api-key-env", type=str, default="OPENROUTER_API_KEY")
    args = p.parse_args()

    with open(args.summary, "r", encoding="utf-8") as f:
        summary = json.load(f)

    api_key = os.getenv(args.llm_api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable: {args.llm_api_key_env}")

    prompt = (
        "Analyze these RL benchmark results for fishery sustainability and safety.\n"
        "Focus on long-horizon return, violation-rate tradeoffs, risk metrics (CVaR), and why "
        "different algorithms behaved differently.\n"
        "Provide concrete recommendations for next experiments.\n\n"
        f"RESULTS_JSON:\n{json.dumps(summary, indent=2)}"
    )

    try:
        analysis = _call_openai_compatible(
            base_url=args.llm_base_url,
            model=args.llm_model,
            api_key=api_key,
            prompt=prompt,
        )
    except (KeyError, IndexError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        raise RuntimeError(f"LLM request failed: {e}") from e

    out = args.out or os.path.join(os.path.dirname(args.summary), "llm_analysis.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(analysis)
    print(out)


if __name__ == "__main__":
    main()
