"""Remote chat models (the judge: gpt-6.1-sol, medium reasoning effort), by default through the OpenAI API
(OPENAI_API_KEY); `provider="openrouter"` keeps the older route (OPENROUTER_API_KEY, model "openai/gpt-6.1-sol").

    from natsteer.remote import chat_many
    results = chat_many([prompt1, prompt2], model="gpt-6.1-sol", effort="medium", workers=8)
    # each: {"text", "in", "out", "reasoning", "cost", "sec"} or {"error"}

The key comes from natsteer.apikeys (never printed, never put in the environment). Errors carry the status code
and the provider's message only (no message for 401, which can quote part of the key), never the request headers.
Every call returns its cost, so callers can keep a running total and stop at a budget (`budget_usd`): OpenRouter
reports the billed cost; OpenAI does not, so the cost is estimated from the token counts at PRICE_PER_M (an upper
bound, since cached input is billed less).
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from natsteer.apikeys import get_key

PROVIDERS = {
    "openai": {"url": "https://api.openai.com/v1/chat/completions", "key": "OPENAI_API_KEY"},
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions", "key": "OPENROUTER_API_KEY"},
}
JUDGE_MODEL = "gpt-6.1-sol"
PRICE_PER_M = {"in": 2.0, "out": 10.0}  # USD per million tokens for gpt-6.1-sol (the OpenRouter list price; OpenAI's is the same)


class BudgetExceeded(RuntimeError):
    pass


def chat(client: httpx.Client, headers: dict, prompt: str, model: str, effort: str, max_retries: int = 4, provider: str = "openai") -> dict:
    if provider == "openai":
        body = {"model": model.removeprefix("openai/"), "reasoning_effort": effort, "messages": [{"role": "user", "content": prompt}]}
    else:
        body = {"model": model if "/" in model else f"openai/{model}", "reasoning": {"effort": effort}, "usage": {"include": True},
                "messages": [{"role": "user", "content": prompt}]}
    t0 = time.time()
    for attempt in range(max_retries + 1):
        try:
            r = client.post(PROVIDERS[provider]["url"], headers=headers, json=body)
        except httpx.HTTPError as e:
            err = f"{type(e).__name__}"
        else:
            d = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            if r.status_code == 200 and "choices" in d:
                u = d.get("usage") or {}
                cost = u.get("cost") or (u.get("prompt_tokens", 0) * PRICE_PER_M["in"] + u.get("completion_tokens", 0) * PRICE_PER_M["out"]) / 1e6
                return {"text": d["choices"][0]["message"].get("content") or "", "in": u.get("prompt_tokens"),
                        "out": u.get("completion_tokens"), "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                        "cost": cost, "sec": round(time.time() - t0, 1)}
            err = f"HTTP {r.status_code}: {'' if r.status_code == 401 else (d.get('error') or {}).get('message', '')[:300]}"
            if r.status_code in (400, 401, 402, 403):  # bad request, bad key, no credit: retrying will not help
                return {"error": err}
        time.sleep(2 ** attempt)
    return {"error": err}


def ledger_total(ledger) -> float:
    """Total spend recorded in a ledger file (jsonl of {"time", "tag", "calls", "cost"})."""
    p = Path(ledger)
    return sum(json.loads(line)["cost"] for line in p.read_text().splitlines() if line.strip()) if p.exists() else 0.0


def chat_many(prompts: list[str], model: str = JUDGE_MODEL, effort: str = "medium", workers: int = 8,
              budget_usd: float | None = None, timeout: float = 600, ledger=None, cap_usd: float | None = None,
              tag: str = "", provider: str = "openai") -> list[dict]:
    """Run prompts in parallel. Calls not yet started are skipped once the spend reaches budget_usd, or, with a
    ledger and cap_usd, once ledger total + this run's spend reaches cap_usd. The run's spend is appended to the
    ledger (also when it fails part way)."""
    if ledger is not None and cap_usd is not None:
        left = cap_usd - ledger_total(ledger)
        if left <= 0:
            raise BudgetExceeded(f"ledger {ledger} is at its cap ${cap_usd:.2f}")
        budget_usd = left if budget_usd is None else min(budget_usd, left)
    headers = get_key(PROVIDERS[provider]["key"]).auth_header()
    spent, lock = [0.0], threading.Lock()

    def one(p):
        with lock:
            if budget_usd is not None and spent[0] >= budget_usd:
                return {"error": "skipped: budget reached"}
        res = chat(client, headers, p, model, effort, provider=provider)
        with lock:
            spent[0] += res.get("cost", 0.0)
        return res

    try:
        with httpx.Client(timeout=timeout) as client, ThreadPoolExecutor(workers) as ex:
            out = list(ex.map(one, prompts))
    finally:
        del headers
        if ledger is not None:
            with open(ledger, "a") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tag": tag, "model": model,
                                    "calls": len(prompts), "cost": spent[0]}) + "\n")
    return out
