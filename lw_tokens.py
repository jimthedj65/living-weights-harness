"""Token-for-token comparison of two servers' greedy replies over the fact and rules panels (see README.md)."""

import argparse
import json
from pathlib import Path

from lw_client import DEFAULT_URL, MAX_TOKENS, CHAT_TIMEOUT_S, post_json, save

HERE = Path(__file__).resolve().parent


def questions(facts, rules):
    """Every question both panels ask, tagged with its panel kind, in a fixed order."""
    out = []
    for fact in facts["facts"]:
        out += [("recall", q) for q in fact["recall"]]
        out.append(("neighbour", fact["neighbour"]["q"]))
    out += [("control", c["q"]) for c in facts["controls"]]
    out += [("general", g["q"]) for g in rules["general"]]
    for rule in rules["rules"]:
        out.append(("rule_recall", rule["recall"]["q"]))
        for prob in rule["problems"]:
            out.append(("from_memory", prob["q"]))
            out.append(("in_context", f"{rule['text']} {prob['q']}"))
    return out


def reply(base, question):
    """Greedy reply text, its server-reported token count and finish reason."""
    body = {"temperature": 0, "max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": question}]}
    data = post_json(f"{base}/v1/chat/completions", body, CHAT_TIMEOUT_S)
    choice = data["choices"][0]
    return choice["message"]["content"] or "", data["usage"]["completion_tokens"], choice["finish_reason"]


def tokenize(base, text):
    """Token ids for text from the server's own tokenizer."""
    return post_json(f"{base}/tokenize", {"prompt": text, "add_special_tokens": False}, CHAT_TIMEOUT_S)["tokens"]


def capture(base, tok_base, qs):
    """Ask every question and keep the reply with its token ids."""
    rows = []
    for kind, q in qs:
        text, n, finish = reply(base, q)
        rows.append({"kind": kind, "q": q, "a": text, "completion_tokens": n, "finish": finish,
                     "ids": tokenize(tok_base, text)})
    return rows


def first_divergence(a, b):
    """Index of the first differing token, or None when the sequences are equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(ref_rows, test_rows):
    """Per-question match and first-divergence token, plus per-kind totals."""
    detail, kinds = [], {}
    for r, t in zip(ref_rows, test_rows, strict=True):
        if r["q"] != t["q"]:
            raise ValueError(f"question order differs: {r['q']!r} vs {t['q']!r}")
        div = first_divergence(r["ids"], t["ids"])
        detail.append({"kind": r["kind"], "q": r["q"], "same": div is None, "first_diff_token": div,
                       "ref_tokens": len(r["ids"]), "test_tokens": len(t["ids"]), "ref": r["a"], "test": t["a"]})
        k = kinds.setdefault(r["kind"], {"same": 0, "total": 0, "first_diff_tokens": []})
        k["total"] += 1
        k["same"] += div is None
        if div is not None:
            k["first_diff_tokens"].append(div)
    return {"kinds": kinds, "detail": detail}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["capture", "compare"])
    p.add_argument("--url", default=DEFAULT_URL, help="server to capture")
    p.add_argument("--tokenize-url", help="server whose tokenizer reads the replies (default: --url)")
    p.add_argument("--facts", default=str(HERE / "facts.json"))
    p.add_argument("--rules", default=str(HERE / "rules.json"))
    p.add_argument("--results", default="results")
    p.add_argument("--label", default="")
    p.add_argument("--ref", help="compare: reference capture JSON (the untaught server)")
    p.add_argument("--test", help="compare: capture JSON to check against it")
    args = p.parse_args()
    if args.phase == "capture":
        qs = questions(json.loads(Path(args.facts).read_text()), json.loads(Path(args.rules).read_text()))
        rows = capture(args.url, args.tokenize_url or args.url, qs)
        save(args.results, "tokens", {"url": args.url, "label": args.label, "rows": rows})
        return
    ref, test = (json.loads(Path(f).read_text()) for f in (args.ref, args.test))
    out = compare(ref["rows"], test["rows"])
    for kind, k in out["kinds"].items():
        print(f"{kind:12} {k['same']}/{k['total']} token-identical; first differing token: {sorted(k['first_diff_tokens'])}")
    save(args.results, "compare", {"ref": args.ref, "test": args.test, "label": args.label, **out})


if __name__ == "__main__":
    main()
