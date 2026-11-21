"""Text helpers: sentence splitting, content words, and re-casing of the lower-cased DelucionQA contexts.

About 40% of reconstructed contexts exist only in Bosch's lower-cased token form. NLI models read cased, detokenised
text better, so those contexts are detokenised and re-cased with a word -> most-frequent-surface-form table built from
the cased passages that were matched exactly. Every context goes through the same path, so cased and re-cased inputs
do not differ in style.
"""
from __future__ import annotations

import collections
import json
import re
from functools import lru_cache

from nltk.tokenize.treebank import TreebankWordDetokenizer

from claimgate.data import DQA_DIR, ROOT, Example

_DETOK = TreebankWordDetokenizer()
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*|\d+(?:\.\d+)?")
STOP = set("""a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him himself his how i
if in into is it its itself just me more most my myself no nor not now of off on once only or other our ours ourselves out over own
same she should so some such than that the their theirs them themselves then there these they this those through to too under until
up very was we were what when where which while who whom why will with would you your yours yourself yourselves also may might must
shall one ensure make sure using use used please note""".split())


@lru_cache(maxsize=1)
def casing_table() -> dict[str, str]:
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for split in ("train", "dev", "test", "unanswerable"):
        p = DQA_DIR / f"{split}.jsonl"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            text = json.loads(line).get("context") or ""
            for w in re.findall(r"[A-Za-z][A-Za-z0-9\-']*", text):
                counts[w.lower()][w] += 1
    manual = ROOT / "data/raw/corvicai_delucionqa/manual_flow.txt"
    if manual.exists():
        for w in re.findall(r"[A-Za-z][A-Za-z0-9\-']*", manual.read_text(errors="ignore")):
            counts[w.lower()][w] += 1
    return {k: c.most_common(1)[0][0] for k, c in counts.items()}


def recase(lower_tokens: str) -> str:
    table = casing_table()
    toks = [table.get(t, t) if t != "[?]" else "…" for t in lower_tokens.split()]
    text = _DETOK.detokenize(toks)
    text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text


_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(•])")


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_END.split(text) if len(s.strip()) > 1]
    return parts or ([text.strip()] if text.strip() else [])


def context_for_nli(ex: Example) -> str:
    """Evidence as the NLI model sees it: user-supplied evidence verbatim, DelucionQA contexts re-cased."""
    return ex.meta.get("evidence") or recase(ex.context)


def content_words(text: str) -> list[str]:
    return [w for w in (m.lower() for m in _WORD.findall(text)) if w not in STOP and len(w) > 1]


def stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w
