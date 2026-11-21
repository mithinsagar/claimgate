"""Reconstruct DelucionQA contexts (released by Bosch Research as word ids) without crawling mopar.com.

Bosch encoded each retrieved context as ids into a vocabulary made of the sorted unique tokens of the crawled manual
(lower-cased, character whitelist, NLTK word_tokenize). The site now blocks automated access, so the official crawler
cannot run. RAGBench (galileo-ai/ragbench, CC-BY-4.0) republished the DelucionQA questions with the retrieved passages as
text. This script:
  1. tokenises every RAGBench passage exactly like Bosch's script,
  2. aligns each DelucionQA context (id sequence) with the passages of the same question that have the same token
     pattern (same length and same positions of repeated ids), collecting id -> token votes,
  3. aligns near matches with difflib and accepts a new id -> token pair only if it keeps the alphabetical order,
  3b. recovers more ids from the manual's own text: unknown ids between known neighbours are read off the places where
     the same neighbours occur in the manual (used only if it recovers >= 99% of hidden known ids correctly),
  4. checks that the mapping has no conflicting votes and is strictly increasing in alphabetical order, which is what a
     sorted vocabulary implies (an independent check that the decoding is right),
  5. tries an order-based gap fill, but uses it only if it recovers >= 99% of hidden known ids correctly,
  6. decodes every context; unknown ids become "[?]" and each row records its coverage.
The cased RAGBench passage text is kept when a context matched one exactly.
Outputs (local only, never committed: the manual text is Stellantis copyright): data/delucionqa/{train,dev,test}.jsonl
"""
from __future__ import annotations

import ast
import bisect
import collections
import csv
import difflib
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
from nltk import word_tokenize

csv.field_size_limit(10**9)
ROOT = Path(__file__).resolve().parents[1]
DQA = ROOT / "third_party/DelucionQA/data/DelucionQA_final"
RB = ROOT / "data/raw/ragbench_delucionqa"
OUT = ROOT / "data/delucionqa"
MANUAL = ROOT / "data/raw/corvicai_delucionqa/manual_flow.txt"
# tokens Bosch's script appends to the vocabulary by hand
BOSCH_SCRIPT = ROOT / "third_party/DelucionQA/src/context_reconstruction/main_convert_context_to_textual.py"
WL = set("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890 !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~'")


def bosch_tokens(text: str) -> list[str]:
    text = text.lower()
    text = "".join(filter(WL.__contains__, text))
    text = re.sub(r"\.(?!(?<=\d\.)\d) ?", ". ", text)
    return word_tokenize(text)


def pattern(seq) -> tuple:
    first = {}
    return tuple(first.setdefault(x, len(first)) for x in seq)


def exact_votes(rows, passages):
    votes = collections.defaultdict(collections.Counter)
    match_text = {}
    for i, r in enumerate(rows):
        c = r["ctx"]
        pc = pattern(c)
        for toks, text in passages.get(r["Question"].strip(), {}).items():
            if len(toks) == len(c) and pattern(toks) == pc:
                for a, b in zip(c, toks, strict=True):
                    votes[a][b] += 1
                match_text[i] = text
                break
    return votes, match_text


def fuzzy_rounds(rows, passages, mapping, rounds: int = 3) -> int:
    total = 0
    for rnd in range(rounds):
        fz = collections.defaultdict(collections.Counter)
        for r in rows:
            c = r["ctx"]
            if all(a in mapping for a in c):
                continue
            seq = [mapping.get(a, f"<{a}>") for a in c]
            best, best_ratio = None, 0.0
            for toks in passages.get(r["Question"].strip(), {}):
                sm = difflib.SequenceMatcher(a=seq, b=list(toks), autojunk=False)
                if sm.ratio() > best_ratio:
                    best, best_ratio = (sm, toks), sm.ratio()
            if not best or best_ratio < 0.8:
                continue
            sm, toks = best
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "replace" and i2 - i1 == j2 - j1:
                    for k in range(i2 - i1):
                        if c[i1 + k] not in mapping:
                            fz[c[i1 + k]][toks[j1 + k]] += 1
        ids_k = sorted(mapping)
        added = 0
        for a, cnt in fz.items():
            tok = cnt.most_common(1)[0][0]
            pos = bisect.bisect_left(ids_k, a)
            if (pos == 0 or mapping[ids_k[pos - 1]] < tok) and (pos == len(ids_k) or tok < mapping[ids_k[pos]]):
                mapping[a] = tok
                ids_k.insert(pos, a)
                added += 1
        print(f"fuzzy round {rnd + 1}: added {added} ids")
        total += added
        if not added:
            break
    return total


def manual_alignment(rows, mapping: dict, manual: list[str], max_run: int = 3, n_left: int = 3, n_right: int = 2) -> dict:
    """Recover unknown ids by finding their known neighbours in the manual's own token stream.

    For a run of up to `max_run` unknown ids with `n_left` known tokens on the left and `n_right` on the right, every place
    in the manual where those neighbours occur with the same gap votes for the tokens in between. An id is assigned only when
    all of its votes agree, the token is not already used by another id, and the assignment keeps the vocabulary in
    alphabetical order.
    """
    index: dict[tuple, list[int]] = {}
    for i in range(len(manual) - n_left + 1):
        index.setdefault(tuple(manual[i:i + n_left]), []).append(i + n_left)
    votes: dict[int, dict[str, int]] = {}
    for r in rows:
        toks = [mapping.get(a) for a in r["ctx"]]
        i = 0
        while i < len(toks):
            if toks[i] is not None:
                i += 1
                continue
            j = i
            while j < len(toks) and toks[j] is None:
                j += 1
            run = j - i
            left, right = toks[max(0, i - n_left):i], toks[j:j + n_right]
            if run <= max_run and len(left) == n_left and None not in left and len(right) == n_right and None not in right:
                for p in index.get(tuple(left), []):
                    if manual[p + run:p + run + n_right] == right:
                        for k in range(run):
                            v = votes.setdefault(r["ctx"][i + k], {})
                            v[manual[p + k]] = v.get(manual[p + k], 0) + 1
            i = j
    used = set(mapping.values())
    ids_k = sorted(mapping)
    out = {}
    for a, v in votes.items():
        if a in mapping or len(v) != 1:
            continue
        tok = next(iter(v))
        pos = bisect.bisect_left(ids_k, a)
        if tok in used or tok in out.values():
            continue
        if (pos == 0 or mapping[ids_k[pos - 1]] < tok) and (pos == len(ids_k) or tok < mapping[ids_k[pos]]):
            out[a] = tok
    return out


CASCADE = [(3, 3, 2), (3, 2, 2), (3, 3, 1), (2, 2, 1)]  # (max_run, n_left, n_right), strictest first


def manual_cascade(rows, mapping: dict, manual: list[str]) -> int:
    added = 0
    for run, n_left, n_right in CASCADE:
        for _ in range(4):
            new = manual_alignment(rows, mapping, manual, run, n_left, n_right)
            mapping.update(new)
            added += len(new)
            if not new:
                break
    return added


def gap_fill(known: dict, ref_sorted: list[str]) -> dict:
    """An id strictly between mapped ids a < x < b must be a token between their tokens; assign when the count is forced."""
    out = {}
    ids_k = sorted(known)
    for a, b in zip(ids_k, ids_k[1:], strict=False):
        gap = b - a - 1
        if gap <= 0:
            continue
        lo = bisect.bisect_right(ref_sorted, known[a])
        hi = bisect.bisect_left(ref_sorted, known[b])
        if hi - lo == gap:
            for k, t in enumerate(ref_sorted[lo:hi]):
                out[a + 1 + k] = t
    return out


def ordered_labels(r) -> tuple[list[str], list[str]]:
    sents = ast.literal_eval(r.get("Answer_sent_tokenized") or "[]")
    labels = ast.literal_eval(r.get("Sentence_labels") or "{}")
    keyed = {}
    for s in sents:
        m = re.match(r"^(R\d+):\s*", s)
        keyed[m.group(1) if m else s] = s[m.end():].strip() if m else s.strip()
    if labels and set(labels) == set(keyed):
        keys = sorted(keyed, key=lambda k: int(k[1:]))
        return [keyed[k] for k in keys], [labels[k] for k in keys]
    return list(keyed.values()), list(labels.values())


def main() -> int:
    rows = []
    for sp in ("train", "dev", "test", "unanswerable"):
        for r in csv.DictReader(open(DQA / f"{sp}.csv", encoding="utf-8")):
            r["split"], r["ctx"] = sp, ast.literal_eval(r["Context"])
            rows.append(r)
    rb = pd.concat([pd.read_parquet(RB / f"{s}.parquet") for s in ("train", "validation", "test")])
    passages = collections.defaultdict(dict)  # question -> {token tuple: original text}
    for _, r in rb.iterrows():
        for d in r.documents:
            passages[r.question.strip()][tuple(bosch_tokens(d))] = d.strip()

    votes, match_text = exact_votes(rows, passages)
    mapping, conflicts = {}, 0
    for a, cnt in votes.items():
        (tok, _), *rest = cnt.most_common()
        mapping[a] = tok
        conflicts += sum(m for _, m in rest)
    exact_ids = len(mapping)
    fuzzy_ids = fuzzy_rounds(rows, passages, mapping)

    manual_toks = bosch_tokens(MANUAL.read_text(encoding="utf-8", errors="ignore")) if MANUAL.exists() else []
    manual_ids, manual_validation = 0, {}
    if manual_toks:
        rng0 = random.Random(1)
        hide = set(rng0.sample(sorted(mapping), len(mapping) // 10))
        trial0 = {k: v for k, v in mapping.items() if k not in hide}
        manual_cascade(rows, trial0, manual_toks)
        rec = [k for k in hide if k in trial0]
        manual_validation = {"hidden": len(hide), "recovered": len(rec), "correct": sum(trial0[k] == mapping[k] for k in rec)}
        manual_validation["precision"] = manual_validation["correct"] / max(1, len(rec))
        print("manual alignment validation on hidden known ids:", manual_validation)
        if manual_validation["precision"] >= 0.99:
            manual_ids = manual_cascade(rows, mapping, manual_toks)
            print(f"manual alignment added {manual_ids} ids")

    reference = {t for qs in passages.values() for toks in qs for t in toks}
    reference.update(manual_toks)
    reference.update(re.findall(r"unique_tokens\.append\('([^']*)'\)", BOSCH_SCRIPT.read_text()))
    ref_sorted = sorted(reference)
    rng = random.Random(0)
    proven = dict(mapping)
    hidden = set(rng.sample(sorted(proven), len(proven) // 10))
    trial = gap_fill({k: v for k, v in proven.items() if k not in hidden}, ref_sorted)
    recovered = [k for k in hidden if k in trial]
    correct = sum(trial[k] == proven[k] for k in recovered)
    validation = {"hidden": len(hidden), "recovered": len(recovered), "correct": correct, "precision": correct / max(1, len(recovered))}
    filled = {k: v for k, v in gap_fill(proven, ref_sorted).items() if k not in proven}
    used = validation["precision"] >= 0.99
    if used:
        mapping.update(filled)
    print(f"gap fill validation {validation}: {'used' if used else 'not used'} ({len(filled)} candidate ids)")

    ids = sorted(mapping)
    inversions = sum(1 for x, y in zip(ids, ids[1:], strict=False) if not mapping[x] < mapping[y])

    OUT.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    by_split = collections.defaultdict(list)
    for i, r in enumerate(rows):
        missing = [a for a in r["ctx"] if a not in mapping]
        stats[(r["split"], "complete" if not missing else "partial")] += 1
        sents, labels = ordered_labels(r)
        by_split[r["split"]].append({
            "sample_id": r.get("sample_id") or f"{r['split']}_{i}", "split": r["split"], "retrieval": r.get("Retreival Setting", ""),
            "question": r["Question"], "context_lower": " ".join(mapping.get(a, "[?]") for a in r["ctx"]), "context": match_text.get(i),
            "complete": not missing, "coverage": 1 - len(missing) / len(r["ctx"]), "answer": r["Answer"], "answer_sentences": sents,
            "sentence_labels": labels, "label": r.get("Label", ""), "answerable": r.get("Answerable", ""), "does_not_answer": r.get("Does_not_answer", ""),
        })
    for sp, items in by_split.items():
        with open(OUT / f"{sp}.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
    report = {f"{sp}_{k}": v for (sp, k), v in sorted(stats.items())}
    report.update({"contexts": len(rows), "matched_to_passage_exactly": len(match_text), "ids_from_exact_matches": exact_ids,
                   "ids_from_fuzzy_alignment": fuzzy_ids,
                   "ids_from_manual_alignment": manual_ids,
                   "manual_alignment_validation": manual_validation, "ids_mapped": len(mapping),
                   "distinct_ids_in_data": len({a for r in rows for a in r["ctx"]}),
                   "conflicting_votes": conflicts, "order_inversions": inversions, "gap_fill_validation": validation, "gap_fill_used": used})
    (OUT / "reconstruction_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0 if inversions == 0 and conflicts == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
