# Rebuilding DelucionQA's contexts

DelucionQA (Sadat et al., EMNLP 2023 Findings, Bosch Research) has 2,038 answers from a retrieval-augmented ChatGPT on
the 2023 Jeep Gladiator owner's manual. Human annotators labelled every answer sentence as Supported, Neither or
Conflicted. An answer counts as Hallucinated if any of its sentences is not Supported.

The release ships the retrieved contexts as integer word ids, not as text. Each id points into a vocabulary made of
the sorted unique tokens of the manual, as Bosch crawled it from mopar.com: lower-cased, filtered to a character
whitelist and run through NLTK `word_tokenize`. The official crawler no longer runs. mopar.com answers automated
browsers with an Akamai "Access Denied" page, and this project does not try to get around that.

## Method

[RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench) (CC BY 4.0) republished the DelucionQA questions with
the retrieved passages as plain text. `scripts/reconstruct_delucionqa.py` uses those passages to recover the
id-to-word mapping:

1. Every RAGBench passage is tokenised exactly the way Bosch's script does it.
2. A DelucionQA context and a passage for the same question match when they have the same length and the same pattern
   of repeated ids. Each such match gives one id-to-token pair per position.
3. Near matches are aligned with `difflib`. A new pair is accepted only if it keeps the vocabulary in alphabetical
   order.
4. Remaining unknown ids are read off the manual's own text: where an unknown id sits between known words, the places
   where the same neighbours occur in the manual vote for the word in between. A vote counts only if all votes agree,
   the word is not used by another id, and alphabetical order holds. This pass is validated first by hiding 10% of the
   known ids and checking how many come back correctly (373 of 374), and would be skipped below 99%.
5. Unknown ids become `[?]`, and every row records its coverage.

## Checks

| Check | Result |
|---|---|
| Distinct ids recovered | 5,190 of 5,367 (4,769 exact, 64 near matches, 357 from the manual text) |
| Ids with conflicting votes (one id seen as two different words) | 0 |
| Order inversions (id order vs alphabetical order of the recovered words) | 0 |
| Manual-text pass, hidden known ids recovered correctly | 373 of 374 |
| Token-level coverage, test split | 99.97% (46 unknown tokens out of 146,296) |
| Test answers whose context has no unknown token | 647 of 671 |
| Bosch's Sim-overlap baseline, rerun on the rebuilt test set | 71.14 macro F1 (published: 71.09) |

The vocabulary is sorted, so a correct decoding has to come out in strictly increasing alphabetical order. The
exact-match pass does not use that order, which makes zero inversions over its 4,769 ids an independent check (the
later passes use the order as a filter, so they are checked by hiding known ids instead). Rerunning Bosch's own method
on the rebuilt data lands within 0.1 points of the published number.

An order-based gap filler (assigning an id when only one word fits alphabetically between its neighbours) was also
tried. It recovered hidden known ids with 98.4% precision. That missed the 99% bar set for it, so it is not used.

Results in this repo are reported on all 671 test answers; the per-system JSON also has the 647 with complete contexts.

## What is and is not in this repo

Nothing from the manual is committed. `data/` and `third_party/` are git-ignored.

| Source | Licence | Use |
|---|---|---|
| DelucionQA labels and answers | CC BY-SA 4.0 | Fetched from Bosch's GitHub |
| RAGBench passages | CC BY 4.0 | Fetched from Hugging Face |
| 2023 Gladiator owner's manual PDF (Stellantis), from the `corvicai/delucionqa` dataset | Owner's copyright | Used locally to recover words and re-case text; never committed |

`make data` downloads all three sources and rebuilds `data/delucionqa/*.jsonl`.
