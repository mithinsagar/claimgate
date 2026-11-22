# ClaimGate

[![ci](https://github.com/mithinsagar/claimgate/actions/workflows/ci.yml/badge.svg)](https://github.com/mithinsagar/claimgate/actions/workflows/ci.yml)

ClaimGate checks an answer from a retrieval-augmented assistant against the documents that were retrieved for it, one
claim at a time, before the answer reaches a user. Numbers and units are checked with arithmetic; everything else by
entailment models and a calibrated risk model. Every verdict comes with its proof and the evidence it was judged
against, and claims too close to call go to a person.

It is evaluated on **DelucionQA** (Sadat et al., EMNLP 2023 Findings): answers from a retrieval-augmented assistant for
a car owner's manual, labelled sentence by sentence by people.

## Results

### DelucionQA test set (671 answers)

![DelucionQA test macro F1](docs/figures/delucionqa.png)

| System | Test macro F1 | 95% CI | Test AUC | ClaimGate better? (paired bootstrap) |
|---|---|---|---|---|
| Keyword-match (paper) | 52.77 | | | |
| Sim-cosine (paper) | 69.45 | | | |
| **Sim-overlap (paper, best published)** | **71.09** | | | |
| Sim-overlap, rerun on the rebuilt data | 71.14 | 67.6 to 74.6 | | +6.9, p < 0.001 |
| MiniCheck-Flan-T5-Large, zero-shot | 71.27 | 67.7 to 74.7 | 0.795 | +6.7, p < 0.001 |
| Local LLM judge, Qwen2.5-7B-Instruct 4-bit | 73.09 | 69.5 to 76.6 | 0.811 | +4.9, p = 0.004 |
| HHEM-2.1-Open, zero-shot | 74.97 | 71.5 to 78.3 | 0.844 | +3.0, p = 0.022 |
| DeBERTa-v3-large NLI, zero-shot | 75.48 | 72.0 to 78.9 | 0.841 | +2.5, p = 0.049 |
| **ClaimGate** | **78.00** | **74.6 to 81.2** | **0.855** | |
| ClaimGate variant: fine-tuned verifier (not selected on dev) | 78.11 | 74.8 to 81.2 | 0.858 | -0.1, p = 0.53 |

* ClaimGate is 6.9 points above the best result in the DelucionQA paper and significantly better than HHEM-2.1,
  MiniCheck, a local 7B LLM judge and zero-shot NLI on the same answers (the last only just: p = 0.049).
* On the 367 test answers whose retrieved context never appears in training, it scores 76.7 (the fine-tuned verifier
  variant 78.9). On the 647 answers whose rebuilt context is complete, 77.7.
* Sentence-level calibration error is 0.022: a reported risk of 30% means roughly that.
* The shipped model uses no fine-tuned network: DeBERTa NLI and HHEM run zero-shot, and only a logistic regression on
  top is trained.

### How the final model was chosen

Model choices were made on train (5-fold cross-validation grouped by answer), thresholds on dev, and test was scored
once per system. The rules are written in `scripts/evaluate_delucionqa.py`.

| Decision | Evidence (train CV answer AUC unless noted) | Outcome |
|---|---|---|
| Add HHEM-2.1 as a signal | 0.873 to 0.881 | added |
| Add MiniCheck as a signal | +0.0001 | not added |
| Train on rule-made contradictions of train sentences | 0.881 to 0.834 | not used (they look grounded but are labelled unsupported) |
| Keep a contradiction gate | hand rule: catches 45% of held-out-fold rule-made contradictions vs 10% without; no dev cost | kept |
| Replace the hand rule with a learned detector | at a 0.5% false-alarm target: 65% caught, dev F1 unchanged; at 3%: 83% caught but -1.2 dev F1 | learned detector, 0.5% |
| How to use the fine-tuned verifier | dev macro F1: feature model 82.4, blend 81.8, verifier alone 81.2 | feature model |
| Split compound sentences into atomic claims (local LLM) | dev macro F1 82.4 to 80.4 (209 sentences split) | not adopted |

A note on the verifier: a first run stacked the verifier's out-of-fold scores directly and turned it down. Train
analysis then showed why: each fold model's scores sit on a different scale, so pooling them misleads the stacker,
while within every fold the verifier beat zero-shot NLI (answer AUC +0.02 to +0.04). The final rule picks among the
feature model, the verifier and a blend on dev. Dev picked the feature model; the verifier variant scored 0.1 points
higher on test, a tie, and promoting it would mean choosing on test. It ships as an option. Likewise, the learned
contradiction detector's false-alarm target was first fixed at 3%, which failed the dev-cost rule; the target is now
chosen on dev from 3%, 2%, 1% and 0.5%.

The learned contradiction detector is a logistic regression over zero-shot signals (NLI contradiction and entailment,
lexical grounding, the quantity checks, HHEM), trained to separate rule-made contradictions of train sentences from
supported train sentences. Atomic splitting asked the local Qwen2.5-7B to break compound sentences into
self-contained facts and let the riskiest part count; it flagged more correct answers than it caught extra
hallucinations, so it stays out (`results/atomic_claims.json`).

### QuantStress: controlled edits of supported claims (held out)

DelucionQA has few wrong numbers and almost no contradictions; most of its hallucinations are true-sounding advice
the manual never gives. `scripts/quantstress.py` takes dev and test sentences that annotators marked Supported and
edits them by rule. The rule-made contradictions used in training come from train answers only, and sentences that
also appear in dev or test are excluded, so this set is held out.

![QuantStress flag rates](docs/figures/quantstress.png)

| Share of claims flagged | original (should pass) | unit converted (should pass) | value shifted | unit swapped | negation flipped | condition swapped |
|---|---|---|---|---|---|---|
| NLI zero-shot | 7.0% | 13.9% | 70.8% | 41.7% | 68.0% | 55.8% |
| HHEM-2.1-Open | 9.8% | 56.2% | 39.6% | 27.8% | 27.3% | 9.2% |
| MiniCheck-Flan-T5-Large | 8.2% | 54.9% | 82.6% | 44.4% | 88.5% | 67.2% |
| **ClaimGate** | **4.8%** | **9.7%** | 96.5% | 91.7% | 64.2% | 47.0% |
| ClaimGate variant: fine-tuned verifier | 9.0% | 29.2% | **100.0%** | **100.0%** | **98.2%** | **96.0%** |

* ClaimGate raises the fewest false alarms on untouched claims and on correctly converted values, and catches 96.5% of
  wrong values and 92% of swapped units: the unit checker rewrites "176 N·m" into the evidence's "130 ft-lb" before
  any model sees it, and a wrong number is caught by arithmetic.
* The fine-tuned verifier learned contradictions from rule-made training edits and catches nearly all of them, at
  the price of more false alarms. Which variant to run depends on whether missed contradictions or blocked correct
  answers cost more.
* The verifier used to flag 48% of correct conversions. Two causes, both fixed: the rebuilt contexts write
  temperatures as "10f (12c)" (the degree sign was stripped), which the checker ignored; and for dual-unit sentences
  such as "500 miles (805 km)" the generator converted only the first number, producing "805 km (805 km)", a pattern
  the verifier had learned to read as tampering. The converted edit now replaces the whole phrase.

### The number checker against human labels (RAGTruth)

The synthetic test above flatters a rule-based checker, so it was also run on every RAGTruth response (no tuning on
RAGTruth). When it calls a number a mismatch, the sentence overlaps a human-annotated hallucination 34% of the time on
QA and 28% on news summaries, and it catches 2 to 3% of annotated hallucinations that contain a number. Most numeric
hallucinations there are numbers absent from the source, which the checker leaves to the learned model; part of the
remaining flags are real disagreements the annotators did not mark (for example "preheat to 500°F" where the source
says 325°F). Running it exposed several parsing bugs (times read as amps, units glued to numbers, fractions in ranges, Celsius read as
Fahrenheit, numbers about different things compared), all fixed and covered by tests (`tests/test_quantities*.py`).
Details: `results/quantity_checker_ragtruth.json`.

### Out of domain: RAGTruth question answering (900 test responses)

The models trained on car-manual answers, applied unchanged to RAGTruth's web-search QA responses (Niu et al., ACL
2024). Response-level precision / recall / F1 of the hallucinated class, as in the paper. Thresholds re-picked on 300
random RAGTruth train responses; nothing retrained.

| System | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Prompted gpt-4-turbo (paper) | 33.2 | 90.6 | 45.6 | |
| Llama-2-13B fine-tuned on RAGTruth (paper) | 61.6 | 76.3 | 68.2 | |
| DeBERTa-v3-large NLI, zero-shot | 31.7 | 75.0 | 44.6 | 0.757 |
| MiniCheck-Flan-T5-Large, zero-shot | 33.1 | 78.1 | 46.5 | 0.780 |
| **HHEM-2.1-Open, zero-shot** | **51.1** | **73.1** | **60.2** | **0.870** |
| ClaimGate, trained on DelucionQA only | 37.4 | 74.4 | 49.8 (44.2 to 55.3) | 0.810 |
| ClaimGate, decision layer retrained on 100 RAGTruth responses | 47.1 | 71.9 | 56.9 (50.9 to 62.5) | 0.847 |
| ClaimGate, decision layer retrained on 300 RAGTruth responses | 41.2 | 86.2 | 55.8 (50.4 to 61.0) | 0.856 |
| ClaimGate, decision layer retrained on 600 RAGTruth responses | 42.6 | 87.5 | 57.3 (51.8 to 62.5) | 0.861 |

Out of domain HHEM is clearly better than ClaimGate trained on car-manual answers: the decision layer learned
DelucionQA's convention (true but unstated advice counts as a hallucination) and carries it over. Retraining only that
layer on RAGTruth responses (disjoint from the 300 used for thresholds; regularisation chosen on those 300) closes most
of the gap, from 49.8 to 57.3 F1 and from 0.810 to 0.861 AUC, but does not pass HHEM on test. On the 300-response
threshold set the retrained layer was ahead of HHEM (AUC 0.862 vs 0.854); on the 900 test responses it is behind.

### Abstaining on close calls

![Accuracy vs coverage](docs/figures/abstention.png)

Deferring the answers closest to the threshold to a person raises accuracy on the rest from 79.6% (deciding everything)
to 83.4% at 80% coverage and 89.1% at 60%. The service marks these answers `REVIEW`; the band is sized on dev so about
20% of answers land there.

### Speed

0.50 s median per answer (about 150 ms per claim) and 1.4 s at the 90th percentile on an Apple M5 on mains power,
measured on 60 real test answers with an empty NLI cache. Long-evidence NLI runs only for claims that no short evidence
window already entails, which halved the median. The committed `results/latency.json` was re-run later in macOS Low
Power Mode on battery, with other apps using the GPU: 1.1 s median, 3.8 s at the 90th percentile.

## What goes in and what comes out

```
POST /verify
{
  "evidence": "... Tighten the lug nuts in a star pattern to 130 ft-lb (176 N·m). Check the torque again after the first 25 miles.
               Tire pressure should be checked at least once a month when the tires are cold. ...
               The spare tire is for temporary use only and should not be driven above 50 mph.",
  "question": "How do I put the wheel back on and what should I check afterwards?",
  "answer":   "Tighten the lug nuts in a star pattern to 176 N·m. The spare tire can be driven at up to 70 mph.
               Check the tire pressure once a month when the tires are hot. Rotate your tires every 5,000 miles to extend their life.",
  "policy":   "strict"
}
```

Output of the shipped model (abridged):

```
verdict: BLOCK
  SUPPORTED        risk 0.03  "Tighten the lug nuts in a star pattern to 176 N·m."
                   '176 N·m' matches evidence '130 ft-lb' after unit conversion; entailed by the evidence window (p=0.99)
  CONTRADICTED     risk 0.99  "The spare tire can be driven at up to 70 mph."
                   claim '70 mph' but evidence says '50 mph' (40% difference)
  CONTRADICTED     risk 0.99  "Check the tire pressure once a month when the tires are hot."
                   contradiction detector p=0.98: an evidence window contradicts the claim (p=1.00) and
                   none entails it (p=0.00)
  NOT_IN_EVIDENCE  risk 0.78  "Rotate your tires every 5,000 miles to extend their life."
                   no quantity like '5,000 miles' in the evidence; best window: entailment 0.00
```

![ClaimGate web UI](docs/figures/ui.png)

`policy: "strict"` blocks any unsupported claim (DelucionQA's convention); `policy: "contradictions"` blocks only
contradictions and reports unsupported claims as warnings.

## How it works

```
answer ──► split into claims ──► for each claim:
             1. quantities    pint-based extraction and unit conversion; a value that disagrees with an evidence
                              number about the same thing (shared nearby words, same clause) is CONTRADICTED
             2. canonicalise  values equal to an evidence value after conversion are rewritten into the evidence's
                              units, so "176 N·m" is not judged unsupported against "130 ft-lb"
             3. evidence      DeBERTa-v3-large NLI over the claim's best evidence windows (and long chunks only when
                              needed), HHEM-2.1, lexical novelty, bge-small similarity
             4. decide        calibrated logistic regression -> risk; overrides from step 1 and from a learned
                              contradiction detector (trained on rule-made contradictions of train sentences, set
                              for 0.5% false alarms); review band around the threshold
answer verdict = PASS | BLOCK | REVIEW   (an answer is as strong as its weakest claim)
```

The fine-tuned verifier variant replaces step 4's feature model with the average of three DeBERTa-v3-large models
fine-tuned on DelucionQA train plus rule-made contradictions, keeping steps 1, 2 and the contradiction detector.

### Fine-tuning on Apple silicon

The first attempt went NaN after one step. The cause was not MPS: the published checkpoint stores float16 weights, and
AdamW's second-moment estimates underflow to zero in float16, so the update divides by zero. Training now keeps
float32 master weights with float16 autocast, freezes the embeddings and the lower 16 of 24 layers, and uses the claim's
best evidence windows as the premise (about 4.5 pairs/s on an M5). The three folds run in separate processes because
MPS keeps memory between folds. `scripts/train_verifier.py`.

## Rebuilding DelucionQA

Bosch released the retrieved contexts as word ids, and the crawler that turns them back into text no longer works
because the manual's website blocks automated browsers. ClaimGate rebuilds them from the passages RAGBench republished
plus the manual's own text: 5,190 of 5,367 word ids, zero conflicts, zero alphabetical-order inversions, 99.97% of
test tokens. Bosch's Sim-overlap rerun on the rebuilt data scores 71.14 against the published 71.09.
[docs/data.md](docs/data.md).

## Reproduce

Python 3.12 and poppler (`pdftotext`). Timings on an Apple M5.

```bash
make setup          # venv + dependencies
make data           # DelucionQA, RAGBench passages, manual text -> data/delucionqa/*.jsonl        (~2 min)
make models         # DeBERTa-v3-large NLI, bge-small, HHEM-2.1-Open, MiniCheck-Flan-T5-Large (~4.5 GB)
make features       # NLI features for train, rule-made edits, dev, test, RAGTruth adaptation  (~1.5 hours, cached)
make verifier       # 3-fold fine-tuning and scoring                                             (~2.5 hours)
make detectors      # HHEM and MiniCheck scores for every set                                    (~1.5 hours)
make evaluate       # model selection on train, thresholds on dev, test once -> results/delucionqa.*
make stress ragtruth quantity-checker figures
make atomic         # atomic-claim splitting with the local LLM (dev; test only if adopted)
make test           # 41 tests: unit, property-based (Hypothesis), API; no model weights needed
```

The LLM-judge baseline and the atomic-claims experiment need `pip install mlx mlx-lm` (Apple silicon) and
`python scripts/fetch_models.py llm`. CI runs the unit tests, a Docker build with a health check, and an end-to-end
test of the service with real weights (a tiny NLI model plus the real HHEM and bge-small), cached between runs.

## Run the service

```bash
make serve                                    # http://localhost:8060 ; UI at /, API at POST /verify
CLAIMGATE_MODEL=models/claimgate_verifier.joblib make serve    # fine-tuned verifier variant
CLAIMGATE_API_KEY=... make serve              # require the X-API-Key header
docker build -t claimgate . && docker run -p 8060:8060 -v "$PWD/models:/app/models" claimgate
```

Open `http://localhost:8060/#example` to run the example above.

## Limitations

* **Rebuilt contexts.** 24 of 671 test contexts still contain unrecovered words, and contexts RAGBench did not carry
  verbatim were re-cased by rule. Bosch's method reproduces within 0.05 points, but it is a rebuild.
* **One labelling convention.** The decision layer learned DelucionQA's rule that true but unstated advice is a
  hallucination. On RAGTruth, HHEM alone stays ahead even after retraining the decision layer on 600 RAGTruth answers.
* **Wide intervals.** With 671 test answers the 95% interval is about ±3.3 points. ClaimGate vs zero-shot NLI is only
  just significant (p = 0.049), and ClaimGate vs its verifier variant is a tie.
* **Rule-based numbers.** On RAGTruth the checker's mismatch calls overlap human labels only about a third of the
  time, and it does not catch numbers that are simply absent from the source.
* **Contradictions in the default model.** It catches 64% of flipped negations and 47% of swapped conditions; the
  verifier variant catches 98% and 96% but raises more false alarms (9% on untouched claims, 29% on correct
  conversions).
* **Claims are sentences.** LLM-based atomic splitting was tried and lowered dev F1, so a sentence with two facts
  still gets one verdict.
* **English only.**

## Licence

Code: MIT. No dataset or manual text is redistributed; see [docs/data.md](docs/data.md) for sources and licences.
