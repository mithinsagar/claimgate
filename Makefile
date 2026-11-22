PY ?= .venv/bin/python

.PHONY: setup data models features verifier detectors evaluate stress ragtruth quantity-checker atomic figures all serve test lint

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && $(PY) -c "import nltk; [nltk.download(p, quiet=True, raise_on_error=True) for p in ('punkt', 'punkt_tab')]"

data:        ## DelucionQA ids + RAGBench passages + manual text -> data/delucionqa/*.jsonl (docs/data.md)
	$(PY) scripts/get_data.py
	$(PY) scripts/reconstruct_delucionqa.py

models:      ## DeBERTa-v3-large NLI, bge-small, HHEM-2.1-Open (+ Flan-T5-base tokenizer), MiniCheck-Flan-T5-Large
	$(PY) scripts/fetch_models.py nli embed hhem flan-t5-base minicheck

features:    ## zero-shot NLI and grounding features for train, rule-made contradictions, dev, test
	$(PY) scripts/build_features.py --splits dev test train train_aug ragtruth_train600

verifier:    ## cross-fitted fine-tuning (about 1.5 h on an M5) and its scores; one process per fold so MPS memory is released
	for k in 0 1 2; do $(PY) scripts/train_verifier.py --only-fold $$k || exit 1; done
	$(PY) scripts/score_verifier.py

detectors:   ## HHEM and MiniCheck scores for every evaluation set
	$(PY) scripts/score_detectors.py hhem
	$(PY) scripts/score_detectors.py minicheck

evaluate:    ## model selection on train, thresholds on dev, test once -> results/delucionqa.*
	$(PY) scripts/evaluate_delucionqa.py

stress:
	$(PY) scripts/quantstress.py

ragtruth:
	$(PY) scripts/evaluate_ragtruth.py

quantity-checker:
	$(PY) scripts/eval_quantity_checker.py

atomic:      ## needs mlx, mlx-lm and the Qwen2.5-7B weights (scripts/fetch_models.py llm)
	$(PY) scripts/atomic_claims.py --stage dev

figures:
	$(PY) scripts/make_figures.py

all: features verifier detectors evaluate stress ragtruth quantity-checker figures

serve:
	PYTHONPATH=src .venv/bin/uvicorn claimgate.serve:app --port 8060

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src scripts tests
