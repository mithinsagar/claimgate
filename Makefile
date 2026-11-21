PY ?= .venv/bin/python

.PHONY: setup data models features evaluate stress finetune serve test lint

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && $(PY) -m nltk.downloader -q punkt punkt_tab

data:  ## DelucionQA ids + RAGBench passages -> data/delucionqa/*.jsonl (see docs/data.md)
	$(PY) scripts/get_data.py
	$(PY) scripts/reconstruct_delucionqa.py

models:
	$(PY) scripts/fetch_models.py nli embed

features:
	$(PY) scripts/build_features.py

evaluate:
	$(PY) scripts/evaluate_delucionqa.py

stress:
	$(PY) scripts/quantstress.py

finetune:
	$(PY) scripts/finetune_verifier.py

serve:
	PYTHONPATH=src .venv/bin/uvicorn claimgate.serve:app --port 8060

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src scripts tests
