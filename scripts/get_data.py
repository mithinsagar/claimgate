"""Fetch the public inputs needed to rebuild DelucionQA locally (nothing here is redistributed by this repo).

  third_party/DelucionQA/                Bosch Research release (labels, answers, contexts as word ids), git clone
  data/raw/ragbench_delucionqa/*.parquet RAGBench's DelucionQA split, which carries the retrieved passages as text
  data/raw/corvicai_delucionqa/          2023 Jeep Gladiator owner's manual PDF, used only to re-case lower-cased words
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co/datasets"


def curl(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    subprocess.run(["curl", "-sfL", "--retry", "5", "-C", "-", "-o", str(dest), url], check=True)
    print("ok", dest.relative_to(ROOT))


def main() -> None:
    dqa = ROOT / "third_party" / "DelucionQA"
    if not dqa.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/boschresearch/DelucionQA", str(dqa)], check=True)
    for split in ("train", "validation", "test"):
        curl(f"{HF}/galileo-ai/ragbench/resolve/main/delucionqa/{split}-00000-of-00001.parquet", ROOT / f"data/raw/ragbench_delucionqa/{split}.parquet")
    pdf = ROOT / "data/raw/corvicai_delucionqa/manual.pdf"
    curl(f"{HF}/corvicai/delucionqa/resolve/main/Jeep_Gladiator_2023_manual.pdf", pdf)
    txt = pdf.with_name("manual_flow.txt")
    if not txt.exists():
        if not shutil.which("pdftotext"):
            raise SystemExit("pdftotext not found: install poppler (brew install poppler / apt install poppler-utils)")
        subprocess.run(["pdftotext", str(pdf), str(txt)], check=True, stderr=subprocess.DEVNULL)
    print("data ready")


if __name__ == "__main__":
    main()
