"""Download model files into models/<name>/ with resume, stall detection and size checks.

Plain HTTPS through curl: on some networks the Hugging Face client hangs on large files, and a stalled CDN connection is
cheaper to drop and resume than to wait out. Usage: python scripts/fetch_models.py [nli] [embed] [llm]
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "nli": ("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "deberta-v3-large-nli",
            ["config.json", "model.safetensors", "spm.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"]),
    "embed": ("BAAI/bge-small-en-v1.5", "bge-small-en-v1.5",
              ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "modules.json",
               "sentence_bert_config.json", "config_sentence_transformers.json", "1_Pooling/config.json"]),
    "llm": ("mlx-community/Qwen2.5-7B-Instruct-4bit", "qwen2.5-7b-instruct-4bit",
            ["config.json", "model.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "vocab.json", "merges.txt", "added_tokens.json"]),
}


def sizes(repo: str) -> dict[str, int]:
    out, stack = {}, [""]
    while stack:
        sub = stack.pop()
        url = f"https://huggingface.co/api/models/{repo}/tree/main" + (f"/{sub}" if sub else "")
        for x in json.load(urllib.request.urlopen(url, timeout=60)):
            if x["type"] == "directory":
                stack.append(x["path"])
            else:
                out[x["path"]] = x["size"]
    return out


def fetch(repo: str, dest: Path, files: list[str], attempts: int = 60) -> None:
    expected = sizes(repo)
    for f in files:
        target = dest / f
        target.parent.mkdir(parents=True, exist_ok=True)
        want = expected[f]
        for _ in range(attempts):
            have = target.stat().st_size if target.exists() else 0
            if have == want:
                break
            if have > want:
                target.unlink()
            subprocess.run(["curl", "-sfL", "-C", "-", "--speed-limit", "100000", "--speed-time", "20", "-o", str(target),
                            f"https://huggingface.co/{repo}/resolve/main/{f}"], check=False)
        have = target.stat().st_size if target.exists() else 0
        if have != want:
            sys.exit(f"failed: {repo}/{f} ({have} of {want} bytes)")
        print(f"ok {repo}/{f} {want / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    for key in sys.argv[1:] or ["nli", "embed"]:
        repo, name, files = MODELS[key]
        fetch(repo, ROOT / "models" / name, files)
