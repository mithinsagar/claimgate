"""Download model files into models/<name>/ with parallel byte ranges, stall restarts and SHA-256 checks.

Plain HTTPS through curl: on some networks the Hugging Face client hangs on large files and single CDN connections
stall, so large files are fetched as several byte ranges in parallel and verified against the LFS hash.
Usage: python scripts/fetch_models.py [nli] [embed] [hhem] [flan-t5-base] [minicheck] [llm]
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "nli": ("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "deberta-v3-large-nli",
            ["config.json", "model.safetensors", "spm.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"]),
    "embed": ("BAAI/bge-small-en-v1.5", "bge-small-en-v1.5",
              ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "modules.json",
               "sentence_bert_config.json", "config_sentence_transformers.json", "1_Pooling/config.json"]),
    "hhem": ("vectara/hallucination_evaluation_model", "hhem-2.1-open", ["config.json", "model.safetensors"]),
    "flan-t5-base": ("google/flan-t5-base", "flan-t5-base",
                     ["config.json", "spiece.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]),
    "minicheck": ("lytang/MiniCheck-Flan-T5-Large", "minicheck-flan-t5-large",
                  ["config.json", "pytorch_model.bin", "spiece.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json"]),
    "llm": ("mlx-community/Qwen2.5-7B-Instruct-4bit", "qwen2.5-7b-instruct-4bit",
            ["config.json", "model.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "vocab.json", "merges.txt", "added_tokens.json"]),
}
PARTS = 6
BIG = 50_000_000


def listing(repo: str) -> dict[str, dict]:
    out, stack = {}, [""]
    while stack:
        sub = stack.pop()
        url = f"https://huggingface.co/api/models/{repo}/tree/main" + (f"/{sub}" if sub else "") + "?expand=true"
        for x in json.load(urllib.request.urlopen(url, timeout=60)):
            if x["type"] == "directory":
                stack.append(x["path"])
            else:
                out[x["path"]] = {"size": x["size"], "sha256": (x.get("lfs") or {}).get("oid")}
    return out


def curl_range(url: str, dest: Path, start: int, end: int, attempts: int = 40) -> None:
    want = end - start + 1
    for _ in range(attempts):
        have = dest.stat().st_size if dest.exists() else 0
        if have == want:
            return
        if have > want:
            dest.unlink()
            have = 0
        subprocess.run(["curl", "-sfL", "--speed-limit", "200000", "--speed-time", "15", "-r", f"{start + have}-{end}", "-o", "-", url],
                       stdout=open(dest, "ab"), check=False)
    raise SystemExit(f"failed range {start}-{end} of {url}")


def fetch_file(repo: str, name: str, meta: dict, target: Path) -> None:
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    size = meta["size"]
    if target.exists() and target.stat().st_size == size:
        pass
    elif size < BIG:
        curl_range(url, target, 0, size - 1)
    else:
        step = -(-size // PARTS)
        ranges = [(i * step, min(size, (i + 1) * step) - 1) for i in range(PARTS)]
        parts = [target.with_name(f"{target.name}.part{i}") for i in range(PARTS)]
        with ThreadPoolExecutor(PARTS) as ex:
            list(ex.map(lambda a: curl_range(url, a[0], *a[1]), zip(parts, ranges, strict=True)))
        with open(target, "wb") as out:
            for p in parts:
                out.write(p.read_bytes())
                p.unlink()
    if meta["sha256"]:
        h = hashlib.sha256()
        with open(target, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        if h.hexdigest() != meta["sha256"]:
            target.unlink()
            raise SystemExit(f"sha256 mismatch for {repo}/{name}; deleted, run again")
    print(f"ok {repo}/{name} {size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    for key in sys.argv[1:] or ["nli", "embed", "hhem", "flan-t5-base", "minicheck"]:
        repo, name, files = MODELS[key]
        meta = listing(repo)
        for f in files:
            fetch_file(repo, f, meta[f], ROOT / "models" / name / f)
