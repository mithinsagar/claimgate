Share of claims flagged. Lower is better for original and converted; higher is better for the four edits.

| System | original (n=400) | converted (n=144) | value_shift (n=144) | unit_swap (n=144) | negation (n=400) | condition_swap (n=400) |
|---|---|---|---|---|---|---|
| NLI zero-shot, no unit canonicalisation | 7.0% | 19.4% | 70.8% | 41.7% | 68.0% | 55.8% |
| NLI zero-shot | 7.0% | 13.9% | 70.8% | 41.7% | 68.0% | 55.8% |
| HHEM-2.1-Open zero-shot | 9.8% | 56.2% | 39.6% | 27.8% | 27.3% | 9.2% |
| MiniCheck-Flan-T5-Large zero-shot | 8.2% | 54.9% | 82.6% | 44.4% | 88.5% | 67.2% |
| Fine-tuned verifier alone | 9.0% | 28.5% | 97.2% | 96.5% | 98.0% | 96.0% |
| ClaimGate, no fine-tuned network | 4.8% | 9.7% | 96.5% | 91.7% | 64.2% | 47.0% |
| ClaimGate | 4.8% | 9.7% | 96.5% | 91.7% | 64.2% | 47.0% |
| ClaimGate variant: fine-tuned verifier | 9.0% | 29.2% | 100.0% | 100.0% | 98.2% | 96.0% |
