Share of claims flagged. Lower is better for original and converted; higher is better for the four edits.

| System | original (n=400) | converted (n=149) | value_shift (n=149) | unit_swap (n=149) | negation (n=400) | condition_swap (n=400) |
|---|---|---|---|---|---|---|
| NLI zero-shot, no unit canonicalisation | 5.5% | 21.5% | 69.8% | 40.9% | 69.8% | 54.2% |
| NLI zero-shot | 5.5% | 6.7% | 63.1% | 45.6% | 69.8% | 54.2% |
| HHEM-2.1-Open zero-shot | 7.8% | 57.0% | 48.3% | 29.5% | 32.0% | 9.0% |
| MiniCheck-Flan-T5-Large zero-shot | 8.8% | 67.8% | 84.6% | 45.0% | 87.8% | 65.2% |
| Fine-tuned verifier alone | 8.0% | 48.3% | 97.3% | 96.6% | 98.8% | 95.5% |
| ClaimGate, no fine-tuned network | 3.0% | 5.4% | 82.6% | 79.2% | 60.5% | 40.5% |
| ClaimGate | 3.0% | 5.4% | 82.6% | 79.2% | 60.5% | 40.5% |
| ClaimGate variant: fine-tuned verifier | 8.0% | 49.0% | 100.0% | 99.3% | 98.8% | 95.5% |
