| System | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Prompt gpt-3.5-turbo (published, Niu et al. 2024) | 18.8 | 84.4 | 30.8 | |
| Prompt gpt-4-turbo (published, Niu et al. 2024) | 33.2 | 90.6 | 45.6 | |
| SelfCheckGPT gpt-3.5-turbo (published, Niu et al. 2024) | 35.0 | 58.0 | 43.7 | |
| LMvLM gpt-4-turbo (published, Niu et al. 2024) | 18.7 | 76.9 | 30.1 | |
| Fine-tuned Llama-2-13B (trained on RAGTruth) (published, Niu et al. 2024) | 61.6 | 76.3 | 68.2 | |
| NLI zero-shot (DeBERTa-v3-large), threshold from RAGTruth 300 | 31.7 | 75.0 | 44.6 (39.1 to 49.7) | 0.757 |
| HHEM-2.1-Open zero-shot, threshold from RAGTruth 300 | 51.1 | 73.1 | 60.2 (54.1 to 66.0) | 0.870 |
| MiniCheck-Flan-T5-Large zero-shot, threshold from RAGTruth 300 | 33.1 | 78.1 | 46.5 (41.1 to 51.6) | 0.780 |
| ClaimGate, threshold from RAGTruth 300 | 37.3 | 72.5 | 49.3 (43.5 to 54.8) | 0.810 |
| ClaimGate, threshold from DelucionQA dev | 32.8 | 89.4 | 48.0 (42.9 to 52.9) | 0.810 |
| ClaimGate without quantity override, threshold from RAGTruth 300 | 37.5 | 72.5 | 49.5 (43.7 to 55.0) | 0.811 |
| ClaimGate without quantity override, threshold from DelucionQA dev | 32.8 | 89.4 | 48.0 (42.9 to 52.9) | 0.811 |
| ClaimGate, no fine-tuned network, threshold from RAGTruth 300 | 37.3 | 72.5 | 49.3 (43.5 to 54.8) | 0.810 |
| ClaimGate, no fine-tuned network, threshold from DelucionQA dev | 32.8 | 89.4 | 48.0 (42.9 to 52.9) | 0.810 |
| ClaimGate, no fine-tuned network without quantity override, threshold from RAGTruth 300 | 37.5 | 72.5 | 49.5 (43.7 to 55.0) | 0.811 |
| ClaimGate, no fine-tuned network without quantity override, threshold from DelucionQA dev | 32.8 | 89.4 | 48.0 (42.9 to 52.9) | 0.811 |
| ClaimGate variant: fine-tuned verifier, threshold from RAGTruth 300 | 30.4 | 83.8 | 44.6 (39.6 to 49.6) | 0.790 |
| ClaimGate variant: fine-tuned verifier, threshold from DelucionQA dev | 29.1 | 90.6 | 44.1 (39.3 to 48.8) | 0.790 |
| ClaimGate variant: fine-tuned verifier without quantity override, threshold from RAGTruth 300 | 30.4 | 83.8 | 44.6 (39.6 to 49.6) | 0.791 |
| ClaimGate variant: fine-tuned verifier without quantity override, threshold from DelucionQA dev | 29.1 | 90.6 | 44.1 (39.3 to 48.8) | 0.791 |
