| System | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Prompt gpt-3.5-turbo (published, Niu et al. 2024) | 18.8 | 84.4 | 30.8 | |
| Prompt gpt-4-turbo (published, Niu et al. 2024) | 33.2 | 90.6 | 45.6 | |
| SelfCheckGPT gpt-3.5-turbo (published, Niu et al. 2024) | 35.0 | 58.0 | 43.7 | |
| LMvLM gpt-4-turbo (published, Niu et al. 2024) | 18.7 | 76.9 | 30.1 | |
| Fine-tuned Llama-2-13B (trained on RAGTruth) (published, Niu et al. 2024) | 61.6 | 76.3 | 68.2 | |
| NLI zero-shot (DeBERTa-v3-large), threshold from RAGTruth 300 | 30.9 | 70.6 | 43.0 (37.5 to 48.2) | 0.757 |
| HHEM-2.1-Open zero-shot, threshold from RAGTruth 300 | 51.1 | 73.1 | 60.2 (54.1 to 66.0) | 0.870 |
| MiniCheck-Flan-T5-Large zero-shot, threshold from RAGTruth 300 | 33.1 | 78.1 | 46.5 (41.1 to 51.6) | 0.780 |
| ClaimGate, threshold from RAGTruth 300 | 37.4 | 74.4 | 49.8 (44.2 to 55.3) | 0.810 |
| ClaimGate, threshold from DelucionQA dev | 33.0 | 90.6 | 48.4 (43.3 to 53.3) | 0.810 |
| ClaimGate without quantity override, threshold from RAGTruth 300 | 35.8 | 86.2 | 50.5 (45.2 to 55.7) | 0.810 |
| ClaimGate without quantity override, threshold from DelucionQA dev | 33.0 | 90.6 | 48.4 (43.3 to 53.3) | 0.810 |
| ClaimGate, no fine-tuned network, threshold from RAGTruth 300 | 37.4 | 74.4 | 49.8 (44.2 to 55.3) | 0.810 |
| ClaimGate, no fine-tuned network, threshold from DelucionQA dev | 33.0 | 90.6 | 48.4 (43.3 to 53.3) | 0.810 |
| ClaimGate, no fine-tuned network without quantity override, threshold from RAGTruth 300 | 35.8 | 86.2 | 50.5 (45.2 to 55.7) | 0.810 |
| ClaimGate, no fine-tuned network without quantity override, threshold from DelucionQA dev | 33.0 | 90.6 | 48.4 (43.3 to 53.3) | 0.810 |
| ClaimGate variant: fine-tuned verifier, threshold from RAGTruth 300 | 33.5 | 69.4 | 45.2 (39.6 to 50.6) | 0.787 |
| ClaimGate variant: fine-tuned verifier, threshold from DelucionQA dev | 29.2 | 90.6 | 44.1 (39.3 to 48.9) | 0.787 |
| ClaimGate variant: fine-tuned verifier without quantity override, threshold from RAGTruth 300 | 33.7 | 69.4 | 45.4 (39.7 to 50.8) | 0.788 |
| ClaimGate variant: fine-tuned verifier without quantity override, threshold from DelucionQA dev | 29.2 | 90.6 | 44.1 (39.3 to 48.9) | 0.788 |
| ClaimGate adapted on 100 RAGTruth responses, threshold from RAGTruth 300 | 47.1 | 71.9 | 56.9 (50.9 to 62.5) | 0.847 |
| ClaimGate adapted on 300 RAGTruth responses, threshold from RAGTruth 300 | 41.2 | 86.2 | 55.8 (50.4 to 61.0) | 0.856 |
| ClaimGate adapted on 600 RAGTruth responses, threshold from RAGTruth 300 | 42.6 | 87.5 | 57.3 (51.8 to 62.5) | 0.861 |
