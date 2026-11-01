# Evaluation: retrieval_nollm

Run: 2026-10-31 19:01

| Metric | Value |
|---|---|
| mode | retrieval (no LLM) |
| answerable_questions | 34 |
| unanswerable_questions | 4 |
| section_hit@1 | 0.647 |
| section_hit@3 | 0.971 |
| section_hit@5 | 1.0 |
| section_hit@10 | 1.0 |
| MRR | 0.787 |
| expected_section_in_llm_context | 1.0 |
| all_expected_sections_in_context | 0.941 |
| abstention_correct_on_unanswerable | 1.0 |
| false_abstention_on_answerable | 0.118 |
| best_score_answerable_median | 0.94 |
| best_score_answerable_min | 0.009 |
| best_score_unanswerable_max | 0.075 |
| min_relevance_threshold | 0.1 |
| median_latency_ms | 3288.5 |
| p90_latency_ms | 3753 |

| ID | Expected | Result | Latency (ms) |
|---|---|---|---|
| q01 | 24, 23 | rank 3: 80EEA, 80EE, 23, 25, 22, 24, 30, 80HHBA | 2118 |
| q02 | 80C | rank 1: 80C, 88, 80DDB, 36, 80U | 2277 |
| q03 | 10, 2 | rank 3: 91, 209, 2, 35C, 10, 54B, 80P, 80PA | 2954 |
| q04 | 6 | rank 1: 6, 80EE, 54F, 27, 23, 80EEA, 94A, 246 | 2765 |
| q05 | 80D | rank 1: 80D, 56, 194D, 30, 80C, 36, 194DA, 80DDB | 1994 |
| q06 | 80E | rank 1: 80E, 193, 80EE, 80EEA, 43B, 40, 58, 25 | 5545 |
| q07 | 80G | rank 1: 80G, 12, 115BBC, 11, 10, 2 | 5629 |
| q08 | 54 | rank 2: 54F, 54, 53, 54GB, 54EC, 54EB, 54EE, 54EA | 2909 |
| q09 | 139 | rank 3: 271, 80M, 139, 115WD, 115BAE, 210, 13, 234A | 3843 |
| q10 | 208, 207 | rank 1: 208, 210, 234C, 273, 115WJ, 211, 234B, 209 | 3559 |
| q11 | 234B, 234C | rank 1: 234B, 215, 218, 234C, 219, 214, 220, 216 | 3396 |
| q12 | 192 | rank 1: 192, 201, 194T, 197, 194M, 16, 271C, 15 | 2541 |
| q13 | 194I | rank 2: 194IB, 194I, 194S, 192, 194C, 203, 80GG, 199 | 2456 |
| q14 | 56 | rank 1: 56, 164, 2, 115BBA, 45, 51, 36, 10 | 3637 |
| q15 | 10 | rank 3: 80CCD, 40A, 10, FOURTHSCHEDULE, 176, 89, 89A, 4 | 3301 |
| q16 | 16 | rank 3: 17, 192, 16, 115BAC, 115JB, 57, 36, 80P | 3481 |
| q17 | 2 | rank 1: 2, 49, 54F, 115C, 10, 43, 28 | 4330 |
| q18 | 71, 71B | rank 1: 71, 71B, 71A, 70, 192, 73, 72, 74 | 2601 |
| q19 | 72 | rank 3: 73, 79, 72, 78, 73A, 72A, 71B, 33A | 3753 |
| q20 | 32 | rank 1: 32, 80I, 41, 32AC, 31, 32AD, 34, 32AB | 3327 |
| q21 | 64 | rank 1: 64, 10, 80GG, 80CCD, 27, 80V, 80U, 222 | 2835 |
| q22 | 87A | rank 1: 87A, 88D, 88C, 88B, 80U, 87, 88A, 89A | 3048 |
| q23 | 115BAC | rank 1: 115BAC, 86, 2, 167B, 115BAB, 192, 174, 115BAE | 3500 |
| q24 | 270A | rank 1: 270A, 271FAA, 272A, 271F, 271GB, 271AAA | 2506 |
| q25 | 149, 148 | rank 1: 149, 148A, 148, 151A, 147, 115WH, 153, 153B | 2669 |
| q26 | 44AD | rank 2: 115JB, 44AD, 44ADA, 43C, 28, 210, 139, 80HHA | 3308 |
| q27 | 44ADA | rank 1: 44ADA, 44A, 288, 28, 30, 44AA, 44AD, 194J | 3417 |
| q28 | 44AB | rank 2: 12A, 44AB, 271B, 288, 115VW, 139, 32AB, 80IA | 3332 |
| q29 | 80TTA | rank 1: 80TTA, 194A, 193, 80CCA, 269T, 80TTB, 94B, 40A | 3182 |
| q30 | 80GG | rank 4: 25A, 30, 194IB, 80GG, 194I, 40A, 23, 80EEA | 3276 |
| q31 | 10 | rank 1: 10, 194IB, 30, 80HHC, 25, 25A, 23, 80GG | 3591 |
| q32 | 40A | rank 1: 40A, 44AB, 194N, 80GGA, 35D, 94B, 80D, 35AD | 2997 |
| q33 | 269SS | rank 2: 271D, 269SS, 193, 269T, 194N, 68, 281, 43CA | 2447 |
| q34 | 80C | rank 1: 80C, 54ED, 80CCD, 80CCE, 80A, 80CCA, 80IC, 10C | 2603 |
| q35 | (none) | miss: 115A, 194O, 80IB, 194J, 92CE, 80HH, 43A, ELEVENTHSCHEDULE | 3473 |
| q36 | (none) | miss: 56, 43CA, 49, 45, 50C, 55, 194IA, 80EEA | 3160 |
| q37 | (none) | miss: 276B, 276, 276BB, 278A, 280, 276C, 245MA, 278B | 3486 |
| q38 | (none) | miss: 28, 2, 49, 47, 115A, 92B, 271G, 80HHC | 3659 |
