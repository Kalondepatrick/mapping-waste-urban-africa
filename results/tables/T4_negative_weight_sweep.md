**Table 4. Effect of the complementary not-waste loss weight. Raising it lowers the false-positive rate but collapses recall; at weights of 1.0 and above the gap is undefined because recall falls below 1%.**

| Not-waste loss weight | F1 | Precision | Recall | FPR | Implied precision | Gap |
|---|---|---|---|---|---|---|
| 0.10   | 0.5577 | 0.7553 | 0.4428 | 0.0049 | 0.0392 | 24.6 x |
| 0.25   | 0.4834 | 0.8333 | 0.3423 | 0.0024 | 0.0620 | 15.2 x |
| 0.50   | 0.0969 | 0.8135 | 0.0578 | 0.0003 | 0.0576 | 17.6 x |
| 1.00      | 0.0164    | 0.5903    | 0.0085    | 0.0001    | 0.0761    | undefined |
| 2.00      | 0.0173    | 0.2301    | 0.0090    | 0.0001    | 0.0095    | undefined |
| 4.00      | 0.0009    | 0.3233    | 0.0004    | 0.0000    | 0.0995    | undefined |
