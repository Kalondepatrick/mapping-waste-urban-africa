**TABLE I**
**Comparison of Mapping Approaches on the Held-Out Partition**

| Approach | Segmentation | Val. F1 | F1 | Precision | Recall | IoU | FPR | Gap |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| GeoBIA, random forest | OTB meanshift | 0.407 | 0.447 (0.007) | 0.572 | 0.367 | 0.288 | 0.0093 | 56.1 |
| U-Net, from scratch | none | 0.753 | 0.693 (0.043) | 0.699 | 0.690 | 0.532 | 0.0102 | 32.7 |
| DINOv3 ViT-L (SAT-493M), fine-tuned | none | 0.822 | 0.736 (0.020) | 0.699 | 0.778 | 0.583 | 0.0114 | 32.7 |

*Waste class, 216 spatially clean held-out tiles, label 99 excluded, mean of three seeds with the standard deviation in parentheses, argmax decision rule. FPR is the false-positive rate. Gap is the achieved FPR divided by the FPR that precision 0.5 would require at the landscape prevalence of 0.045%; a directly usable map would have a gap of 1.0. Models are selected on the validation split by the one-standard-error rule (highest validation F1, then lowest deployability gap among configurations within one standard error of it) and only then evaluated on the held-out partition. The rule selects DINOv3 ViT-L (SAT-493M). Val. F1 is shown so that the selection can be checked; it is not a performance claim.*
