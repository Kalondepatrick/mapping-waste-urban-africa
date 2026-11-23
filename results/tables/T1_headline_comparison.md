**Table 1. Waste-class accuracy on the spatially clean held-out partition (216 tiles, label 99 excluded, 3 seeds, argmax rule). F1 reported as mean (SD). The deployability gap is the achieved false-positive rate divided by the rate required for precision 0.5 at landscape prevalence (0.045%).**

| Approach | Segmentation required | F1 | Precision | Recall | IoU | FPR | Deployability gap |
|---|---|---|---|---|---|---|---|
| GeoBIA, random forest          | yes, automated (OTB meanshift) | 0.4475 (0.0066)                | 0.5724                         | 0.3673                         | 0.2883                         | 0.0093                         | 56.1 x                         |
| U-Net, trained from scratch | no                          | 0.6932 (0.0428)             | 0.6992                      | 0.6904                      | 0.5315                      | 0.0102                      | 32.7 x                      |
| U-Net with pretrained backbone (best) | no                                    | 0.7412 (0.0166)                       | 0.7337                                | 0.7493                                | 0.5890                                | 0.0093                                | 27.6 x                                |
