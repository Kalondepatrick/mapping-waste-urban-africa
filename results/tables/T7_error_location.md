**TABLE VII**
**Spatial Distribution of False Positives on the Held-Out Partition**

| Configuration | Seeds | False-positive pixels | In tiles containing waste | In tiles with no waste |
| :--- | ---: | ---: | ---: | ---: |
| ResNet-34 (ImageNet), hard negatives | 3 | 45,868 | 94.6% | 5.4% |
| EfficientNet-B0 (ImageNet), hard negatives | 3 | 56,908 | 94.0% | 6.0% |
| DINOv2 (web), hard negatives | 3 | 48,189 | 90.7% | 9.3% |
| U-Net from scratch | 3 | 132,608 | 83.8% | 16.2% |
| DINOv2 (web) | 3 | 105,523 | 83.7% | 16.3% |
| ResNet-34 (ImageNet) | 3 | 137,444 | 82.7% | 17.3% |
| EfficientNet-B0 (ImageNet) | 3 | 129,793 | 79.2% | 20.8% |
| U-Net from scratch (TensorFlow) | 3 | 142,557 | 68.8% | 31.2% |
| GeoBIA, OTB segments (RF) | 1 | 124,388 | 62.7% | 37.3% |
| U-Net from scratch, hard negatives | 3 | 4,805 | 39.7% | 60.3% |

*Of the 216 spatially clean held-out tiles, 80 contain at least one waste pixel. A false positive inside such a tile is a delineation error; one in a tile with no waste is a false detection in unfamiliar background. Counts are means over seeds. Reported for every configuration that stored per-tile confusion counts; the backbone comparison of Table IV did not store them, so the selected model is not listed and the pattern should be read as a property of this imagery rather than of one architecture.*
