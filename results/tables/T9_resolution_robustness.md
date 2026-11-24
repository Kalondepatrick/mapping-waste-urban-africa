**TABLE IX**
**Waste-Class F1 Under Simulated Loss of Spatial Resolution**

| Configuration | 3.8 cm | 5.7 cm | 7.6 cm | 11.4 cm | 15.2 cm | 30.5 cm | Retained at 7.6 cm |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv3 ViT-L (SAT-493M), fine-tuned | 0.734 | 0.726 | 0.721 | 0.622 | 0.322 | 0.000 | 98% |
| DINOv3 ViT-L (web), fine-tuned | 0.741 | 0.739 | 0.715 | 0.560 | 0.259 | 0.025 | 96% |
| DINOv3 ViT-L (web), frozen | 0.734 | 0.719 | 0.708 | 0.640 | 0.576 | 0.266 | 96% |
| EfficientNet-B4 (ImageNet), fine-tuned | 0.707 | 0.703 | 0.680 | 0.510 | 0.166 | 0.000 | 96% |
| EfficientNet-B0 (ImageNet) | 0.682 | 0.672 | 0.654 | 0.453 | 0.097 | 0.001 | 96% |
| SatlasPretrain (aerial), frozen | 0.681 | 0.676 | 0.642 | 0.451 | 0.297 | 0.024 | 94% |
| SatlasPretrain (aerial), fine-tuned | 0.739 | 0.728 | 0.690 | 0.514 | 0.285 | 0.021 | 93% |
| EfficientNet-B4 (ImageNet), frozen | 0.691 | 0.667 | 0.637 | 0.405 | 0.095 | 0.000 | 92% |
| ResNet-34 (ImageNet) | 0.707 | 0.691 | 0.647 | 0.447 | 0.120 | 0.000 | 92% |
| DINOv3 ViT-L (SAT-493M), frozen | 0.704 | 0.661 | 0.641 | 0.579 | 0.485 | 0.365 | 91% |
| DINOv2 (web) | 0.678 | 0.618 | 0.492 | 0.290 | 0.189 | 0.082 | 73% |
| U-Net from scratch | 0.716 | 0.689 | 0.462 | 0.020 | 0.000 | 0.000 | 65% |

*Held-out imagery reduced by area averaging and restored to the 256 x 256 evaluation grid by bilinear interpolation, so every value is measured against the same reference mask. The first column is the native ground sample distance of 3.81 cm. Mean of 3 seeds. Rows ordered by the fraction of accuracy retained at twice the pixel size. Masks are never perturbed.*
