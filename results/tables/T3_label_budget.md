**Table 3. Waste-class F1 on held-out data as a function of the number of annotated training tiles, by pretraining domain. Mean of 3 seeds.**

| Backbone | n = 64 | n = 128 | n = 320 | n = 639 |
|---|---|---|---|---|
| DINOv3 ViT-S | 0.7011       | 0.6918       | 0.7407       | 0.7222       |
| DINOv3-SAT | 0.7007     | 0.6857     | 0.7488     | 0.7450     |
| DINOv3 ConvNeXt | 0.6976          | 0.6595          | 0.6923          | 0.7276          |
| OpenEarthMap | 0.6418       | 0.6482       | 0.7139       | 0.7003       |
| ResNet-34 (ImageNet) | 0.6279               | 0.6162               | 0.6973               | 0.7262               |
| EfficientNet-B0 (ImageNet) | 0.6030                     | 0.5594                     | 0.6743                     | 0.6843                     |
| U-Net from scratch | 0.5603             | 0.5548             | 0.6933             | 0.6992             |
