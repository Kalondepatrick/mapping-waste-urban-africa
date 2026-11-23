**Table 5. Every reported configuration ranked by deployability gap. No configuration approaches 1.0, which is the value a directly usable map would require.**

| Configuration | Family | F1 | Recall | FPR | Implied precision at 0.045% | Gap |
|---|---|---|---|---|---|---|
| torch_scratch_hn | PyTorch baseline | 0.1054           | 0.0579           | 0.0004           | 0.0922           | 11.2             |
| nw_scratch_w0.25 | Negative weight  | 0.4834           | 0.3423           | 0.0024           | 0.0620           | 15.2             |
| torch_dinov2_hn  | PyTorch baseline | 0.5498           | 0.4276           | 0.0034           | 0.0693           | 16.0             |
| torch_resnet34_hn | PyTorch baseline  | 0.5535            | 0.4256            | 0.0033            | 0.0627            | 16.5              |
| nw_scratch_w0.5 | Negative weight | 0.0969          | 0.0578          | 0.0003          | 0.0576          | 17.6            |
| torch_efficientnetb0_hn | PyTorch baseline        | 0.5581                  | 0.4334                  | 0.0040                  | 0.0509                  | 20.4                    |
| fx_scratch_n639 | Label budget    | 0.6992          | 0.6424          | 0.0066          | 0.0434          | 22.8            |
| fx_dinov3convnext_n320 | Label budget           | 0.6923                 | 0.6409                 | 0.0070                 | 0.0397                 | 24.3                   |
| nw_scratch_w0.1 | Negative weight | 0.5577          | 0.4428          | 0.0049          | 0.0392          | 24.6            |
| v6_satlasaerial_finetuned | Pretrained backbone       | 0.7383                    | 0.7307                    | 0.0085                    | 0.0379                    | 25.8                      |
| torch_dinov2     | PyTorch baseline | 0.6723           | 0.6187           | 0.0075           | 0.0394           | 26.5             |
| fx_dinov3sat_n320 | Label budget      | 0.7488            | 0.7615            | 0.0093            | 0.0356            | 27.1              |
| fx_dinov3convnext_n639 | Label budget           | 0.7276                 | 0.7206                 | 0.0089                 | 0.0355                 | 27.4                   |
| v6_dinov3vitl_finetuned | Pretrained backbone     | 0.7412                  | 0.7493                  | 0.0093                  | 0.0354                  | 27.6                    |
| fx_openearthmap_n320 | Label budget         | 0.7139               | 0.7035               | 0.0091               | 0.0337               | 28.9                 |
| v6_satlasaerial_frozen | Pretrained backbone    | 0.6800                 | 0.6435                 | 0.0085                 | 0.0332                 | 29.2                   |
| v6_dinov3vitl_frozen | Pretrained backbone  | 0.7288               | 0.7409               | 0.0100               | 0.0338               | 29.5                 |
| torch_scratch    | PyTorch baseline | 0.7121           | 0.7063           | 0.0095           | 0.0327           | 29.7             |
| fx_openearthmap_n639 | Label budget         | 0.7003               | 0.6852               | 0.0093               | 0.0322               | 30.0                 |
| fx_efficientnetb0_n639 | Label budget           | 0.6843                 | 0.6599                 | 0.0092                 | 0.0319                 | 30.7                   |
| fx_dinov3sat_n639 | Label budget      | 0.7450            | 0.7827            | 0.0109            | 0.0314            | 30.9              |
| torch_resnet34   | PyTorch baseline | 0.7053           | 0.7020           | 0.0098           | 0.0312           | 31.1             |
| torch_efficientnetb0 | PyTorch baseline     | 0.6807               | 0.6567               | 0.0093               | 0.0312               | 31.3                 |
| v6_dinov3satvitl_frozen | Pretrained backbone     | 0.7021                  | 0.6990                  | 0.0100                  | 0.0307                  | 31.7                    |
| v6_efficientnetb4oem_finetuned | Pretrained backbone            | 0.6999                         | 0.6944                         | 0.0099                         | 0.0308                         | 31.7                           |
| fx_dinov3sat_n64 | Label budget     | 0.7007           | 0.7048           | 0.0104           | 0.0333           | 32.1             |
| fx_resnet34_n320 | Label budget     | 0.6973           | 0.6950           | 0.0102           | 0.0309           | 32.4             |
| fx_resnet34_n639 | Label budget     | 0.7262           | 0.7541           | 0.0110           | 0.0300           | 32.5             |
| fx_scratch_n320 | Label budget    | 0.6933          | 0.6852          | 0.0100          | 0.0307          | 32.5            |
| v6_dinov3satvitl_finetuned | Pretrained backbone        | 0.7361                     | 0.7781                     | 0.0114                     | 0.0297                     | 32.7                       |
| unet_f16_clean   | PyTorch baseline | 0.6932           | 0.6904           | 0.0102           | 0.0299           | 32.7             |
| fx_efficientnetb0_n320 | Label budget           | 0.6743                 | 0.6524                 | 0.0097                 | 0.0306                 | 32.9                   |
| v6_efficientnetb4imagenet_finetuned | Pretrained backbone                 | 0.7083                              | 0.7236                              | 0.0109                              | 0.0291                              | 33.5                                |
| fx_dinov3vits_n320 | Label budget       | 0.7407             | 0.8073             | 0.0128             | 0.0284             | 35.1               |
| fx_dinov3convnext_n64 | Label budget          | 0.6976                | 0.7295                | 0.0124                | 0.0270                | 37.3                  |
| fx_dinov3vits_n639 | Label budget       | 0.7222             | 0.7893             | 0.0135             | 0.0256             | 38.1               |
| v6_efficientnetb4imagenet_frozen | Pretrained backbone              | 0.6903                           | 0.7193                           | 0.0125                           | 0.0265                           | 38.4                             |
| v6_efficientnetb4oem_frozen | Pretrained backbone         | 0.6781                      | 0.6944                      | 0.0121                      | 0.0258                      | 38.4                        |
| fx_dinov3vits_n64 | Label budget      | 0.7011            | 0.7412            | 0.0128            | 0.0263            | 38.6              |
| fx_openearthmap_n64 | Label budget        | 0.6418              | 0.6375              | 0.0118              | 0.0238              | 41.1                |
| fx_dinov3vits_n128 | Label budget       | 0.6918             | 0.7587             | 0.0150             | 0.0235             | 43.7               |
| fx_efficientnetb0_n128 | Label budget           | 0.5594                 | 0.5062                 | 0.0101                 | 0.0232                 | 44.1                   |
| fx_openearthmap_n128 | Label budget         | 0.6482               | 0.6715               | 0.0137               | 0.0217               | 45.3                 |
| fx_scratch_n64 | Label budget   | 0.5603         | 0.5129         | 0.0108         | 0.0224         | 46.4           |
| fx_dinov3sat_n128 | Label budget      | 0.6857            | 0.7835            | 0.0171            | 0.0203            | 48.5              |
| fx_dinov3convnext_n128 | Label budget           | 0.6595                 | 0.7329                 | 0.0171                 | 0.0209                 | 51.7                   |
| fx_efficientnetb0_n64 | Label budget          | 0.6030                | 0.6250                | 0.0154                | 0.0183                | 54.5                  |
| geobia_otb_rf        | GeoBIA (pixel level) | 0.4475               | 0.3673               | 0.0093               | 0.0175               | 56.1                 |
| fx_scratch_n128 | Label budget    | 0.5548          | 0.5343          | 0.0134          | 0.0186          | 57.0            |
| fx_resnet34_n64 | Label budget    | 0.6279          | 0.7006          | 0.0182          | 0.0176          | 57.4            |
| fx_resnet34_n128 | Label budget     | 0.6162           | 0.7842           | 0.0268           | 0.0145           | 74.6             |
