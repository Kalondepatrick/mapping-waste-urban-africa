PAIRED SPATIAL TEST-AREA THRESHOLD SWEEP
=================================================================
Reference: C:\Users\Peter\Documents\GitHub\mapping-waste-urban-africa\05_finetuning\step0\waste.shp
Saved source features: 39
Source fingerprint: 23074454f785f40d3081b20d40ef1d4cef3df4f04b6051dbaef4a1a02b189210
Contiguous reference units: 37
Reference completeness confirmed: True
Thresholds: (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
Minimum predicted object area: 0.25 m2
Matching: 8-connected, one-to-one by IoU, >=1% rasterized reference coverage.
All other valid test-area pixels are non-waste when reference completeness is confirmed.
Raw pixel metrics omit the object-area filter; filtered metrics use the final object map.
Object FP means unmatched predictions; some may be additional fragments overlapping real waste.
Diagnostic best thresholds are descriptive test-area results, not independently validated operational choices.
Reference edits change the denominators and require recalculation. Do not compare pile IDs across versions without the geometry UID/source-row audit.
Source inference settings may differ, so this is a map-product comparison rather than an isolated fine-tuning effect.

Fixed-threshold results:
original @ 0.50: objects=1,529; TP=30; FP=1,499; FN=7; object P/R/F1=0.0196/0.8108/0.0383; pixel P/R/F1/IoU=0.1014/0.8092/0.1802/0.0990
refined @ 0.50: objects=705; TP=29; FP=676; FN=8; object P/R/F1=0.0411/0.7838/0.0782; pixel P/R/F1/IoU=0.2640/0.6880/0.3816/0.2358
original @ 0.90: objects=579; TP=25; FP=554; FN=12; object P/R/F1=0.0432/0.6757/0.0812; pixel P/R/F1/IoU=0.2282/0.4446/0.3016/0.1776
refined @ 0.90: objects=119; TP=23; FP=96; FN=14; object P/R/F1=0.1933/0.6216/0.2949; pixel P/R/F1/IoU=0.4776/0.1630/0.2431/0.1383

Outputs: spatial_object_metrics.csv, spatial_pixel_metrics.csv, spatial_pixel_metrics_raw.csv, paired_object_thresholds.csv, paired_pixel_thresholds.csv, diagnostic_threshold_choices.csv, paired_reference_detection_*.csv, per-threshold reference_detection.csv and predicted_objects.csv, reference snapshots and audit manifest.
Results directory: C:\Users\Peter\Documents\GitHub\mapping-waste-urban-africa\results\spatial_test_comparison\run_20260907_155905_146341