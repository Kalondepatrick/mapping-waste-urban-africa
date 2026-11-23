PAIRED SPATIAL WASTE-MAP COMPARISON
=================================================================
Area: data/aoi/testarea.shp; NOT the 222 waste-centred validation tiles.
Reference: C:\Users\Peter\Documents\GitHub\mapping-waste-urban-africa\05_finetuning\step0\waste.shp
Reference completeness confirmed: True
Fixed thresholds: (0.5, 0.9); minimum object area: 0.25 m2
Object matching: 8-connected, >= 1.0% reference coverage, greedy one-to-one by IoU.
The test area was used for error discovery; this is a paired spatial diagnostic evaluation, not an untouched external test.
No thresholds were selected using these results.

MAIN RESULTS (0.50)

original: predicted=1,529, TP=30, unmatched=1,499, FN=8
Object precision=0.0196; recall=0.7895; F1=0.0383
Mapped area=7472.74 m2; reference area=942.22 m2

refined: predicted=705, TP=29, unmatched=676, FN=9
Object precision=0.0411; recall=0.7632; F1=0.0781
Mapped area=2439.67 m2; reference area=942.22 m2

REFINED MINUS ORIGINAL (0.50)
Predicted objects: -824
Matched reference piles: -1
Unmatched candidates: -823
Missed reference piles: +1
Mapped area: -5033.07 m2
Object precision change: +0.0215
Object recall change: -0.0263
Object F1 change: +0.0398

The outputs use the same corrected reference and common valid raster pixels. If the source models used different inference mosaicking/halos, the comparison measures map-product changes rather than isolating fine-tuning effects.

Results directory: C:\Users\Peter\Documents\GitHub\mapping-waste-urban-africa\results\spatial_test_comparison\run_20260907_134911