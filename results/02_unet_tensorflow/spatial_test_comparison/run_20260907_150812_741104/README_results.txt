PAIRED SPATIAL WASTE-MAP COMPARISON
=================================================================
Area: data/aoi/testarea.shp; NOT the 222 waste-centred validation tiles.
Reference: C:\Users\Peter\Documents\GitHub\mapping-waste-urban-africa\05_finetuning\step0\waste.shp
Reference completeness confirmed: True
Fixed thresholds: (0.5, 0.9); minimum object area: 0.25 m2
Reference units: 38 source features / 42 polygon parts -> 39 contiguous units.
Touching/overlapping polygon parts merge; disconnected multipart pieces remain separate. No distance buffer is applied.
Object matching: 8-connected, >= 1.0% reference coverage, greedy one-to-one by IoU.
The test area was used for error discovery; this is a paired spatial diagnostic evaluation, not an untouched external test.
No thresholds were selected using these results.

MAIN RESULTS (0.50)

original: predicted=1,529, TP=30, FP=1,499, FN=9
Object precision=0.0196; recall=0.7692; F1=0.0383
Pixel TP=521,731; FP=4,623,464; FN=127,044; TN=278,749,521
Filtered pixel precision=0.1014; recall=0.8042; F1=0.1801; IoU=0.0990; accuracy=0.9833
Raw-threshold pixel precision=0.0975; recall=0.8074; F1=0.1741; IoU=0.0953; accuracy=0.9825
Mapped area=7472.74 m2; reference area=942.22 m2

refined: predicted=705, TP=29, FP=676, FN=10
Object precision=0.0411; recall=0.7436; F1=0.0780
Pixel TP=443,540; FP=1,236,242; FN=205,235; TN=282,136,743
Filtered pixel precision=0.2640; recall=0.6837; F1=0.3810; IoU=0.2353; accuracy=0.9949
Raw-threshold pixel precision=0.2487; recall=0.6881; F1=0.3653; IoU=0.2235; accuracy=0.9945
Mapped area=2439.67 m2; reference area=942.22 m2

REFINED MINUS ORIGINAL (0.50)
Predicted objects: -824
Matched reference piles: -1
Unmatched candidates: -823
Missed reference piles: +1
Mapped area: -5033.07 m2
Object precision change: +0.0215
Object recall change: -0.0256
Object F1 change: +0.0397
Pixel precision change: +0.1626
Pixel recall change: -0.1205
Pixel F1 change: +0.2009
Pixel IoU change: +0.1363
Pixel accuracy change: +0.0117

The outputs use the same corrected reference and common valid raster pixels. If the source models used different inference mosaicking/halos, the comparison measures map-product changes rather than isolating fine-tuning effects.

Results directory: C:\Users\Peter\Documents\GitHub\mapping-waste-urban-africa\results\spatial_test_comparison\run_20260907_150812_741104

Pixel accuracy is evaluated over the common valid test-area pixels. The raw-threshold results do not apply the minimum-area filter; the filtered results use the same final binary map as the object analysis.
The current reference is used exactly as supplied. If inspection identifies incorrect reference polygons, correct them in a separate reviewed version and rerun both models against that same version. Earlier numerical results are not directly comparable after reference-unit changes.