# Duplicate GeoBIA models

These four files sat in the `01_geobia/` code folder. They are **byte-identical**
(verified by MD5) to the canonical copies in `data/trained_models/geobia_r/`:

    geobia_modelsANN.rds  ==  data/trained_models/geobia_r/ANN.rds
    geobia_modelsNB.rds   ==  data/trained_models/geobia_r/NB.rds
    geobia_modelsRF.rds   ==  data/trained_models/geobia_r/RF.rds
    geobia_modelsSVM.rds  ==  data/trained_models/geobia_r/SVM.rds

Kept rather than deleted, but nothing reads them. Use the copies under `data/`.
