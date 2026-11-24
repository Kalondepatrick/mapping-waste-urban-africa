**TABLE II**
**GeoBIA Model Development, Evaluated at Segment Level**

| Feature set | Classifier | Accuracy | Precision | Recall | F1 | IoU |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Optimal feature set (RFE) | Random forest | 0.827 | 0.774 | 0.867 | 0.818 | 0.692 |
| Optimal feature set (RFE) | Linear SVM | 0.825 | 0.762 | 0.853 | 0.805 | 0.674 |
| Optimal feature set (RFE) | Naive Bayes | 0.575 | 0.495 | 0.733 | 0.591 | 0.420 |
| Optimal feature set (RFE) | Neural network | 0.723 | 0.641 | 0.453 | 0.531 | 0.362 |
| Optical bands only | Random forest | 0.713 | 0.512 | 0.547 | 0.529 | 0.360 |
| Optical bands only | Linear SVM | 0.675 | 0.452 | 0.507 | 0.478 | 0.314 |
| Optical bands only | Neural network | 0.615 | 0.333 | 0.200 | 0.250 | 0.143 |
| Optical bands only | Naive Bayes | 0.489 | 0.192 | 0.067 | 0.099 | 0.052 |

*Segment-level accuracy measures how well segments are labelled once they exist. It is not comparable with the pixel-level results in Tables I and III to XI, which measure how well the resulting map delineates waste on the ground, and it is reported only to document how the GeoBIA classifier was chosen. The pixel-level GeoBIA result is in Table I.*
