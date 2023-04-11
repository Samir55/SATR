# SATR: Zero-Shot Semantic Segmentation of 3D Shapes

[![Website Badge](https://raw.githubusercontent.com/referit3d/referit3d/eccv/images/project_website_badge.svg)](https://samir55.github.io/SATR/)
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=plastic)]()

Code & Benchmark will be released soon!    
Stay tuned! 

<p align="center">
  <img src="https://samir55.github.io/SATR/images/teaser.jpg" width=80% height=80% >
</p>

# Abstract
We explore the task of zero-shot semantic segmentation of 3D shapes by using large-scale off-the-shelf 2D image recognition models. Surprisingly, we find that modern zero-shot 2D object detectors are better suited for this task than contemporary text/image similarity predictors or even zero-shot 2D segmentation networks. Our key finding is that it is possible to extract accurate 3D segmentation maps from multi-view bounding box predictions by using the topological properties of the underlying surface. For this, we develop the Segmentation Assignment with Topological Reweighting (SATR) algorithm and evaluate it on two challenging benchmarks: FAUST and ShapeNetPart. On these datasets, SATR achieves state-of-the-art performance and outperforms prior work by at least 22% on average in terms of mIoU.

For additional detail, please see "[SATR: Zero-Shot Semantic Segmentation of 3D Shapes]()"  
by [Ahmed Abdelreheem](https://samir55.github.io/), [Ivan Skorokhodov](https://universome.github.io/),
[Maks Ovsjanikov](https://www.lix.polytechnique.fr/~maks/), and [Peter Wonka](https://peterwonka.net/)  
from KAUST and LIX, Ecole Polytechnique.

# Citation
```
@article{abdelreheem2023SATR,
        author = {Abdelreheem, Ahmed and Skorokhodov, Ivan and Ovsjanikov, Maks and Wonka, Peter}
        title = {SATR: Zero-Shot Semantic Segmentation of 3D Shapes},
        journal = Computing Research Repository (CoRR),
        volume = {abs/},
        year = {2023}
}
      
```

