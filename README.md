# SATR: Zero-Shot Semantic Segmentation of 3D Shapes
[![Website Badge](https://raw.githubusercontent.com/referit3d/referit3d/eccv/images/project_website_badge.svg)](https://samir55.github.io/SATR/)
[![arXiv](https://img.shields.io/badge/arXiv-2304.04909-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2304.04909)

<!-- <p align="center">
  <img src="https://samir55.github.io/SATR/images/teaser.jpg" width=80% height=80% >
</p> -->

<object data="https://samir55.github.io/SATR/images/teaser2-cropped.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://samir55.github.io/SATR/images/teaser2-cropped.pdf">
        <!-- <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://samir55.github.io/SATR/images/teaser2-cropped">Download PDF</a>.</p> -->
    </embed>
</object>

## Introduction
We explore the task of zero-shot semantic segmentation of 3D shapes by using large-scale off-the-shelf 2D im- age recognition models. Surprisingly, we find that modern zero-shot 2D object detectors are better suited for this task than contemporary text/image similarity predictors or even zero-shot 2D segmentation networks. Our key finding is that it is possible to extract accurate 3D segmentation maps from multi-view bounding box predictions by using the topological properties of the underlying surface. For this, we develop the Segmentation Assignment with Topological Reweighting (SATR) algorithm and evaluate it on ShapeNetPart and our proposed FAUST benchmarks. SATR achieves state-of-the-art performance and outperforms a baseline algorithm by 1.3% and 4% average mIoU on the FAUST coarse and fine-grained benchmarks, respectively, and by 5.2% average mIoU on the ShapeNetPart benchmark. Our source code and data will be publicly released. Project webpage: https://samir55.github.io/SATR/.

For additional detail, please see "[SATR: Zero-Shot Semantic Segmentation of 3D Shapes](https://arxiv.org/abs/2304.04909)"  
by [Ahmed Abdelreheem](https://samir55.github.io/), [Ivan Skorokhodov](https://universome.github.io/),
[Maks Ovsjanikov](https://www.lix.polytechnique.fr/~maks/), and [Peter Wonka](https://peterwonka.net/)  
from KAUST and LIX, Ecole Polytechnique.

## Installation

### A. Create Environment
```shell
conda create -n meshseg python=3.9
conda activate meshseg
conda install cudatoolkit=11.1 -c conda-forge
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

### B. Build/Install Kaolin
At first, you may also try installing pre-built wheels found [here](https://kaolin.readthedocs.io/en/latest/notes/installation.html). For example, you can run this command for CUDA 11.3 and PyTorch 1.10.0 for Kaolin 0.13.0

```shell
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.10.0_cu111.html
```

But if this didn't work out for you, please do the steps below:

- Clone Kaolin in some directory outside the repo.
  ```
  git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
  cd kaolin
  ```
- then,

  ```
  git checkout v0.13.0 # optional
  pip install -r tools/build_requirements.txt -r tools/viz_requirements.txt -r tools/requirements.txt
  python setup.py develop
  ```

### C. Install the package
```shell
git clone https://github.com/Samir55/SATR
cd SATR/
pip install -e .
```

### D. Install GLIP
```shell
cd GLIP/
python setup.py build develop --user
```
**NOTE**: Download the pretrained GLIP model from [here](https://drive.google.com/drive/folders/1qcRTh3omFbiF76XnGOfOwkxm38nLgm3f?usp=sharing), and put it in ```GLIP/MODEL/```


## Datasets

- For FAUST, please download the FAUST benchmark dataset from [this link](https://drive.google.com/drive/folders/1T5reNd6GqRfQRyhw8lmhQwCVWLcCOZVN?usp=sharing) and put them in ``data\FAUST``. 
- For the [ShapeNetPart dataset](https://cs.stanford.edu/~ericyi/project_page/part_annotation/), please download the labelled meshes from [this link](http://people.cs.umass.edu/~kalo/papers/shapepfcn/index.html). We use the official test split provided [here](http://people.cs.umass.edu/~kalo/papers/shapepfcn/index.html).


## Code Running

### Demo
Please create a suitable config file to run on an input mesh (see the ``configs`` folder for examples). For instance, to run on a penguin example, use the following command from the repository root directory:

```shell
CUDA_VISIBLE_DEVICES=0 python scripts/single_dataset_example.py -cfg configs/demo/penguin.yaml -mesh_name penguin.obj -output_dir outputs/demo/penguin
``` 

### FAUST/ShapeNetPart
To run on a single example (for instance, tr_scan_000) of the FAUST dataset on the coarse segmentation, please use the following command
```shell
CUDA_VISIBLE_DEVICES=0 python scripts/single_dataset_example.py -cfg configs/faust/coarse.yaml -mesh_name tr_scan_000.obj -output_dir path_to_output_dir
``` 
and for the fine-grained segmentation
```shell
CUDA_VISIBLE_DEVICES=0 python scripts/single_dataset_example.py -cfg configs/faust/fine_grained.yaml -mesh_name tr_scan_000.obj -output_dir path_to_output_dir
``` 

For the ShapeNetPart models, please run ``scripts/single_dataset_example.py`` with the suitable config file for each category found in ``configs/shapenetpart``

## Evaluation

Given an output dir (for example ``coarse_output_dir``) containing the coarse or fine-grained predictions for the 100 scans, run the following:
```shell
python scripts/evaluate_faust.py -output_dir outputs/coarse_output_dir
```
or for the fine_grained:

```shell
python scripts/evaluate_faust.py --fine_grained -output_dir outputs/fine_grained_output_dir
```

## Credits
This codebase used some of [3DHighlighter](https://github.com/threedle/3DHighlighter), [GLIP HuggingFace demo](https://huggingface.co/spaces/haotiz/glip-zeroshot-demo), and [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) repositories. Thanks to the authors for their awesome work!  

## Citation
If you find this work useful in your research, please consider citing:

```
@article{abdelreheem2023SATR,
        author = {Abdelreheem, Ahmed and Skorokhodov, Ivan and Ovsjanikov, Maks and Wonka, Peter}
        title = {SATR: Zero-Shot Semantic Segmentation of 3D Shapes},
        journal = Computing Research Repository (CoRR),
        volume = {abs/2304.04909},
        year = {2023}
}
      
```

