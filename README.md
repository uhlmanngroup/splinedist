
# SplineDist: Automated Cell Segmentation with Spline Curves 

This repository contains the implementation of SplineDist, a machine learning framework for automated cell segmentation with spline curves.

The [manuscript](https://www.biorxiv.org/content/10.1101/2020.10.27.357640v1) is currently under review. The code in its current state allows reproducing the paper experiments but is still in development.

We are currently working to package it in a cleaned and optimized form. In the meantime, we encourage interested end-users to contact us for more information and assistance.


## Overview
SplineDist has been designed for performing instance segmentation in bioimages. Our method has been built by extending the popular [StarDist](https://arxiv.org/abs/1806.03535) framework.

While StarDist models objects with star-convex polygonal representation, SplineDist models objects as parametric spline curves.
Our representation is more general and allows modelling non-star-convex objects as well, with the possibility of conducting further statistical shape analysis.

Our repository relies on the [StarDist](https://github.com/mpicbg-csbd/stardist) repository.  We encourage the user to explore StarDist repository for further details on the StarDist method.

## Requirements 

In order to use StarDist, you need to install:

1. Tensorflow (see [their documentation](https://www.tensorflow.org/install))

2. [StarDist](https://github.com/mpicbg-csbd/stardist), which be installed with `pip`: 
```bash
$ pip install stardist
```

## Walkthrough

Three walk-through notebooks have been included in this repository:
 1. [`notebooks/data.ipynb`](notebooks/data.ipynb): for data exploration
 2. [`notebooks/training.ipynb`](notebooks/training.ipynb): for training a model with SplineDist
 3. [`notebooks/prediction.ipynb`](notebooks/prediction.ipynb): for predicting outputs with SplineDist

## How to cite

You can cite our [manuscript](https://www.biorxiv.org/content/10.1101/2020.10.27.357640v1) as follows:

```bibtex
@article {Mandal2020.10.27.357640,
    author = {Mandal, Soham and Uhlmann, Virginie},
    title = {SplineDist: Automated Cell Segmentation with Spline Curves},
    elocation-id = {2020.10.27.357640},
    year = {2020},
    doi = {10.1101/2020.10.27.357640},
    publisher = {Cold Spring Harbor Laboratory},
    abstract = {We present SplineDist, an instance segmentation convolutional neural network for bioimages extending the popular StarDist method. While StarDist describes objects as star-convex polygons, SplineDist uses a more flexible and general representation by modelling objects as planar parametric spline curves. Based on a new loss formulation that exploits the properties of spline constructions, we can incorporate our new object model in StarDist{\textquoteright}s architecture with minimal changes. We demonstrate in synthetic and real images that SplineDist produces segmentation outlines of equal quality than StarDist with smaller network size and accurately captures non-star-convex objects that cannot be segmented with StarDist.Competing Interest StatementThe authors have declared no competing interest.},
    URL = {https://www.biorxiv.org/content/early/2020/10/28/2020.10.27.357640},
    eprint = {https://www.biorxiv.org/content/early/2020/10/28/2020.10.27.357640.full.pdf},
    journal = {bioRxiv}
}
```