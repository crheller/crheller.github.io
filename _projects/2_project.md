---
layout: page
title: PCA - Derivations, extensions, and limitations
description: In this project, I explore different methods for performing Principal Component Analysis with the goal of providing a deeper intuitive understanding of the method and its applicability to data science problems.
img: assets/img/12.jpg
importance: 1
category: fun
related_publications: true
---

## Background
Principal component analysis, typically referred to as PCA, is a popular dimensionality reduction technique utlized across many fields. Applications of PCA include data compression, data visualization, and latent variable discovery. Using open source tools, such as `sklearn`, today anyone can easily apply PCA to their data. However, this ease of access means that PCA is often applied without full consideration of whether or not it is the appropriate method for a given data analysis problem. 

Here, we dig a bit deeper by exploring some of the different possible methods for implementing PCA. Our goal is not to go through the mathematical derivations in full detail, but rather just to give a general intuition for PCA, how and when it can be applied to data, and ways to extend / customize the methods depending on your needs. 

#### Tools used:
```
numpy
scipy
sklearn
matplotlib
```
Full code available at: [https://github.com/crheller/PCAdemo.git](https://github.com/crheller/PCAdemo.git)

## Outline
1. [The basics](#basics)
2. [PCA as an eigendecomposition problem](#edecomp)
3. [PCA as a reconstruction optimization problem](#reconstruction)
4. [Extensions of PCA - Sparse PCA](#sparse)
5. [Limitations of PCA for latent variable disovery](#limitations)


## <a name="basics"></a>The basics
PCA is a linear dimensionality reduction method. The goal of dimensionalty reduction, in general, is to find a represenation of the original data which maintains the data's overall structure while reducing the number of dimensions needed to describe it. In the case of PCA, this is done by finding the ordered set of orthonormal **basis** vectors which capture the principal axes of variation in the original data. To reduce the dimensionality, we then discard the basis vectors which capture the least amount of variance and **transform** the orignal data by projecting it onto the remaining basis vectors. We can then **reconstruct** a "denoised", low-rank version of the original data. This basic idea is illustrated in the cartoon below.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/pca_basic.png" title="PCA basics" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Example visualization of PCA applied to a 2-D data matrix, X.
</div>


## <a name="edecomp"></a>PCA as an eigendecomposition problem


## <a name="reconstruction"></a>PCA as a reconstruction optimization problem


## <a name="sparse"></a>Extensions of PCA - Sparse PCA


## <a name="limitations"></a>Limitations of PCA for latent variable disovery