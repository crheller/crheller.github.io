---
layout: page
title: PCA - Derivations, limitations, and extensions
description: In this project, I explore different methods for performing Principal Component Analysis. I highlight extensions of some approaches (e.g., Sparse PCA). Finally, I demonstrate some limitations of the method for latent variable discovery.
img: assets/img/12.jpg
importance: 1
category: fun
related_publications: true
---

## Background
Principal component analysis, typically just refered to as PCA, is a popular dimensionality reduction technique utlized across many fields. Applications of PCA include data compression, data visualization, and latent variable discovery. Using open source tools, such as `sklearn`, anyone can easily apply PCA to their data. However, this ease of access means that PCA is often applied without deep consideration of whether or not it is the appropriate method for a given data analysis problem. 

Here, we dig a bit deeper, exploring some of the different possible methods for implementing PCA. This exercise will provide us with a better fundamental understanding of the underlying math, avenues for customizing / extending PCA, and an intuition for when it is (or is not) the best choice for dimensionality reduction of a given dataset.

#### Tools used:
```
numpy
scipy
sklearn
```
Full code available at: [https://github.com/crheller/PCAdemo.git](https://github.com/crheller/PCAdemo.git)

## Outline
1. [PCA as an eigendecomposition problem](#edecomp)
2. [PCA as a reconstruction optimization problem](#reconstruction)
3. [Extensions of PCA - Sparse PCA](#sparse)
4. [Limitations of PCA for latent variable disovery](#limitations)


## <a name="edecomp"></a>PCA as an eigendecomposition problem


## <a name="reconstruction"></a>PCA as a reconstruction optimization problem


## <a name="sparse"></a>Extensions of PCA - Sparse PCA


## <a name="limitations"></a>Limitations of PCA for latent variable disovery