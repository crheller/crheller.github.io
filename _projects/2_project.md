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
seaborn
```
Full code available at: [https://github.com/crheller/PCAdemo.git](https://github.com/crheller/PCAdemo.git)

## Outline
1. [The basics](#basics)
2. [PCA as an eigendecomposition problem](#edecomp)
3. [PCA as a reconstruction optimization problem](#reconstruction)
4. [Extensions of PCA - Sparse PCA](#sparse)
5. [Limitations of PCA for latent variable disovery](#limitations)
6. [Summary](#summary)


## <a name="basics"></a>The basics
PCA is a linear dimensionality reduction method. The goal of dimensionalty reduction, in general, is to find a represenation of the original data which maintains the data's overall structure while reducing the number of dimensions needed to describe it. In the case of PCA, this is done by finding the ordered set of orthonormal **basis** vectors which capture the principal axes of variation in the original data. To reduce the dimensionality, we then discard the basis vectors which capture the least amount of variance and **transform** the orignal data by projecting it onto the remaining basis vectors. We can then **reconstruct** a "denoised", low-rank version of the original data. This basic idea is illustrated in the cartoon below.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/pca/pca_basic.png" title="PCA basics" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Example visualization of PCA applied to a 2-D data matrix, X.
</div>


## <a name="edecomp"></a>PCA as an eigendecomposition problem
The most typical way PCA is solved is by performing an eigendecomposition of the raw data covariance matrix. This is the method used by most open source libraries, such as `sklearn.decomposition.PCA`. Here, we step through this approach using "low level" linear algebra operations available in `numpy` to reproduce the output of `sklearn`.

#### Data generation
To illustrate this approach, we will use randomly generated 2-D synthetic data as above. To simplify our approach slightly, we generate mean-centered data.

#### Step 1 - Compute the covariance matrix
```
cov = np.cov(X) # use numpy covariance function to compute covariance matrix of data
```
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/pca/pca_data.png" title="pca data" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/pca/cov_matrix.png" title="covariance matrix" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Left: Synthetic input data. Right: Covariance matrix of synthetic data. 
</div>

#### Step 2 - Perform eigendecomposition
```
eigenvalues, eigenvectors = np.linalg.norm(cov) 

# sort according to variance explained
sort_args = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, sort_args]
eigenvalues = eigenvalues[sort_args]
```

#### Step 3 - Compute fraction variance explained by each component
```
var_explained = eigenvalues / np.sum(eigenvalues)
```

#### Step 4 - Verify that we have reproduced `sklearn` results
```
# sklearn pca
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X.T)
```

The eigenvectors we found are identical to the components returned by `sklearn`.
```
# the dot product of our eigenvectors and `sklearn`'s components should return the identity matrix
dp = abs.(eigenvectors.T @ pca.components_)
print(dp)

array([[ 1.00000000e+00,  1.97596122e-16],
       [-2.21149847e-16,  1.00000000e+00]])
```

We have correctly measured the variance explained by each component.
```
print(f"Eigendecomposition, variance explained ratio: {var_explained}")
print(f"sklearn, variance explained ratio: {pca.explained_variance_ratio_}")

Eigendecomposition, variance explained ratio: [0.88977678 0.11022322]
sklearn, variance explained ratio: [0.88977678 0.11022322]
```

Using the first PC to reconstruct our data, we get identical results.
```
e_reconstructed = X.T @ eigenvectors[:, [0]] @ eigenvectors[:, [0]].T
sk_reconstructed = X.T @ pca.components_[[0], :].T @ pca.components_[[0], :]
print(f"sum of reconstruction differences: {np.sum(e_reconstructed - sk_reconstructed)}")

sum of reconstruction differences: -1.8512968935624485e-14
```

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/pca/eig_vs_sklearn_summary.png" title="loading similarity" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Summary results of eigendecomposition and sklearn approaches to PCA. Both yield identical results.
</div>


## <a name="reconstruction"></a>PCA as a reconstruction optimization problem


## <a name="sparse"></a>Extensions of PCA - Sparse PCA
Use a large population here to highlight the advantage of sparse PCA for interpretability. Also - useful as regularization? i.e. performs better on held out data?


## <a name="limitations"></a>Limitations of PCA for latent variable disovery
Fake height / weight data. Goal is to discover the latent relationship between height / weight. But, imagine we don't know male vs. female. This latent factor could affect the relationship between height / weight that PCA discovers. Thus, if the latent dimensions of variation are not orthogonal, PCA fails to accurately capture the underlying latent relationships.

Non linear latent relationships are not well captured by PCA.

## <a name="summary"></a>Summary