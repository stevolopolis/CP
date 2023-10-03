# Context Required To Understand the "Coordinate Permutation" Perspective

We are proposing to understand Instant-NGP and its family of hybrid INRs with the perspective of "coordinate permutations". We hypothesize the embedding part of these hybrid INRs is simply learning a permutation (or transformation) of the original coordinate system. This is contrary to the common understanding of the embeddings learning "features" of the data. 

To better understand the intuition behind our proposed perspective, we compiled a list of relevant resources to get you familiar with all the necessary prerequisites. The topics include:
- Implicit Neural Representations (INR)
    - Purely implicit INR
    - Hybrid INR
- How INRs are learning signals effectively
- Applications of INRs

We also highly recommend the repo: [Awesome-NeRFS](https://github.com/awesome-NeRF/awesome-NeRF), which contains a much more comprehensive list of relevant works (though more specific to NeRFS).

Content:
1. [Reading List](#reading-list)
2. [Extremely Relevant Code Bases](#extremely-relevant-code-bases)


## Reading list
**Suggested structure of our project (by Prof. David Lindell)**
- [Dissecting GANs](https://arxiv.org/pdf/1811.10597)

**Extremely relevant papers**
- [DINER](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_DINER_Disorder-Invariant_Implicit_Neural_Representation_CVPR_2023_paper.pdf)
- [NGLOD](http://openaccess.thecvf.com/content/CVPR2021/papers/Takikawa_Neural_Geometric_Level_of_Detail_Real-Time_Rendering_With_Implicit_3D_CVPR_2021_paper.pdf)
- [Instant-NGP](https://nvlabs.github.io/instant-ngp/)
- [Neuralangelo](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Neuralangelo_High-Fidelity_Neural_Surface_Reconstruction_CVPR_2023_paper.pdf)

**Purely implicit INR**
- [NeRF](https://www.matthewtancik.com/nerf)
- [SIREN](https://proceedings.neurips.cc/paper/2020/file/53c04118df112c13a8c34b38343b9c10-Paper.pdf)
- [DeepSDF](https://arxiv.org/abs/1901.05103)
- [NeRV](https://proceedings.neurips.cc/paper_files/paper/2021/file/b44182379bf9fae976e6ae5996e13cd8-Paper.pdf)
- [pi-GAN](https://marcoamonteiro.github.io/pi-GAN-website/)

**Hybrid INR**
- [Plenoxel](https://alexyu.net/plenoxels/)
- [Neural geometric level of details (NGLOD)](http://openaccess.thecvf.com/content/CVPR2021/papers/Takikawa_Neural_Geometric_Level_of_Detail_Real-Time_Rendering_With_Implicit_3D_CVPR_2021_paper.pdf)
- [Instant-NGP](https://nvlabs.github.io/instant-ngp/)
- [ACORN](https://arxiv.org/pdf/2105.02788)
- [LIIF](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.pdf)

**How INRs learn**
- In my opinion, NTK is the best way to understand how INRs learn since INRs match with the assumptions of NTK theory quite well
- [A Structured Dictionary Perspective on Implicit Neural Representations](https://openaccess.thecvf.com/content/CVPR2022/papers/Yuce_A_Structured_Dictionary_Perspective_on_Implicit_Neural_Representations_CVPR_2022_paper.pdf)

**Applications of INR**
- {Novel view synthesis | 3D shape representation | Editable 3D scenes | Super-resolution}
  - Please refer to [Awesome-NeRFS](https://github.com/awesome-NeRF/awesome-NeRF) for a more detailed list.
- Data compression
  - [Learned-initialization for Optimizing Coordinate-based Neural Representations](https://openaccess.thecvf.com/content/CVPR2021/papers/Tancik_Learned_Initializations_for_Optimizing_Coordinate-Based_Neural_Representations_CVPR_2021_paper.pdf)
  - [Functa](https://arxiv.org/pdf/2201.12204https://arxiv.org/pdf/2201.12204)
  - [Modality-agnostic Variational Compression of INR](https://arxiv.org/pdf/2301.09479)
  - [Meta-Learning Sparse Compression Networks](https://arxiv.org/pdf/2205.08957)
- Networks weights as data
  - [Deep Learning on Implicit Neural Representations Of Shapes](https://arxiv.org/pdf/2302.05438)
  - [Signal Processing for Implicit Neural Representations](https://proceedings.neurips.cc/paper_files/paper/2022/file/575c450013d0e99e4b0ecf82bd1afaa4-Paper-Conference.pdf)
  - [Spatial Functa](https://arxiv.org/pdf/2302.03130)
- INR as a hypernetwork for other neural networks

## Extremely relevant code bases
- DINER -- [github](https://github.com/Ezio77/DINER)
