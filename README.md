# Multiresolution Deep Implicit Functions for 3D Shape Representation

Tensorflow implementation of the paper ["Multiresolution Deep Implicit Functions for 3D Shape Representation"](https://arxiv.org/abs/2109.05591), ICCV 2021.

## Introduction

This project proposes a new deep multi-resolution deep implicit network for 3D shape representation. Traditional 3D shape representations such as point clouds, meshes, voxels usually suffer from undefined surface, fixed topology or inefficient storage. On the other hand, implicit function continuously associates each location in 3D space to geometric properties such as signed distance function or occupancy, and allows better incorporation with data-driven methods. However, while recent advances in deep implicit function provides improved flexibility and quality, there still exists many areas for improvement. Our project investigates the incorporation of global and local deep implicit functions, which can produce more efficient representation for detailed geometry as well as completing detailed partial shapes. Specifically, we propose a multi-resolution network architecture consisting of encoders, decoders and cross-resolution relationships to exploit the nature of multiple detail levels in 3D shapes. Apart from shape auto-encoding and completion, the outcome of this project can also potentially be used in other applications such as shape reconstruction, differentiable rendering and neural rendering.

This is not an officially supported Google product.

## Citation

If you find this code useful in your research, please cite:

```bibtex
@inproceedings{Chen2021Multiresolution,
  title = {{Multiresolution Deep Implicit Functions for 3D Shape Representation}},
  author = {Chen, Zhang and Zhang, Yinda and Genova, Kyle and Funkhouse, Thomas and Fanello, Sean and Bouaziz, Sofien and Haene, Christian and Du, Ruofei and Keskin, Cem and Tang, Danhang},
  booktitle = {2021 IEEE/CVF International Conference on Computer Vision},
  year = {2021},
  publisher = {IEEE},
  series = {ICCV},
}
```
