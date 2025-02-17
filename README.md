# GDN

Code implementation for : [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series(AAAI'21)](https://arxiv.org/pdf/2106.06947.pdf)

We apply GDN for our project.(DryStrip datasets from PSK)

# Installation
### Requirements
* Python >= 3.6
* cuda == 10.2
* [Pytorch==1.5.1](https://pytorch.org/)
* [PyG: torch-geometric==1.5.0](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

## Run
```
    # using gpu
    bash main.sh 
```
You can change running parameters in the run.sh.

# Cite
```
@inproceedings{deng2021graph,
  title={Graph neural network-based anomaly detection in multivariate time series},
  author={Deng, Ailin and Hooi, Bryan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={5},
  pages={4027--4035},
  year={2021}
}
```
