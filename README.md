# Predictive Collective Variable Discovery with Deep Bayesian Models

Markus Schöberl, [Nicholas Zabaras](https://www.zabaras.com), [Phaedon-Stelios Koutsourelakis](http://www.contmech.mw.tum.de)

Python/PyTorch implementation of the discovery of collective variables (CV) in atomistic systems.
Extending spatio-temporal scale limitations of models for complex atomistic systems considered in biochemistry and materials science necessitates the development of enhanced sampling methods. The potential acceleration in exploring the configurational space by enhanced sampling methods depends on the choice of collective variables (CVs). This software implements the discovery of CVs as a Bayesian inference problem and considers CVs as hidden generators of the full-atomistic trajectory. The ability to generate samples of the fine-scale atomistic configurations using limited training data allows to compute estimates of observables as well as our probabilistic confidence on them. The formulation is based on emerging methodological advances in machine learning and variational inference. The discovered CVs are related to physicochemical properties which are essential for understanding mechanisms especially in unexplored complex systems. We provide a quantitative assessment of the CVs in terms of their predictive ability for alanine dipeptide (ALA-2) and ALA-15 peptide.

This code was utilized for the following publication: [Predictive Collective Variable Discovery with Deep Bayesian Models](https://arxiv.org/abs/1809.06913).


## Dependencies
- Python 2.7.9
- PyTorch 0.3.0
- Scipy
- Matplotlib
- Imageio
- Future

## Installation
- Install PyTorch and other dependencies
- Clone this repo:
```
git clone https://github.com/mjschoeberl/predictive-cvs.git
cd predictive-cvs
```

## Dataset
The datasets for ALA-2 and ALA-15 are located in the subfolder `./data_peptide/ala-2/.` and `./data_peptide/ala-15/.`,
respectively. In both cases, scripts for running the original molecular dynamic (MD) simulation leading to the reference trajectory are placed in the subfolder `./data_peptide/<peptide-name>/gromacs/.`. `Gromacs 4.6.7` has been used. For further information regarding MD simulations it is refered to (http://www.gromacs.org).

### ALA-2
For alanine dipeptide, we provide prepared datasets with N=[50,100,200,500] samples as used in the corresponding publication.

### ALA-15
For alanine 15 peptide, we provide prepared datasets with N=[300, 1500, 3000, 5000].

## Training

### MAP Estimate 

Train the model and obtain a MAP estimate of the predicted trajectory.
```
python main.py --dataset ma_200 --epoch 8000 --batch_size 64 --z_dim 2 --seed 3251 --samples_pred 1000 --ard 1.0e-5 
```
The runs are saved at `./results/<dataset>/<date>/.`.
The predicted trajectory is stored in a *.txt file named `samples_aevb_<dataset>_z_<dim>_<batch-size>_<max-epoch>.txt`.
Aforementioned command produces 1000 samples 
<img src="http://latex.codecogs.com/svg.latex?\boldsymbol{x}\sim%20p(\boldsymbol{x}|\boldsymbol{\theta}_{\text{MAP}})" border="2"/>.

### Uncertainty Quantification with the approximate posterior <img src="http://latex.codecogs.com/svg.latex?p(\boldsymbol{\theta}|\boldsymbol{X})" border="1"/>

Train the model and obtain a MAP estimate of the predicted trajectory and calculate the approximate posterior distribution
of the decoding network parameters.
```
python main.py --dataset ma_200 --epoch 8000 --batch_size 64 --z_dim 2 --seed 3251 --samples_pred 1000 --ard 1.0e-5 
--npostS 500
```
The runs are saved at `./results/<dataset>/<date>/.`.
The predicted trajectory is stored in a *.txt file named `samples_aevb_<dataset>_z_<dim>_<batch-size>_<max-epoch>.txt`.
Aforementioned command produces 500 samples of the decoding network parameters
<img src="http://latex.codecogs.com/svg.latex?\boldsymbol{\theta}_i\sim\%20p(\boldsymbol{\theta}|\boldsymbol{X})" border="2"/>.
For each <img src="http://latex.codecogs.com/svg.latex?\boldsymbol{\theta}_i" border="2"/>, 1000 samples <img src="http://latex.codecogs.com/svg.latex?\boldsymbol{x}\sim%20p(\boldsymbol{x}|\boldsymbol{\theta}_{\text{i}})" border="2"/> are predicted.

## Citation

Please consider to cite the following work:
```latex
@article{schoeberl2018,
title = "Predictive Collective Variable Discovery with Deep Bayesian Models",
journal = "Journal of Chemical Physics",
volume = "",
pages = "",
year = "",
issn = "",
doi = "",
url = "",
author = "Markus Sch{\"{o}}berl, and Nicholas Zabaras and Phaedon-Stelios Koutsourelakis"
}
```

## Acknowledgments

The code is inspired by the implementation of Kingma and Welling in [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
