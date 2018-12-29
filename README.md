# Predictive Collective Variable Discovery

[Predictive Collective Variable Discovery with Deep Bayesian Models](https://arxiv.org/abs/1809.06913)

Markus Schöberl, [Nicholas Zabaras](https://www.zabaras.com), [Phaedon-Stelios Koutsourelakis](http://www.contmech.mw.tum.de)

Python/PyTorch implementation of the discovery of collective variables (CV) in atomistic systems.
The discovered CVs (e.g. for ALA-2 and ALA-15 peptides) are related to physicochemical properties
which are essential for understanding mechanisms especially in unexplored complex systems.
The CVs are *predictive* and are facilitated for the efficient calculation properties considering the
all-atom system. Predicted properties are augmented by confidence intervals accounting for epistemic uncertainty which
unavoidably occurs due to limited data.

## Dependencies
- Python 2
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
The dataset for ALA-2 and ALA-15 are contained in the subfolder `./data_peptide/ala-2/.` and `./data_peptide/ala-15/.`,
respectively. In both cases scripts of the molecular dynamic (MD) simulation leading to the reference trajectory can be found
in both cases in the subfolder `gromacs`. `Gromacs 4.6.7` has been used. For further information regarding MD simulations
it is refered to (http://www.gromacs.org).

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
<img src="http://latex.codecogs.com/svg.latex?\boldsymbol{x}\simp(\boldsymbol{x}|\boldsymbol{\theta}_{\text{MAP}})" border="0"/>
$\boldsymbol{x} \sim p(\boldsymbol{x})$.

### Uncertainty Quantification

Train the model and obtain a MAP estimate of the predicted trajectory and calculate the approximate posterior distribution
of the decoding network parameters.
```
python main.py --dataset ma_200 --epoch 8000 --batch_size 64 --z_dim 2 --seed 3251 --samples_pred 1000 --ard 1.0e-5 
--npostS 500
```
The runs are saved at `./results/<dataset>/<date>/.`.
The predicted trajectory is stored in a *.txt file named `samples_aevb_<dataset>_z_<dim>_<batch-size>_<max-epoch>.txt`.
Aforementioned command produces 500 samples of the decoding network parameters
$\boldsymbol{\theta} \sim p(\boldsymbol{\theta}|\boldsymbol{X})$.
For each $\boldsymbol{\theta}_i$ 1000 samples $\boldsymbol{x} \sim p(\boldsymbol{x}|\boldsymbol{\theta}_i)$ are predicted.

## Citation

If you find this repo useful for your research, please consider to cite:
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
