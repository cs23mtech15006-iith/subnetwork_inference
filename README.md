# End Semester Project, Bayesian Data Analysis (CS5350)

**Topic**

Bayesian Deep Learning via Subnetwork Inference

**Team Details**

| No. | Name                    | Roll Number    |
|-----|-------------------------|----------------|
| 1   | Srijani Das             | CS23MTECH15022 |
| 2   | Arghyadeb Bandyopadhyay | CS23MTECH15006 |

## Instructions

Please refer to the file `notebooks/evaluate.ipynb` for our implementation of Bayesian Deep Learning via Subnetwork Inference. To run and evaluate different test setups, update the following hyperparameters:

* `n_weights_subnet`: The size of the pruned subnetwork. Set a number smaller than the original network, yet large enough to capture the uncertainty modeling capability of the original model. We used a default value of 5000.
* `subnet_selection`:  The algorithm applied for selecting the subnetworks. Supported values: "snr", "magnitude", "min-wass", "random"
* `layer_weight`: Whether to add more weightage to the earlier layers or the later layers of the network. Allowed values: None, "forward", "backward". This hyperparameter is considered only when `subnet_selection` mode is set to "min-wass".
* `methods`: Assign a distinctly identifiable name. This will be used for plotting results. Accepts an array of strings.
* `scaling_factor`: When `subnet_selection` is set to "min-wass" and `layer_weight` is set to "forward" or "backward", an optional parameter `scaling_factor` can be passed when selecting the weights during the pruning stage. This instructs how much weightage a layer should receive relative to the previous or the immediate next layer. Accepts a value between 0 and 1. We recommend that a value close to 1 is used. Default value is `0.9`.

## Contents

This repository contains the code for the project Bayesian Deep Learning via Subnetwork Inference.

We took inspiration from the repository published by the authors of the paper _Bayesian Deep Learning via Subnetwork Inference_. The original github repository is https://github.com/HanKimHan/subnetwork_inference

The _src_ directory contains all the necessary helper functions for the implementation. And the model creation, training and evaluation is done in the file `notebooks/evaluate.ipynb`.

# References

1. Bayesian deep learning via subnetwork inference, Daxberger, Erik, et al. "Bayesian deep learning via subnetwork inference." International Conference on Machine Learning. PMLR, 2021.
