[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# A Machine Learning-Enabled Partially Observable Markov Decision Process Framework for Early Sepsis Prediction

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The source code and data in this repository are a snapshot of the software and data
that were used in the research reported on in the manuscript under revision 
[A Machine Learning-Enabled Partially Observable Markov Decision Process Framework for Early Sepsis Prediction](https://www.researchgate.net/publication/341078371_A_Machine_Learning-Enabled_Partially_Observable_Markov_Decision_Process_Prediction_Framework) by Z. Liu et. al. This repository contains preprocessed data from the [eICU Collaborative Research Batabase](https://eicu-crd.mit.edu/).

## Cite

To cite this repository, please cite the [manuscript](https://www.researchgate.net/publication/341078371_A_Machine_Learning-Enabled_Partially_Observable_Markov_Decision_Process_Prediction_Framework).

Below is the BibTex for citing the manuscript.

```
@article{Liu2020,
  title={A Machine Learning-Enabled Partially Observable Markov Decision Process Prediction Framework},
  author={Liu, Zeyu and Khojandi, Anahita and Li, Xueping and Davis, Robert L and Kamaleswaran, Rishikesan},
  journal={Preprint},
  volume={10},
  year={2020},
  doi={10.13140/RG.2.2.17143.37280/1},
  url={https://www.researchgate.net/publication/341078371_A_Machine_Learning-Enabled_Partially_Observable_Markov_Decision_Process_Prediction_Framework}
}
```

## Description

The goal of this repository is to predict sepsis as early as possible using the MLePOMDP framewrok.

The framework contains two stages. The first stage is a machine learning (ML) model and the second stage is a partially observable Markov decision process (POMDP). This repository implements a random forest (RF) and a neural network (NN). Please refer to the manuscript for further details.

## Python Prerequisite

The following Python libraries are required to run the source codes:
1. `numpy`
2. `scipy`
3. `torch`
4. `pandas`
5. `pickle`
6. `sklearn`

## Usage

The two stages are stored in separate folders in the `scripts_and_data` directory.

With the required Python libraries, simply run the `main.py` file each of the stages to obtain the results.

## Support

For support in using this software, submit an
[issue](https://github.com/ILABUTK/MLePOMDP_Early_Sepsis_Detection/issues/new).