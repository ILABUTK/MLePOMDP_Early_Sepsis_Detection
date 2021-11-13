[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# A Machine Learning-Enabled Partially Observable Markov Decision Process Framework for Early Sepsis Prediction

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The source code and data in this repository are a snapshot of the software and data
that were used in the research reported on in the manuscript under revision 
[A Machine Learning-Enabled Partially Observable Markov Decision Process Framework for Early Sepsis Prediction](https://www.researchgate.net/publication/341078371_A_Machine_Learning-Enabled_Partially_Observable_Markov_Decision_Process_Prediction_Framework) by Z. Liu et. al. This study utilized patient data collected by the [eICU Collaborative Research Batabase](https://eicu-crd.mit.edu/). To access the data, please refer to the [eICU Collaborative Research Batabase](https://eicu-crd.mit.edu/).

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

The codes contain four parts. First, we determine the ground truth of sepsis for available patients. Second, we preprocess the data to extract features. Then, we implement the first and the second stage of the framewrok.

The first stage of MLePOMDP is a machine learning (ML) model and the second stage is a partially observable Markov decision process (POMDP). This repository implements a random forest (RF) and a neural network (NN). Please refer to the manuscript for further details.

## Python Prerequisite

The following Python libraries are required to run the source codes:
1. `numpy`;
2. `scipy`;
3. `torch`;
4. `pandas`;
5. `pickle`;
6. `sklearn`;
7. `matplotlib`.

## Usage

### Ground Truth of Sepsis

After accessing the eICU database, put the following data files in the directory `scripts/0_sepsis_ground_truth/`:
1. `infusionDrug.csv`;
2. `lab.csv`;
3. `medication.csv`;
4. `microlab.csv`;
5. `nurseCharting.csv`;
6. `patient.csv`;
7. `respiratoryCharting.csv`;
8. `treatment.csv`;
9. `vitalPeriodic.csv`.

Dummy files are prepared at the directory to show the setup.

Then, run the `main.py` file to generate the ground truth of sepsis, as well as the patient data used in later stages. Note that to process the eICU data, a relatively large RAM is required.

### Preprocessing

Run through steps 0 to 3 in the `scripts/1_preprocessing/` folder. Before running each file, remember to change the data directory to the appropriate one, indicated in the files.

### First Stage

First, prepare the data as follows:
1. Copy all the files in `scripts/1_preprocessing/processed_data/3_12h_feature/` and paste to `scripts/2_first_stage/data/feature_data/patient_feature/`;
2. Copy the file `scripts/1_preprocessing/processed_data/3_train/train_data.csv` and paste to `scripts/2_first_stage/data/feature_data/`;
3. Copy all the files in `scripts/1_preprocessing/processed_data/ids/` and paste to `scripts/2_first_stage/data/ids/`.

Dummy files are prepared at each directory to show the setup.

Then, run the `RF.py` or the `NN.py` file to generate sepsis probability files for the second stage.

### Second Stage

First, copy all files in `scripts/2_first_stage/data/pr/` and paste to `scripts/3_second_stage/data/`. Dummy files are prepared at the directory to show the setup.

Then, run the `main.py` file to obtain the results. The final prediction results are stored in `scripts/3_second_stage/results/`. Solutions to the second-stage POMDP can be found at `scripts/3_second_stage/solutions/`. 

## Support

For support in using this software, submit an
[issue](https://github.com/ILABUTK/MLePOMDP_Early_Sepsis_Detection/issues/new).