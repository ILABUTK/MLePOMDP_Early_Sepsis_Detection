"""
RF Prediction
"""

# import
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# NN building
class MLP(nn.Module):
    """
    MLP
    """
    def __init__(self, hidden_layers, input_size, output_size, seed=1):
        """
        `hidden_layers`: list, the number of neurons for every layer;
        `input_size`: number of states;
        `output_size`: number of actions;
        `seed`: random seed.
        """
        super().__init__()
        # parameters
        self.seed = torch.manual_seed(seed)
        # NN, adding layers dynamically.
        self.layers = nn.Sequential()
        # ---------------------- input -----------------------
        self.layers.add_module(
            'Linear_inp', nn.Linear(input_size, hidden_layers[0])
        )
        self.layers.add_module('Act_inp', nn.ReLU())
        # ---------------------- hidden ----------------------
        for i in range(1, len(hidden_layers)):
            self.layers.add_module(
                'Linear_{}'.format(i),
                nn.Linear(hidden_layers[i - 1], hidden_layers[i])
            )
            self.layers.add_module('Act_{}'.format(i), nn.ReLU())
        # ----------------------- output ---------------------
        self.layers.add_module(
            'Linear_out', nn.Linear(hidden_layers[-1], output_size)
        )
        self.layers.add_module('Act_out', nn.Softmax(dim=1))

    def forward(self, input_seq):
        """
        `input_seq`: states, torch.FloatTensor.
        """
        # return
        return self.layers(input_seq)


# NN training
def NN_training(hidden_size, X, y, train=True, train_iter=3000):
    """
    train the RF model with the given dataset.
    return a trained model.
    - train_data: with 'label' indicate the classes.
    """
    # model
    NN_model = MLP(
        hidden_layers=hidden_size,
        input_size=X.shape[1],
        output_size=2
    )
    # Train
    if train:
        # optimizer
        NN_optimizer = optim.Adam(
            NN_model.parameters(), lr=0.0001
        )
        loss_hist = {}
        train_iter = 12000
        batch_size = 1000
        print("Training...")
        for iter in range(train_iter):
            if iter % 1000 == 0:
                print("Iteration: {}".format(iter))
            # zero grad
            NN_optimizer.zero_grad()
            # batch training
            sample_ind = np.random.choice(
                range(X.shape[0]), batch_size, False
            )
            # X batch
            X_batch = torch.FloatTensor(X.iloc[sample_ind, :].values)
            # y_batch
            y_true_batch = torch.LongTensor([
                int(y[i]) for i in sample_ind
            ])
            # predict
            y_pred_batch = NN_model(X_batch)
            # loss
            loss = F.cross_entropy(input=y_pred_batch, target=y_true_batch)
            loss_hist[iter] = loss.detach().item()
            # backpropogate
            loss.backward()
            NN_optimizer.step()
        # save model
        torch.save(NN_model.state_dict(), "models/NN_{}.pt".format(
            hidden_size
        ))
    # not train
    else:
        NN_model.load_state_dict(torch.load(
            "models/NN_{}.pt".format(hidden_size)
        ))
        NN_model.eval()
    return NN_model


# temporal test
def temporal_test(horizon, train=True, hidden_size=[100]):
    """
    temporal testing
    """
    # =========================== Train ? =============================
    # load patient ID
    separate_id = pickle.load(open('data/ids/train_ID.pickle', 'rb'))
    sepsis_id, nonsep_id = separate_id['sepsis'], separate_id['nonsep']
    # load data
    train_data = pd.read_csv(
        "data/feature_data/train_data.csv",
        index_col=False
    )
    # drop lab variables
    train_data = train_data.drop([
        'paO2_FiO2', 'platelets_x_1000', 'total_bilirubin',
        'urinary_creatinine', 'creatinine', 'HCO3', 'pH', 'paCO2',
        'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
        'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium',
        'hct', 'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin',
        'GCS_Score', 'ventilator'
    ], axis=1)
    # split data, ensure balanced train data
    sepsis_data = train_data.loc[
        train_data['patientunitstayid'].isin(sepsis_id)
    ]
    sepsis_data = sepsis_data.drop(['patientunitstayid', 'label'], axis=1)
    nonsep_data = train_data.loc[
        train_data['patientunitstayid'].isin(nonsep_id)
    ]
    nonsep_data = nonsep_data.drop(['patientunitstayid', 'label'], axis=1)
    # combined
    X_tr = sepsis_data.append(nonsep_data)
    y_tr = [1] * sepsis_data.shape[0] + [0] * nonsep_data.shape[0]
    # train model
    model = NN_training(hidden_size, X_tr, y_tr, train)
    # =========================== Test =============================
    print("Tesing...")
    # load patient id
    separate_id = pickle.load(open('data/ids/test_ID.pickle', 'rb'))
    sepsis_id, nonsep_id = separate_id['sepsis'], separate_id['nonsep']
    all_id = sepsis_id + nonsep_id
    # prediction results
    col_names = ['patientunitstayid', 'label']\
        + list(range(-60 * horizon, -55, 5))
    cohort_pr = pd.DataFrame(columns=col_names)
    # loop
    for p_id in all_id:
        # load data
        patient_data = pd.read_csv(
            "data/feature_data/patient_feature/{}.csv".format(p_id),
            index_col=False
        )
        # 6 h, otherwise delete
        patient_data = patient_data.loc[
            (patient_data['offset'] >= -60 * horizon) &
            (patient_data['offset'] <= -60)
        ]
        # remove p_id, label, offset
        patient_data = patient_data.drop(
            ['patientunitstayid', 'label', 'offset'], axis=1
        )
        # remove lab
        patient_data = patient_data.drop([
            'paO2_FiO2', 'platelets_x_1000', 'total_bilirubin',
            'urinary_creatinine', 'creatinine', 'HCO3', 'pH', 'paCO2',
            'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
            'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium',
            'hct', 'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin',
            'GCS_Score', 'ventilator'
        ], axis=1)
        # make tensor
        patient_data = torch.FloatTensor(patient_data.values)
        # predict
        pred_pr = model(patient_data).detach().numpy()
        pred_pr = [
            pr[1] for pr in pred_pr
        ]
        # true label
        true_label = 1 if p_id in sepsis_id else 0
        # add pr to dataframe
        patient_record = [int(p_id), true_label] + pred_pr
        # dict
        patient_dict = {
            col_names[i]: patient_record[i] for i in range(len(col_names))
        }
        # add pr to dataframe
        cohort_pr = cohort_pr.append(patient_dict, ignore_index=True)
    # write pr table
    cohort_pr.to_csv("data/pr/NN-{}.csv".format(horizon), index=False)
    return


# Main
def main():
    """main"""
    np.random.seed(1)
    horizon = 6
    temporal_test(
        horizon, hidden_size=[100, 500, 100], train=True
    )
    return


if __name__ == "__main__":
    main()
