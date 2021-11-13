"""
RF Prediction
"""

# import
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split


# RF training
def RF_training(num_tree, X, y):
    """
    train the RF model with the given dataset.
    return a trained model.
    - train_data: with 'label' indicate the classes.
    """
    RF_model = RandomForestClassifier(
        n_estimators=num_tree, oob_score=True
    )
    print("Traing...")
    RF_model = RF_model.fit(X, y)
    return RF_model


# temporal test
def temporal_test(horizon, train=True, num_tree=1000):
    """
    temporal testing
    """
    model_name = "RF_{}".format(num_tree)
    # =========================== Train ? =============================
    if train:
        # load patient ID
        separate_id = pickle.load(open('data/ids/1h_ID.pickle', 'rb'))
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
        # ========================= Train Model ============================
        # train model
        model = RF_training(num_tree, X_tr, y_tr)
        # save model
        pickle.dump(model, open(
            "models/RF_{}.pickle".format(num_tree), "wb"
        ))
    else:
        # load model
        model = pickle.load(open(
            "models/{}.pickle".format(model_name), "rb"
        ))
    # =========================== Test =============================
    print("Tesing...")
    # load patient id
    separate_id = pickle.load(open('data/ids/12h_ID.pickle', 'rb'))
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
        # remove lab variables
        patient_data = patient_data.drop([
            'paO2_FiO2', 'platelets_x_1000', 'total_bilirubin',
            'urinary_creatinine', 'creatinine', 'HCO3', 'pH', 'paCO2',
            'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
            'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium',
            'hct', 'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin',
            'GCS_Score', 'ventilator'
        ], axis=1)
        # predict
        pred_pr = model.predict_proba(X=patient_data)
        pred_pr = [
            pr[list(model.classes_).index(1)] for pr in pred_pr
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
    cohort_pr.to_csv("data/pr/RF-{}.csv".format(horizon), index=False)
    return


# Main
def main():
    """main"""
    np.random.seed(1)
    horizon = 6
    temporal_test(horizon, num_tree=1000, train=True)
    return


if __name__ == "__main__":
    main()
