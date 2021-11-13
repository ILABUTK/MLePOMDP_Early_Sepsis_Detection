"""
Main, POMDP prediction. eICU.
"""

# import
import time
import pickle
import logging
import numpy as np
import pandas as pd
import scipy.stats as st
from PBVI import PBVI_OS
from POMDP import POMDP_OS
from copy import deepcopy as dcopy

from sklearn.metrics import confusion_matrix


def discretize_data(stream_data, levels):
    """make data discrete"""
    for i in range(stream_data.shape[0]):
        for j in range(stream_data.shape[1]):
            for level in levels:
                if stream_data.iloc[i, j] <= level:
                    stream_data.iloc[i, j] = level
                    break
    return stream_data


def seperate_data(original_data, data_names, percentage):
    """
    train-validation-test
    """
    # dimension of data
    all_index = list(range(original_data.shape[0]))
    index_dic = {}
    # randomly choose index
    for i in range(len(percentage)):
        if i == (len(percentage) - 1):
            index_dic[i] = all_index
            break
        index_dic[i] = list(np.random.choice(
            a=all_index,
            size=int(
                len(all_index) * percentage[i] / sum(percentage[i:])
            ),
            replace=False,
            p=[1/len(all_index)] * len(all_index)
        ))
        for item in index_dic[i]:
            all_index.remove(item)
    # get data in dataframe
    sliced_data = {}
    for index_name in index_dic:
        new_data = pd.DataFrame()
        for index in index_dic[index_name]:
            new_data = new_data.append(
                original_data.iloc[index]
            )
        sliced_data[data_names[index_name]] = new_data
    return sliced_data


def estimate_observation_pr(observations, sepsis_data, nonsep_data, interval):
    """
    eatimate the observation probability
    interval: data interval in real time.
    """
    # construct an empty dataframe
    obser_pr_dict = {}
    for obser in observations:
        obser_pr_dict[obser] = pd.Series(
            data=[0, 0],
            index=["sepsis", "nonsep"]
        )
    obser_pr = pd.DataFrame(obser_pr_dict)
    # count
    for i in range(sepsis_data.shape[0]):
        for j in range(
            # int(sepsis_data.shape[1]-60/interval), sepsis_data.shape[1]
            sepsis_data.shape[1]
        ):
            obser_pr.loc["sepsis", sepsis_data.iloc[i, j]] += 1

    for i in range(nonsep_data.shape[0]):
        for j in range(nonsep_data.shape[1]):
            obser_pr.loc["nonsep", nonsep_data.iloc[i, j]] += 1
    # normalize
    for i in range(2):
        row_sum = np.sum(obser_pr.iloc[i])
        for j in range(obser_pr.shape[1]):
            obser_pr.iloc[i, j] = obser_pr.iloc[i, j] / row_sum
    # obser_pr.to_csv("observ_pr_data/observ_matrix.csv")
    return obser_pr


def bm_output(
    model_name, bootstrap, bm_sensitivity, bm_specificity, bm_precision,
    bm_f_1, bm_ave_time, first_stage, horizon
):
    # ------------------ Benchmark Metrics--------------------
    # mean
    bm_sens_mean = np.mean(list(bm_sensitivity.values()))
    bm_spec_mean = np.mean(list(bm_specificity.values()))
    bm_prec_mean = np.mean(list(bm_precision.values()))
    bm_f1_mean = np.mean(list(bm_f_1.values()))
    bm_time_mean = np.mean(list(bm_ave_time.values()))
    # confidence interval
    bm_sens_ci = st.t.interval(
        0.95, bootstrap - 1, loc=bm_sens_mean,
        scale=st.sem(list(bm_sensitivity.values()))
    )
    bm_spec_ci = st.t.interval(
        0.95, bootstrap - 1, loc=bm_spec_mean,
        scale=st.sem(list(bm_specificity.values()))
    )
    bm_prec_ci = st.t.interval(
        0.95, bootstrap - 1, loc=bm_prec_mean,
        scale=st.sem(list(bm_precision.values()))
    )
    bm_f1_ci = st.t.interval(
        0.95, bootstrap - 1, loc=bm_f1_mean,
        scale=st.sem(list(bm_f_1.values()))
    )
    bm_time_ci = st.t.interval(
        0.95, bootstrap - 1, loc=bm_time_mean,
        scale=st.sem(list(bm_ave_time.values()))
    )
    # print result
    file = open("results/{}_hour_{}/BM-{}.txt".format(
        horizon, first_stage, model_name
    ), mode="w+")
    file.write("=====================================\n")
    file.write("BENCHMARK\n")
    file.write("=====================================\n")
    file.write("sensitivity = {}\n".format(bm_sens_mean))
    file.write("    CI = {}\n".format(
        (bm_sens_ci[1]-bm_sens_ci[0])/2
    ))
    file.write("specificity = {}\n".format(bm_spec_mean))
    file.write("    CI = {}\n".format(
        (bm_spec_ci[1] - bm_spec_ci[0])/2
    ))
    file.write("precision = {}\n".format(bm_prec_mean))
    file.write("    CI = {}\n".format(
        (bm_prec_ci[1]-bm_prec_ci[0])/2
    ))
    file.write("F1 score = {}\n".format(bm_f1_mean))
    file.write("    CI = {}\n".format(
        (bm_f1_ci[1]-bm_f1_ci[0])/2
    ))
    file.write("Prediction time = {}\n".format(bm_time_mean))
    file.write("    CI = {}\n".format(
        (bm_time_ci[1]-bm_time_ci[0])/2
    ))
    file.write("=====================================\n")
    file.write("sensitivity: {}\n".format(bm_sensitivity))
    file.write("specificity: {}\n".format(bm_specificity))
    file.write("precision: {}\n".format(bm_precision))
    file.write("F1: {}\n".format(bm_f_1))
    file.write("Time: {}\n".format(bm_ave_time))
    file.close()
    return


def POMDP_output(
    model_name, bootstrap, sensitivity, specificity, precision,
    f_1, ave_time, first_stage, horizon
):
    """POMDP output"""
    # mean
    sens_mean = np.mean(list(sensitivity.values()))
    spec_mean = np.mean(list(specificity.values()))
    prec_mean = np.mean(list(precision.values()))
    f1_mean = np.mean(list(f_1.values()))
    time_mean = np.mean(list(ave_time.values()))
    # confidence interval
    sens_ci = st.t.interval(
        0.95, bootstrap - 1, loc=sens_mean,
        scale=st.sem(list(sensitivity.values()))
    )
    spec_ci = st.t.interval(
        0.95, bootstrap - 1, loc=spec_mean,
        scale=st.sem(list(specificity.values()))
    )
    prec_ci = st.t.interval(
        0.95, bootstrap - 1, loc=prec_mean,
        scale=st.sem(list(precision.values()))
    )
    f1_ci = st.t.interval(
        0.95, bootstrap - 1, loc=f1_mean,
        scale=st.sem(list(f_1.values()))
    )
    time_ci = st.t.interval(
        0.95, bootstrap - 1, loc=time_mean,
        scale=st.sem(list(ave_time.values()))
    )

    # print result
    file = open("results/{}_hour_{}/{}.txt".format(
        horizon, first_stage, model_name
    ), mode="w+")
    # file = open("results/MLH_model/{}-{}-{}.txt".format(
    #     horizon, first_stage, model_name
    # ), mode="w+")
    file.write("=====================================\n")
    file.write("POMDP\n")
    file.write("=====================================\n")
    file.write("sensitivity = {}\n".format(sens_mean))
    file.write("    CI = {}\n".format(
        (sens_ci[1]-sens_ci[0])/2
    ))
    file.write("specificity = {}\n".format(spec_mean))
    file.write("    CI = {}\n".format(
        (spec_ci[1]-spec_ci[0])/2
    ))
    file.write("precision = {}\n".format(prec_mean))
    file.write("    CI = {}\n".format(
        (prec_ci[1]-prec_ci[0])/2
    ))
    file.write("F1 score = {}\n".format(f1_mean))
    file.write("    CI = {}\n".format(
        (f1_ci[1]-f1_ci[0])/2
    ))
    file.write("Prediction time = {}\n".format(time_mean))
    file.write("    CI = {}\n".format(
        (time_ci[1]-time_ci[0])/2
    ))
    file.write("=====================================\n")
    file.write("sensitivity: {}\n".format(sensitivity))
    file.write("specificity: {}\n".format(specificity))
    file.write("precision: {}\n".format(precision))
    file.write("F1: {}\n".format(f_1))
    file.write("Time: {}\n".format(ave_time))
    file.close()
    return


def test_POMDP(POMDP, policy, test_data, status):
    """simulation"""
    # Basic settings
    p = POMDP
    ind_iter = 0
    horizon = len(test_data)
    state = status
    action = p.actions[0]
    belief = p.init_belief
    reward = 0
    state_set = [state]
    action_set = []
    observation_set = ["null"]
    alpha_length = len(p.states)
    while True:
        # make an action
        ind_key = np.argmax([
            np.dot(
                policy[key][:alpha_length],
                belief
            )
            for key in policy.keys()
        ])
        action = policy[list(policy.keys())[ind_key]][alpha_length]
        action_set.append(action)
        # get a reward
        reward = reward + p.reward_func(state=state, action=action)
        # check stop condition
        ind_iter = ind_iter + 1
        if ind_iter >= horizon:
            break
        # state doesn't change
        state = state
        state_set.append(state)
        # make an observation
        observation = test_data.iloc[ind_iter]
        observation_set.append(observation)
        # update belief
        belief = [
            p.observ_func(observation, s_new, action) *
            np.sum([
                p.trans_func(s_new, s_old, action) *
                belief[p.states.index(s_old)]
                for s_old in p.states
            ])
            for s_new in p.states
        ]
        normalize_const = 1 / sum(belief)
        belief = np.multiply(belief, normalize_const)
    return action_set


def temporal_testing(
    horizon, model, observ_interval, first_stage,
    bm_threshold, ratio, bootstrap, epsilon, solve
):
    """
    first stage random forest, cross validation,
    not selecting a best model,
    without separate testing
    """
    model_name = "horizon-%s-ratio-%0.2f" % (horizon, ratio)
    # ====================== load data ========================
    observ_horizon = (horizon - 1) * 60
    interval = 5 if first_stage == "RF" else 12
    ML_data = pd.read_csv(
        'data/{}-{}.csv'.format(first_stage, horizon),
        index_col=False
    )
    sepsis_stream = ML_data.loc[ML_data['label'] == 1]
    sepsis_stream = sepsis_stream.reset_index(drop=True)
    sepsis_stream = sepsis_stream.drop(
        ['patientunitstayid', 'label'], axis=1
    )
    nonsep_stream = ML_data.loc[ML_data['label'] == 0]
    nonsep_stream = nonsep_stream.reset_index(drop=True)
    nonsep_stream = nonsep_stream.drop(
        ['patientunitstayid', 'label'], axis=1
    )
    # ===================== discretize data =========================
    sepsis_discr = discretize_data(
        stream_data=dcopy(sepsis_stream), levels=dcopy(model.observations)
    )
    nonsep_discr = discretize_data(
        stream_data=dcopy(nonsep_stream), levels=dcopy(model.observations)
    )
    # =========================== Bootstrapping ===============================
    # metrics
    sensitivity, specificity, precision, f_1, ave_time = {}, {}, {}, {}, {}
    bm_sensitivity, bm_specificity, bm_precision = {}, {}, {}
    bm_f_1, bm_ave_time = {}, {}

    # update trans_function according to observation_interval
    def trans_func(new_state, old_state, action):
        """transition function"""
        p = 0.99967 ** (observ_interval * interval)
        if old_state == "sepsis":
            if new_state == "sepsis":
                return 1.0
            if new_state == "nonsep":
                return 0.0
        if old_state == "nonsep":
            if new_state == "sepsis":
                return 1 - p
            if new_state == "nonsep":
                return p
        return 0
    model.trans_func = trans_func
    # start bootstrap
    for boot in range(bootstrap):
        logging.info("Bootstrap: {}\n".format(boot))
        # -------------- sample data ---------------
        # index
        sepsis_tr_ind = np.random.choice(
            range(sepsis_discr.shape[0]), 500, False
        )
        nonsep_tr_ind = np.random.choice(
            range(nonsep_discr.shape[0]), 500, False
        )
        # data
        sepsis_data, nonsep_data = {}, {}
        # train data
        sepsis_data['train'] = sepsis_discr.iloc[sepsis_tr_ind, :]
        nonsep_data['train'] = nonsep_discr.iloc[nonsep_tr_ind, :]
        # test data
        sepsis_data['test'] = sepsis_discr[
            ~sepsis_discr.index.isin(sepsis_tr_ind)
        ]
        nonsep_data['test'] = nonsep_discr.iloc[
            ~nonsep_discr.index.isin(nonsep_tr_ind)
        ]
        # -------------- estimate observation probability -----------------
        model.name = "{}_{}_{}".format(first_stage, horizon, boot)
        obs_mat = estimate_observation_pr(
            observations=dcopy(model.observations),
            sepsis_data=dcopy(sepsis_data['train']),
            nonsep_data=dcopy(nonsep_data['train']),
            interval=1
        )

        # update observ matrix
        def observ_func(observation, state, action):
            """observation function"""
            obser_matrix = obs_mat
            return obser_matrix.loc[
                "{}".format(state), observation
            ]
        model.observ_func = observ_func
        logging.info("Problem Loaded!\n")
        # ---------------------- solving --------------------------
        solve_time = time.time()
        if not solve:
            alpha_vectors = pickle.load(open(
                'solutions/{}-{}-boot_{}.pickle'.format(
                    first_stage, horizon, boot
                ), 'rb'
            ))
        else:
            alpha_vectors = PBVI_OS(
                POMDP_OS=model, epsilon=epsilon,
                iterations=10, fig_dir='figures/solution'
            )
            pickle.dump(alpha_vectors, open(
                'solutions/{}-{}-boot_{}.pickle'.format(
                    first_stage, horizon, boot
                ), 'wb'
            ))
            logging.info("Solving Time = {}\n".format(
                time.time() - solve_time
            ))
        # -------------------- testing -------------------------
        logging.info("Testing...")
        prediciton_time, sepsis_cohort, nonsep_cohort = [], [], []
        bm_prediciton_time = []
        bm_sepsis_cohort, bm_nonsep_cohort = [], []
        for test_name in ["sepsis", "nonsep"]:
            if test_name == "sepsis":
                test_data = sepsis_data['test']
                iter_list = range(int(ratio * test_data.shape[0]))
            elif test_name == "nonsep":
                test_data = nonsep_data['test']
                iter_list = range(test_data.shape[0])
            # for each patient
            for i in iter_list:
                # ------------ benchmark test -----------------
                bm_result = []
                for t in range(len(test_data.iloc[i, ])):
                    if test_data.iloc[i, t] > bm_threshold:
                        bm_result.append(1)
                    else:
                        bm_result.append(0)
                try:
                    bm_prediciton_time.append(np.sum([
                        -1 * (observ_horizon + 60),
                        observ_interval * bm_result.index(1)
                    ]))
                    if test_name == "sepsis":
                        bm_sepsis_cohort.append(1)
                    elif test_name == "nonsep":
                        bm_nonsep_cohort.append(1)
                except ValueError:
                    if test_name == "sepsis":
                        bm_sepsis_cohort.append(0)
                    elif test_name == "nonsep":
                        bm_nonsep_cohort.append(0)
                # --------------- POMDP test ----------------
                result = test_POMDP(
                    POMDP=model, policy=alpha_vectors,
                    test_data=test_data.iloc[i], status=test_name
                )
                try:
                    prediciton_time.append(np.sum([
                        -1 * (observ_horizon + 60),
                        observ_interval * result.index("sepsis")
                    ]))
                    if test_name == "sepsis":
                        sepsis_cohort.append(1)
                    elif test_name == "nonsep":
                        nonsep_cohort.append(1)
                except ValueError:
                    if test_name == "sepsis":
                        sepsis_cohort.append(0)
                    elif test_name == "nonsep":
                        nonsep_cohort.append(0)
        # ----------------- benchmark statistics ----------------
        tn, fp, fn, tp = confusion_matrix(
            y_true=[0] * len(bm_nonsep_cohort) + [1] * len(bm_sepsis_cohort),
            y_pred=bm_nonsep_cohort + bm_sepsis_cohort
        ).ravel()
        bm_sensitivity[boot] = tp / (tp + fn)
        bm_specificity[boot] = 'Inf' if tn + fp == 0 else tn / (tn + fp)
        bm_precision[boot] = 'Inf' if tp + fp == 0 else tp / (tp + fp)
        bm_f_1[boot] = 'Inf' if 2 * tp + fp + fn == 0 else 2*tp / (2*tp+fp+fn)
        bm_ave_time[boot] = np.mean(bm_prediciton_time)
        # ----------------- POMDP statistics -------------------
        tn, fp, fn, tp = confusion_matrix(
            y_true=[0] * len(nonsep_cohort) + [1] * len(sepsis_cohort),
            y_pred=nonsep_cohort + sepsis_cohort
        ).ravel()
        sensitivity[boot] = tp / (tp + fn)
        specificity[boot] = 'Inf' if tn + fp == 0 else tn / (tn + fp)
        precision[boot] = 'Inf' if tp + fp == 0 else tp / (tp + fp)
        f_1[boot] = 'Inf' if 2 * tp + fp + fn == 0 else 2 * tp / (2*tp+fp+fn)
        ave_time[boot] = np.mean(prediciton_time)
    # ------------------ Output --------------------
    bm_output(
        model_name, bootstrap, bm_sensitivity, bm_specificity, bm_precision,
        bm_f_1, bm_ave_time, first_stage, horizon
    )
    POMDP_output(
        model_name, bootstrap, sensitivity, specificity, precision,
        f_1, ave_time, first_stage, horizon
    )
    # --------------- Done ---------------
    logging.info("Done!\n")
    return {
        'sens': list(sensitivity.values()),
        'spec': list(specificity.values()),
        'prec': list(precision.values()),
        'f_1': list(f_1.values()),
        'time': list(ave_time.values())
    }


def sepsis_prediction(horizon, first_stage):
    """
    predicting sepsis using POMDP
    """
    # logging
    logging.basicConfig(
        filename='logs/POMDP.log', filemode='w+',
        format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO
    )
    # random seed
    np.random.seed(1)
    # ================== POMDP definition ===================
    states = ["sepsis", "nonsep"]
    actions = ["sepsis", "undecided", "nonsep"]
    stops = ["sepsis"]
    observations = [
        0.2, 0.3, 0.4, 0.45, 0.475, 0.495, 0.5,
        0.505, 0.525, 0.55, 0.6, 0.7, 1.0
    ]
    init_belief = [0.0, 1.0]
    disc_factor = 0.9999
    origin_state = "nonsep"

    # transition func
    def trans_func(new_state, old_state, action):
        """transition function"""
        p = 0.99976
        if old_state == "sepsis":
            if new_state == "sepsis":
                return 1.0
            if new_state == "nonsep":
                return 0.0
        if old_state == "nonsep":
            if new_state == "sepsis":
                return 1 - p
            if new_state == "nonsep":
                return p
        return 0

    # observ func
    def observ_func(observation, state, action):
        """observation function"""
        obser_matrix = pd.read_csv(
            "observ_pr_data/observ_matrix.csv",
            index_col=0
        )
        return obser_matrix.loc[
            "{}".format(state),
            "{}".format(observation)
        ]

    # reward func
    def reward_func(state, action):
        """define reward function here"""
        if state == "sepsis":
            if action == "sepsis":
                return 584350
            elif action == "nonsep":
                return -183.49
            elif action == "undecided":
                return 130
        if state == "nonsep":
            if action == "sepsis":
                return -1000.0
            elif action == "nonsep":
                return 147.31
            elif action == "undecided":
                return 130

    POMDP_metrics = {}
    solve = True
    # ratio
    for a in [1 / 1, 1 / 3]:
        # observation interval
        for b in [1]:
            POMDP_metrics[b] = temporal_testing(
                horizon=horizon,
                model=POMDP_OS(
                    name="{}_{}".format(first_stage, horizon),
                    states=dcopy(states),
                    actions=dcopy(actions), stops=dcopy(stops),
                    observations=dcopy(observations),
                    trans_func=dcopy(trans_func),
                    observ_func=dcopy(observ_func),
                    reward_func=dcopy(reward_func),
                    init_belief=dcopy(init_belief),
                    disc_factor=dcopy(disc_factor),
                    origin_state=dcopy(origin_state)
                ),
                observ_interval=b, first_stage=first_stage,
                bm_threshold=0.5, ratio=a, bootstrap=10, epsilon=0.01,
                solve=solve
            )
            solve = False
    return


def main():
    """
    Main
    """
    # ------------- Main Results ---------------
    np.random.seed(1)
    # horizon, 6.
    horizon = 6
    # first stage model, RF or NN.
    first_stage = "NN"
    # main result
    sepsis_prediction(horizon, first_stage)
    return


if __name__ == "__main__":
    main()
