"""
Algorithm of Point-based Value Iteration for POMDP
"""

# import
import logging
import scipy.stats as st
import numpy as np
from copy import deepcopy as dcopy


# back up function for optimal stopping problem
def backup_os(p, old_alpha_vectors, beliefs):
    """
    compute new value using newly generated alpha vectors
    Bellman Equation = R() + Sum T()Omega()old_alpha;
    R(): reward_vector;
    Sum T()Omega()old_alpha: future_vector
    """

    alpha_length = len(p.states)
    stop_vectors = {}
    vectors = []
    # seperate old alpha vectors into two categories
    for key in old_alpha_vectors.keys():
        # record stopping vectors from the last iteration
        if old_alpha_vectors[key][alpha_length] in p.stops:
            stop_vectors[key] = old_alpha_vectors[key]
        # get rid of action, used for generating new alphas
        else:
            vectors.append(
                old_alpha_vectors[key][:alpha_length]
            )
    # if there's no alpha vectors corresponds to non-stop actions,
    # generate new vectors of [0.0]
    if vectors == []:
        if len(beliefs) == len(stop_vectors):
            return stop_vectors
        else:
            for key_b in beliefs.keys():
                if key_b not in stop_vectors.keys():
                    vectors.append(
                        [0.0] * alpha_length
                    )

    # construct reward_vectors, length=len(state)
    reward_vectors = {}
    for a in p.actions:
        reward_vectors[a] = [
            p.reward_func(s, a)
            for s in p.states
        ]

    # construct future vectors, lenth=len(state)
    # total number: action * observation * old_alpha
    future_vectors = {}
    for a in p.actions:
        for o in p.observations:
            for ind_alpha in range(len(vectors)):
                future_vectors[a, o, ind_alpha] = [
                    p.disc_factor * sum([
                        p.trans_func(s, old_s, a) *
                        p.observ_func(o, old_s, a) *
                        vectors[ind_alpha][p.states.index(old_s)]
                        for old_s in p.states
                    ])
                    for s in p.states
                ]
    # for an observation, a belief and an action,
    # find an future vector that maximize future value, over all old_alpha;
    belief_action_vectors = {}
    for key_b in beliefs.keys():
        for a in p.actions:
            best_vectors_for_o = []
            for o in p.observations:
                # compute future values for all old_alpha
                future_value_alpha = [
                    np.dot(
                        future_vectors[a, o, ind_alpha],
                        beliefs[key_b]
                    )
                    for ind_alpha in range(len(vectors))
                ]
                # find and record the best vector
                best_vectors_for_o.append(future_vectors[
                    a, o,
                    np.argmax(future_value_alpha)
                ])
            # sum best future vectors over observation
            # then sum reward with it.
            belief_action_vectors[key_b, a] = np.sum([
                np.dot(1 / len(p.observations), reward_vectors[a]),
                np.sum(best_vectors_for_o, axis=0)
            ], axis=0) if a not in p.stops else np.dot(
                1 / len(p.observations), reward_vectors[a]
            )

    # find best action for each belief among the newly generated alphas and
    # stop vectors from the last iteration.
    new_alpha_vectors = {}
    for key_b in beliefs.keys():
        action_value = {}
        for a in p.actions:
            action_value[a] = np.dot(
                belief_action_vectors[key_b, a],
                beliefs[key_b]
            )
        # value of last iteration stopping action for the same belief
        if key_b in list(stop_vectors.keys()):
            stop_val = np.dot(
                stop_vectors[key_b][:alpha_length],
                beliefs[key_b]
            )
        else:
            stop_val = float('-inf')
        # if last iteration is better, include in the new set
        if stop_val >= np.max(list(action_value.values())):
            new_alpha_vectors[key_b] = list(stop_vectors[key_b])
        # otherwise, choose form newly generated vectors.
        else:
            ind_best_action = np.argmax(list(action_value.values()))
            new_alpha_vectors[key_b] = list(belief_action_vectors[
                key_b,
                p.actions[ind_best_action]
            ])
            # add the best action to the end of vector.
            new_alpha_vectors[key_b].append(p.actions[ind_best_action])

    return new_alpha_vectors


def expand_os(p, beliefs):
    """belief expansion"""
    # print("expanding beliefs...")
    belief_new = []
    for key_b in beliefs.keys():
        belief_tem = []
        for a in p.actions:
            # sample a state s
            state = np.random.choice(
                a=p.states,
                size=1,
                replace=False,
                p=beliefs[key_b]
            )[0]
            # sample an observation
            o = np.random.choice(
                a=p.observations,
                size=1,
                replace=False,
                p=[
                    p.observ_func(observation, state, a)
                    for observation in p.observations
                ]
            )[0]
            # generate new belief for one action, save in belief_tem
            b_tem = [
                p.observ_func(o, s, a) *
                np.sum([
                    p.trans_func(s, p.states[ind_s_old], a) *
                    beliefs[key_b][ind_s_old]
                    for ind_s_old in range(0, len(p.states))
                ])
                for s in p.states
            ]
            # computing the normalization constant
            normal_constant = 1 / np.sum([
                beliefs[key_b][ind_s_old] *
                np.sum([
                    p.trans_func(s_new, p.states[ind_s_old], a) *
                    p.observ_func(o, s_new, a)
                    for s_new in p.states
                ])
                for ind_s_old in range(0, len(p.states))
            ])
            # normalizeing b_tem and add it in to belief_tem
            belief_tem.append(list(np.dot(normal_constant, b_tem)))

        # choosing the farest new belief
        dist = float("-inf")
        for b_cand in belief_tem:
            dist_cand = np.linalg.norm(
                np.sum([beliefs[key_b], -1 * b_cand], axis=0),
                ord=1
            )
            if dist <= dist_cand:
                dist = dist_cand
                b_new = b_cand
                continue
            else:
                continue
        belief_new.append(b_new)
    for belief in belief_new:
        if belief not in list(beliefs.values()):
            key = "b{}".format(len(beliefs))
            beliefs[key] = belief
        else:
            continue
    return beliefs


def expand_os_even(p, beliefs):
    """belief expansion"""
    # print("expanding beliefs...")
    belief_new = []
    pr = st.uniform.rvs(loc=0, scale=1, size=len(beliefs))
    for p in pr:
        belief_new.append([
            p,
            1 - p
        ])
    for belief in belief_new:
        if belief not in list(beliefs.values()):
            key = "b{}".format(len(beliefs))
            beliefs[key] = belief
        else:
            continue
    return beliefs


def compute_value_os(beliefs, alpha_vectors):
    """
    calculating value: belief dot alpha_vec;
    returns a list.
    """
    # compute value: belief dot alpha_vec.
    values = []
    for key_b in beliefs.keys():
        vector = alpha_vectors[key_b][
            :len(alpha_vectors[key_b]) - 1
        ]
        values.append(np.dot(
            vector,
            beliefs[key_b]
        ))

    return values


def checking_condition(val_old, val_new, epsilon):
    """
    check conditions:
    - whether to continue iteration;
    - or to expand belief.
    """
    for ind in range(len(val_old)):
        if np.absolute(val_old[ind] - val_new[ind]) <= epsilon:
            return True
        else:
            continue
    return False


def PBVI_OS(POMDP_OS, epsilon, iterations, fig_dir="None"):

    """PBVI algorithm"""

    logging.info("solving...\n")

    # Basic settings
    p = POMDP_OS
    beliefs = {'b0': p.init_belief}

    # alpha_vectors: length = len(states) + 1, incorporating actions.
    alpha_vectors = {}
    for key_b in beliefs.keys():
        alpha_vectors[key_b] = [0.0] * (len(p.states) + 1)

    # Iteration
    ind_iter = 0
    while ind_iter < iterations:

        logging.info("Iteration %d\n" % ind_iter)

        # iteration for one belief set.
        val_old = [float("-inf")] * len(beliefs)
        while True:

            # backup
            alpha_vectors = backup_os(
                p=dcopy(p),
                old_alpha_vectors=dcopy(alpha_vectors),
                beliefs=dcopy(beliefs)
            )
            # new value
            val_new = compute_value_os(
                beliefs=dcopy(beliefs),
                alpha_vectors=dcopy(alpha_vectors)
            )

            # if convergence, break and expand belief
            if checking_condition(val_old, val_new, epsilon):
                break
            else:
                val_old = val_new
                continue

        # belief expansion
        beliefs = expand_os_even(p, beliefs)

        # next iteration
        ind_iter = ind_iter + 1

    if fig_dir == "None":
        logging.info("Solved!\n")
        return alpha_vectors
    logging.info("Solved!\n")
    return alpha_vectors

# End of PBVI algorithm.
