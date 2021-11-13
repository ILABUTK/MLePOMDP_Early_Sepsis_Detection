"""
POMDP class.
"""

# POMDP class
# Remembe to add discount factor later!!!!!!!!!!!!


class POMDP:
    """
    defines POMDP class.
    """
    def __init__(
        self,
        states,         #
        actions,        #
        observations,   #
        trans_func,     #
        observ_func,    #
        reward_func,    #
        init_belief,    #
        disc_factor,    #
        origin_state      #
    ):
        self.states = states
        self.actions = actions
        self.observations = observations
        self.trans_func = trans_func
        self.observ_func = observ_func
        self.reward_func = reward_func
        self.init_belief = init_belief
        self.disc_factor = disc_factor
        self.orig_state = origin_state


class POMDP_OS:
    """
    defines POMDP class for optimal stopping problem.
    """
    def __init__(
        self,
        name,
        states,         #
        actions,        #
        stops,
        observations,   #
        trans_func,     #
        observ_func,    #
        reward_func,    #
        init_belief,    #
        disc_factor,    #
        origin_state    #
    ):
        self.name = name
        self.states = states
        self.actions = actions
        self.stops = stops
        self.observations = observations
        self.trans_func = trans_func
        self.observ_func = observ_func
        self.reward_func = reward_func
        self.init_belief = init_belief
        self.disc_factor = disc_factor
        self.orig_state = origin_state
