import random as rand
import numpy as np


class QLearner(object):
    """
    Tabular Q-learning agent with optional Dyna-style planning.

    - Uses an epsilon-greedy policy (rar) to balance exploration/exploitation.
    - Updates Q(s,a) using the standard Bellman target:
        Q(s,a) <- (1-alpha)*Q(s,a) + alpha*(r + gamma*max_a' Q(s',a'))
    - If dyna > 0, the agent also performs additional synthetic updates using
      a learned model of rewards and transitions.
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  

    def __init__(  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self,  		  	   		 	 	 		  		  		    	 		 		   		 		  
        num_states=100,  		  	   		 	 	 		  		  		    	 		 		   		 		  
        num_actions=4,  		  	   		 	 	 		  		  		    	 		 		   		 		  
        alpha=0.2,  		  	   		 	 	 		  		  		    	 		 		   		 		  
        gamma=0.9,  		  	   		 	 	 		  		  		    	 		 		   		 		  
        rar=0.5,  		  	   		 	 	 		  		  		    	 		 		   		 		  
        radr=0.99,  		  	   		 	 	 		  		  		    	 		 		   		 		  
        dyna=0,  		  	   		 	 	 		  		  		    	 		 		   		 		  
        verbose=False,  		  	   		 	 	 		  		  		    	 		 		   		 		  
    ):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """
        Initialize a tabular Q-learning agent.
    
        Parameters
        ----------
        num_states : int
            Number of discrete states.
        num_actions : int
            Number of discrete actions.
        alpha : float
            Learning rate.
        gamma : float
            Discount factor.
        rar : float
            Exploration probability (epsilon).
        radr : float
            Exploration decay multiplier applied each step.
        dyna : int
            Number of planning updates per real experience.
        verbose : bool
            Print debug info when True.
        """
      
        self.verbose = verbose
        self.num_actions = num_actions
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.s = 0
        self.a = 0
        self.R_model = np.zeros((num_states, num_actions)) # estimated immediate reward
        self.Tc = np.zeros((num_states, num_actions, num_states)) * 1e-5 # transition counts with small values as described in lecture
        self.seen = set()  # store tuples (s, a) - O(1) lookup
        self.seen_list = [] # O(1) random indexing

    def set_state(self, s):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """
        Set the current state and choose an action without updating Q.
    
        Returns an epsilon-greedy action based on the current Q-table.
        """ 		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.s = s  		  	   		 	 	 		

        r = rand.uniform(0, 1)

        if r < self.rar: # explore -> pick random action
            action = rand.randint(0, self.num_actions - 1)

        else: # r >= rar -> pick best action from Q-table
            action = np.argmax(self.Q[s, :])

        self.a = int(action) 

        if self.verbose:
            print(f"s = {s}, a = {action}")

        return action
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    def step(self, s_prime, r):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        """
        Update Q using (s, a, r, s') and return the next epsilon-greedy action.
    
        Also performs optional Dyna planning updates if enabled.
        """  		  	   		 	 	 		  	

        prev_s = self.s
        prev_a = self.a

        best_pred_return = np.max(self.Q[s_prime, :]) # max over actions

        target = r + self.gamma * best_pred_return

        # Update Q-table, using Update equation from lecture
        self.Q[prev_s, prev_a] = (1 - self.alpha) * self.Q[prev_s, prev_a] + self.alpha * target

        self.rar *= self.radr

        # Learning R -> DYNA
        self.R_model[prev_s, prev_a] = (1 - self.alpha) * self.R_model[prev_s, prev_a] + self.alpha * r

        # Learning T -> DYNA
        self.Tc[prev_s, prev_a, s_prime] += 1

        # visited set, to prevent infinite loops 
        pair = (prev_s, prev_a)
        if pair not in self.seen:
            self.seen.add(pair)
            self.seen_list.append(pair) # to sample from the cached list and not rebuild for every dyna iteration

        # Dyna loop (derived from lecture/discussion post pseudocode) -> https://edstem.org/us/courses/81415/discussion/6862747
        for _ in range(self.dyna):
            s_rand, a_rand = rand.choice(self.seen_list) # randomly pick a past (state, action) pair to replay

            # prob = self.Tc[s_rand, a_rand, :] / np.sum(self.Tc[s_rand, a_rand, :]) # as per formula from lecture 
            # # ref -> https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
            # np.random.choice performance mentioned in https://edstem.org/us/courses/81415/discussion/7239629
            # & https://edstem.org/us/courses/81415/discussion/7233048
            # s_rand_prime = np.random.choice(self.Q.shape[0], p=prob)

            # find the most likely next state given that (s,a), based on the transition counts Tc
            counts = self.Tc[s_rand, a_rand, :]
            if counts.sum() <= 0: # guard: skip if we've never seen a next state
                continue
            
            # instead of sampling by probability (slow w/ np.random.choice), argmax (most frequent successor) is equivalent when environment is deterministic
            # finds the index of the largest value (most frequently observed next state)
            s_rand_prime = int(np.argmax(counts))

            # retrieve the expected reward for (s,a) from learned R-model (hallucination)
            r_rand = self.R_model[s_rand, a_rand]

            # perform a synthetic Q-update on that (s,a), using model-generated transition ("planning step", without using real world exp)
            target_rand = r_rand + self.gamma * (np.max(self.Q[s_rand_prime, :]))

            self.Q[s_rand, a_rand] = (1 - self.alpha) * self.Q[s_rand, a_rand] + self.alpha * target_rand

        # ε-greedy action selection:
        if rand.uniform(0, 1) < self.rar:
            # with probability rar, take a random exploratory action (exploration)
            action = rand.randint(0, self.num_actions - 1)
        else:
            # otherwise choose the best known action from the Q-table (exploitation)
            action = int(np.argmax(self.Q[s_prime, :]))
		  
        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")

        self.s = s_prime
        self.a = action

        return action

if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  	  	   	
    # learner = QLearner(num_states=100, num_actions=4)
    # print(learner.Q.shape)
    # print(learner.alpha, learner.gamma, learner.rar)

    # init learner w/ small numbers
    learner = QLearner(num_states=5, num_actions=4, rar=0.5, radr=0.99, verbose=True)

    # fill the Q-table with values to make greedy behavior visible
    learner.Q[2, :] = [0.1, 0.5, 0.9, 0.3]

    # call set_state several times from the same state
    print("Testing set_state() on state 2:")
    for i in range(10):
        a = learner.set_state(2)
        print(f"Run {i+1}: chose action {a}")

    print("Testing Dyna:")
    learner = QLearner(num_states=5, num_actions=3, dyna=200, verbose=False)

    # simulate one real experience
    print("Running one step with Dyna:")
    learner.set_state(1)
    learner.step(2, 1.0)  # experience: from state 1 -> took action a -> got reward 1.0 -> went to 2
    
    # ref -> https://www.geeksforgeeks.org/python/numpy-count_nonzero-method-python/
    print("Nonzero Tc entries:", np.count_nonzero(learner.Tc))
    print("Nonzero R_model entries:", np.count_nonzero(learner.R_model))
    print("Sample Q-values:\n", learner.Q)
