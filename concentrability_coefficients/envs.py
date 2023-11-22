"""
    Environments.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


ACTIONS_DICT = {0: 'u', 1: 'd', 2: 'l', 3: 'r', 4: 's'}

class GridEnv:
    
    # Adapted from https://github.com/KaleabTessera/Gridworld-Markov-Decision-Process/blob/master/MDPs.ipynb

    def __init__(self, height, width, starting_state_dist, rewards, time_limit, stochasticity=0.0):
        
        self.width = width
        self.height = height
        
        self.stochasticity = stochasticity

        # Reward function.
        self.R = np.zeros((height,width))

        # Obstacles.
        #self.R[2,:-1] = np.nan

        # Goal state.
        self.rewards = rewards
        for r in self.rewards:
            self.R[r] = 1
        
        # States.
        self.S = [] # state to idx.
        idxs = []
        for index, reward in np.ndenumerate(self.R):
            # Not an obstacle
            if(not np.isnan(reward)):
                self.S.append(index)
                idxs.append(index)
        self.S = np.asarray(self.S)
        self.idx_to_state = {str(k): v for (k,v) in zip(idxs,range(len(idxs)))}

        # Action space.
        self.A = list(ACTIONS_DICT.values())
        
        # Initial state.
        self.starting_state_dist = starting_state_dist

        self.time_limit = time_limit
        self.step_counter = 0
    
    def get_state_space_size(self):
        return len(self.S)    

    def get_action_space_size(self):
        return len(self.A)
    
    def render(self):
            
        grid_world = np.empty((self.width, self.height))
        grid_world[:] = np.nan
        for r in self.rewards:
            grid_world[r] = 1

        grid_world[self.s[0],self.s[1]] = 2

        fig, ax = plt.subplots(figsize=(6,6))

        plt.pcolor(grid_world[::-1], edgecolors='w', linewidths=2)
        loc = matplotlib.ticker.MultipleLocator(base=1.0)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)            

        plt.grid(color='black')
        
        plt.show()
        
        
    def plot_mu(self, mu, vmin, vmax):

        fig, ax = plt.subplots(figsize=(4,4))
        
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        mesh = plt.pcolormesh(mu, norm=norm)
        
        fig.colorbar(mesh,norm=norm)

        plt.grid(color='black')
        plt.show()
        
    def reset(self):
        starting_state = np.random.choice(np.arange(len(self.S)), p=self.starting_state_dist)
        self.s = list(self.S[starting_state])
        self.step_counter = 0
        return self.idx_to_state[str(tuple(self.s))]
            
    def step(self, action):
        new_s, new_r = self.take_step(self.s, action)
        
        if np.random.rand() <= self.stochasticity:
            new_s = np.random.choice(np.arange(len(self.S)))
            new_s = list(self.S[new_s])

        self.s = new_s
            
        self.step_counter += 1
        is_done = False
        if self.step_counter >= self.time_limit:
            is_done = True
            
        return new_r, self.idx_to_state[str(tuple(new_s))], is_done
        
    def check_if_s_in_S(self,s):
        for possible_s in self.S:
            if(possible_s[0] == s[0] and possible_s[1] == s[1]):
                return True
        return False
    
    def is_out_of_bounds(self,new_s):
        if((new_s[0] < 0 or new_s[0]>self.height-1) or (new_s[1] < 0 or new_s[1]>self.width-1)):
            return True
        return False
    
    def take_step(self, s, a):
        if(a not in self.A):
            raise ValueError('Unknown action', a)
        new_s = s.copy()
        if(a == 'u'):
            new_s[0] -= 1
        elif(a == 'd'):
            new_s[0] += 1
        elif(a == 'l'):
            new_s[1] -= 1
        elif(a == 'r'):
            new_s[1] += 1
        elif(a == 's'):
            new_s[1] = new_s[1]

        # Out of bounds - stay in same place
        if(self.is_out_of_bounds(new_s)):
            return s, 0.0
        # Check Obstacles - not in possible states - stay in same place
        elif(not self.check_if_s_in_S(new_s)):
            return s, 0.0
        return new_s, self.R[new_s[0],new_s[1]]
    
    def is_obstacle(self,s):
        if(np.isnan(self.R[s[0],s[1]])):
            return True
        
    def get_R(self):
        rewards_list = []
        for index, reward in np.ndenumerate(self.R):
            # Not an obstacle
            if(not np.isnan(reward)):
                rewards_list.append(reward)
        return np.array(rewards_list)
    
    def get_P(self):
        nS = self.get_state_space_size()
        nA = self.get_action_space_size()
        P = np.zeros((nA,nS,nS))
        for act_idx, act in enumerate(self.A):
            for state_idx, state in enumerate(self.S):
                new_state, _ = self.take_step(state,act)
                new_state_idx = self.idx_to_state[str(tuple(new_state))]
                P[act_idx,state_idx,new_state_idx] = 1.0
                
        for act_idx, act in enumerate(self.A):
            for state_idx, state in enumerate(self.S):
                P[act_idx,state_idx] = P[act_idx,state_idx]*(1-self.stochasticity)
                P[act_idx,state_idx] = P[act_idx,state_idx] + (self.stochasticity / len(P[act_idx,state_idx]))
        
        return P

def run_episode(env, policy, render=False):
    states = []
    actions = []
    cumulative_reward = 0.0
    t = 0

    state = env.reset()
    states.append(state)
    
    if render:
        print("t=", t)
        print("state", state)
        env.render()

    is_done = False
    while not is_done:
        
        action = np.random.choice(np.arange(len(policy[state,:])), p=policy[state,:])
        action = ACTIONS_DICT[action]
        
        reward, state, is_done = env.step(action)
        
        cumulative_reward += reward
        
        actions.append(action)
        states.append(state)

        t += 1

        if render:
            print("action", action)
            print("t=", t)
            print("state", state)
            env.render()
        
    return states, actions, cumulative_reward

def multi_path_mdp_get_transitions_aux(s, a):
    correct_actions = np.random.randint(low=0, high=5, size=5*5+2, dtype=np.int32)
    init_action_random_p = 0.01 # first action randomness.

    if s == 0: # Start state.
        probs = [init_action_random_p]*5
        probs[a] += 1 - init_action_random_p*5
        return [1,6,11,16,21], probs

    elif s == 26: # Terminal (dead) state.
        return [s], [1.0]

    elif s in [5,10,15,20,25]: # Terminal (win) state.
        return [s], [1.0]

    else:
        good_action_idx = correct_actions[s]
        if a == good_action_idx:
            return [s+1], [1.0] # Move one step forward.
        else:
            return [26], [1.0] # Move to dead state.
        
def multi_path_mdp_get_transitions(s, a):
    next_states, probs = multi_path_mdp_get_transitions_aux(s, a)
    return {next_state: prob for next_state, prob in zip(next_states, probs)}

def multi_path_mdp_get_P():
    P_matrix = []
    for a in range(5):
        a_list = []
        for s in range(5*5+2):
            transitions = multi_path_mdp_get_transitions(s,a)
            p_line = np.zeros(5*5+2)
            for s, prob in transitions.items():
                p_line[s] = prob
            a_list.append(p_line)
        P_matrix.append(a_list)
    return np.array(P_matrix)
