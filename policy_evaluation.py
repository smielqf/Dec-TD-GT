
import numpy as np
import tensorflow as tf
import gym
import multiagent
from make_env import make_env
import networkx as nx
# from sklearn.preprocessing import normalize

import time
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from tensorflow.keras.utils import to_categorical
random.seed(0)

class Agent():
    def __init__(self, omega_dim):
        # self.omega = np.asarray([random.uniform(0.0, 1.0,) for i in range(omega_dim)])
        self.omega = np.zeros((omega_dim, ))
        # print(self.omega)
        self.tracking_gradient = np.zeros((omega_dim, ))
        self.pre_gradient = np.zeros((omega_dim, ))
        self.gradient_tracking_initialized = False

    def learn(self, state, reward, next_state, alpha, consensus_omega):
        td_error = reward + 0.95 * np.dot(self.omega, next_state) - np.dot(self.omega, state)
        gradient = td_error * state
        update_omega = consensus_omega + alpha * gradient

        ## ce error
        ce = np.sum(np.square(self.omega - consensus_omega))

        ## sbe error
        sbe = np.square(td_error)

        self.omega = update_omega

        return ce, sbe, gradient

    def gt_init(self, state, reward, next_state):
        td_error = reward + 0.95 * np.dot(self.omega, next_state) - np.dot(self.omega, state)
        cur_gradient = td_error * state * -1
        self.tracking_gradient = np.copy(cur_gradient)
        self.pre_gradient = np.copy(cur_gradient)

    def gt_learn(self, state, reward, next_state, alpha, consensus_omega, consensus_tracking_gradient):
        if self.gradient_tracking_initialized:
            old_gradient = np.copy(self.tracking_gradient)
            td_error = reward + 0.95 * np.dot(self.omega, next_state) - np.dot(self.omega, state)
            cur_gradient = td_error * state * -1
            self.tracking_gradient = consensus_tracking_gradient + cur_gradient  - self.pre_gradient
            self.omega = consensus_omega - alpha * self.tracking_gradient
            self.pre_gradient = np.copy(cur_gradient)
        else:
            td_error = reward + 0.95 * np.dot(self.omega, next_state) - np.dot(self.omega, state)
            cur_gradient = td_error * state * -1
            self.pre_gradient = np.copy(cur_gradient)
            old_gradient = np.copy(cur_gradient)
            self.omega = consensus_omega - alpha * self.tracking_gradient
            self.gradient_tracking_initialized = True

        return old_gradient
    
    def gc_learn(self, alpha, gradient_consensus, omega_consensus):
        """
            Policy evaluation using gradient consenus rather than model consensus.
        """
        # self.omega = omega_consensus + alpha * gradient_consensus
        self.omega = omega_consensus + alpha * gradient_consensus

    def gt_gc_init(self, gradient_consensus):
        self.tracking_gradient = np.copy(gradient_consensus)
        self.pre_gradient = np.copy(gradient_consensus)

    def gt_gc_learn(self, alpha, consensus_omega, gradient_consensus, consensus_tracking_gradient):
        if self.gradient_tracking_initialized:
            old_gradient = np.copy(self.tracking_gradient)
            self.tracking_gradient = consensus_tracking_gradient + gradient_consensus  - self.pre_gradient
            self.omega = consensus_omega + alpha * self.tracking_gradient
            self.pre_gradient = np.copy(gradient_consensus)
        else:
            self.pre_gradient = np.copy(gradient_consensus)
            old_gradient = np.copy(gradient_consensus)
            self.omega = consensus_omega + alpha * self.tracking_gradient
            self.gradient_tracking_initialized = True

        return old_gradient

    def gt_cta_learn(self, state, reward, next_state, alpha, consensus_tracking_gradient):
        if self.gradient_tracking_initialized:
            old_gradient = np.copy(self.tracking_gradient)
            td_error = reward + 0.95 * np.dot(self.omega, next_state) - np.dot(self.omega, state)
            cur_gradient = td_error * state * -1
            self.tracking_gradient = consensus_tracking_gradient + cur_gradient  - self.pre_gradient
            self.omega = self.omega - alpha * self.tracking_gradient
            self.pre_gradient = np.copy(cur_gradient)
        else:
            td_error = reward + 0.95 * np.dot(self.omega, next_state) - np.dot(self.omega, state)
            cur_gradient = td_error * state * -1
            self.pre_gradient = np.copy(cur_gradient)
            old_gradient = np.copy(cur_gradient)
            self.omega = self.omega - alpha * self.tracking_gradient
            self.gradient_tracking_initialized = True
        return old_gradient


    def val(self, state):
        return np.dot(self.omega, state)


def consensus(agents, commu_graph):
    adjacent_matrix = np.asarray(nx.to_numpy_matrix(commu_graph))
    n_agent = len(agents)
    omega_list = [[] for _ in range(n_agent)]
    tracking_gradient_list = [[] for _ in range(n_agent)]
    for i, agent in enumerate(agents):
        for j, _agent in enumerate(agents):
            if i == j or adjacent_matrix[i, j] > 0.0:
                omega_list[i].append(_agent.omega)
                tracking_gradient_list[i].append(np.copy(_agent.tracking_gradient))
    
    consensus_omega = [np.mean(np.asarray(x), axis=0) for x in omega_list]
    tracking_gradient_consensus = [np.mean(np.asarray(x), axis=0) for x in tracking_gradient_list]
    return consensus_omega, tracking_gradient_consensus

def consensus_omega(agents, commu_graph):
    adjacent_matrix = np.asarray(nx.to_numpy_matrix(commu_graph))
    n_agent = len(agents)
    omega_list = [[] for _ in range(n_agent)]
    for i, agent in enumerate(agents):
        for j, _agent in enumerate(agents):
            if i == j or adjacent_matrix[i, j] > 0.0:
                omega_list[i].append(_agent.omega)
    
    consensus_omega = [np.mean(np.asarray(x), axis=0) for x in omega_list]
    return consensus_omega

def consensus_gradient(agents, commu_graph, state, reward, next_state):
    gradient_list = []
    adjacent_matrix = np.asarray(nx.to_numpy_matrix(commu_graph))
    for i, agent in enumerate(agents):
        td_error = reward[i] + 0.95 * np.dot(agent.omega, next_state[i]) - np.dot(agent.omega, state[i])
        gradient = td_error * state[i]
        gradient_list.append(np.copy(gradient))
    gradient_consensus_list = [[] for _ in range(len(agents))]
    for i, agent in enumerate(agents):
        for j, _agent in enumerate(agents):
            if i == j or adjacent_matrix[i, j] > 0.0:
                gradient_consensus_list[i].append(np.copy(gradient_list[j]))
    gradient_consensus = [np.mean(np.asarray(x), axis=0) for x in gradient_consensus_list]
    return gradient_consensus

def cta_grad(agents, commu_graph, state, reward, next_state):
    gradient_list = [[] for _ in range(len(agents))]
    adjacent_matrix = np.asarray(nx.to_numpy_matrix(commu_graph))
    for i, agent in enumerate(agents):
        for j, _agent in enumerate(agents):
            if i == j or adjacent_matrix[i, j] > 0.0:
                td_error = reward[j] + 0.95 * np.dot(agent.omega, next_state[j]) - np.dot(agent.omega, state[j])
                gradient = td_error * state[j]
                gradient_list[i].append(np.copy(gradient))
    gradient_consensus = [np.mean(np.asarray(x), axis=0) for x in gradient_list]
    return gradient_consensus

def compute_ce(agents):
    omega = []
    for i, agent in enumerate(agents):
        omega.append(agent.omega)
    omega = np.asarray(omega)
    omega = np.mean(omega, axis=0)
    ce = []
    for i, agent in enumerate(agents):
        ce_error = np.mean(np.square(agent.omega - omega))
        ce.append(ce_error)
    return np.mean(ce)

def compute_sbe(agents, state, reward, next_state):
    mean_reward = np.mean(reward)
    sbe = []
    for i, agent in enumerate(agents):
        td_error = mean_reward + 0.95 * np.dot(agent.omega, next_state[i]) - np.dot(agent.omega, state[i])
        local_sbe = np.square(td_error)
        sbe.append(local_sbe)
    return np.mean(sbe)
    


    



env = make_env('simple_spread_custom_local_9',benchmark=False)
# env.seed(3)  # reproducible

N_state = env.observation_space[0].shape[0]
N_agent = env.n
N_action = env.action_space[0].n

# er = nx.erdos_renyi_graph(N_agent , 1.0, seed=0)
er= nx.cycle_graph(N_agent)
nx.draw(er)
plt.savefig('network.png')


# agent = Agent(N_state)
agent_list = [Agent(N_state)  for i in range(N_agent)]

ce_list = []
sbe_list = []
gradient_list = []
tracking_gradient_list = []

GT_INIT = False
mode = 3
if mode == 0:
    postfix = ''
elif mode == 1:
    postfix = '_gt'
elif mode == 2:
    postfix = '_gc'
elif mode == 3:
    postfix = '_gt_gc'
# elif mode == 4:
#     postfix = '_cta'

for i_episode in range(1, 2):
    s = env.reset()
    ce_episode = []
    sbe_episode = []
    gradient_episode = [[] for _ in range(N_agent)]
    tracking_gradient_episode = [[] for _ in range(N_agent)]
    # alpha = 0.05/np.sqrt(i_episode)
    # alpha=0.001
    for t in range(500):
        alpha = 0.1/np.sqrt(t+1)
        # action_n = [np.random.uniform(low=-1.0, high=1.0, size=2) for i in range(N_agent)]
        s = np.array([s[0]] * N_agent)
        action_n = np.random.randint(N_action, size=(env.n,1))
        onehot_n = to_categorical(action_n, N_action)
        s_, r, done, info = env.step(onehot_n)
        r= [ x for x in r]
        s_ = np.asarray([s_[0]] * N_agent)
        

        # consensus step
        if mode == 1 and GT_INIT is False:
            for i, agent in enumerate(agent_list):
                agent.gt_init(s[i], r[i], s_[i])
            GT_INIT = True
        omega_consensus, tracking_gradient_consensus = consensus(agent_list, er)
        gc_gradient_consensus = consensus_gradient(agent_list, er, s, r, s_)
        ce_error = compute_ce(agent_list)
        sbe_error = compute_sbe(agent_list, s, r, s_)
        if mode == 3 and GT_INIT is False:
            for i, agent in enumerate(agent_list):
                agent.gt_gc_init(gc_gradient_consensus[i])
            GT_INIT = True


        ce_s = []
        sbe_s = []
        for i, agent in enumerate(agent_list):
            if mode == 0: # Dis-TTD(0)
                ce, sbe, gradient = agent.learn(s[i], r[i], s_[i], alpha, omega_consensus[i])
                gradient_episode[i].append(gradient)
                # ce_s.append(ce)
                # sbe_s.append(sbe)
            elif mode == 1: # GT
                tracking_gradient = agent.gt_learn(s[i], r[i], s_[i], alpha, omega_consensus[i], tracking_gradient_consensus[i])
                tracking_gradient_episode[i].append(tracking_gradient)
            elif mode == 2: # GC
                agent.gc_learn(alpha, gc_gradient_consensus[i], omega_consensus[i])
            elif mode == 3: # GT-GC
                agent.gt_gc_learn(alpha, omega_consensus[i], gc_gradient_consensus[i], tracking_gradient_consensus[i])
        # if mode == 4: # CTA
        #     for i, agent in enumerate(agent_list):
        #         agent.omega = np.copy(omega_consensus[i])
        #         # cta_gradient_consensus = consensus_gradient(agent_list, er, s, r, s_)
        #         cta_gradient_consensus = cta_grad(agent_list, er, s, r, s_)
        #         agent.omega = agent.omega + alpha * cta_gradient_consensus[i]

        # ce_episode.append(np.mean(ce_s))
        # if mode == 2:
        #     omega_consensus = consensus_omega(agent_list, er)
        #     for i, agent in enumerate(agent_list):
        #         agent.omega = np.copy(omega_consensus[i])
        
        ce_episode.append(ce_error)
        # sbe_episode.append(np.mean(sbe_s))
        sbe_episode.append(sbe_error)

        s = np.copy(s_)
    # ce_list.append(np.mean(ce_episode))
    # sbe_list.append(np.mean(sbe_episode))
    ce_list = ce_episode[:]
    sbe_list = sbe_episode[:]

plt.clf()
plt.plot(ce_list)
# plt.yticks([ x * 0.002 for x in range(8)])
# plt.ylim(0.0, 0.014)
plt.savefig('ce{}.png'.format(postfix))
plt.clf()
plt.plot(sbe_list)
# plt.ylim(0.0, 50)
plt.savefig('sbe{}.png'.format(postfix))
np.savetxt('ce{}_list.txt'.format(postfix), ce_list, fmt='%.8lf')
np.savetxt('sbe{}_list.txt'.format(postfix), sbe_list, fmt='%.8lf')