
import os
import time
import copy
import numpy as np
import random
import math
from collections import deque
import matplotlib.pyplot as plt
import csv
import pickle
import time
import data_file
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from multiprocessing import Pool
import logging

def func(_args):
    algo_option = "rl_modified_ddswa"

    train_iter = _args[0]
    sim = _args[1]
    

    if data_file.rl_flag:
        if data_file.rl_algo_opt == "DDPG":
            from ddpg_torch import DDPG as Agent

        elif data_file.rl_algo_opt == "MADDPG":
            from maddpg import DDPG as Agent


        ss = [int(data_file.buff_size*len(data_file.arr_rates_to_simulate)), 64]
        actor_lr = 0.0001
        critic_lr = 0.001
        p_factor = 0.0001
        d_factor = 0.99

        max_learn_iter = 100100

        read_file_path = f"./data/merged_replay_buffer_with_next_state/merged_replay_buffer"

        write_trained_policy_file_path = f"./data/merged_replay_buffer_with_next_state/train_sim_{sim}/trained_weights"

        init_weights_path = f"./data/merged_replay_buffer_with_next_state/train_sim_{sim}/trained_weights"
        

        #### RL agent object creation ####
        if data_file.rl_algo_opt == "DDPG":
            if algo_option == "rl_modified_ddswa":
                agent = Agent(sim, samp_size=ss[1], buff_size=ss[0], act_lr=actor_lr, cri_lr=critic_lr, polyak_factor=p_factor, disc_factor=d_factor)
                
                read_file = open(f"{read_file_path}", 'rb')
                buffer = pickle.load(read_file)
                read_file.close()

                agent.buffer.state_buffer = buffer["state_buffer"]
                agent.buffer.action_buffer = buffer["action_buffer"]
                agent.buffer.reward_buffer = buffer["reward_buffer"]
                agent.buffer.next_state_buffer = buffer["next_state_buffer"]

                agent.buffer.buffer_counter = ss[0]

                for i in range(max_learn_iter):
                    agent.learn()

                    agent.update_target(agent.target_actor_, agent.actor_model_, agent.polyak_factor)
                    agent.update_target(agent.target_critic_, agent.critic_model_, agent.polyak_factor)
                    
                    if (i % 5000) == 0:
                        logging.info('iter:{}'.format(i))
                        torch.save(agent.actor_model_.state_dict(), f"{write_trained_policy_file_path}/actor_weights_itr_{i}.pth")
                        torch.save(agent.critic_model_.state_dict(), f"{write_trained_policy_file_path}/critic_weights_itr_{i}.pth")

                    print(f"learning iteration: {i+1} out of {max_learn_iter}", end="\r")

            elif algo_option == "rl_ddswa":
                agent = Agent(algo_opt=algo_option, state_size=data_file.num_features*data_file.lane_max, action_size=2*data_file.lane_max)
        
        elif data_file.rl_algo_opt == "MADDPG":
            if algo_option == "rl_modified_ddswa":
                agent = Agent(algo_opt=algo_option, num_of_agents=data_file.max_vehi_per_lane*data_file.lane_max, state_size=data_file.num_features, action_size=2)

            elif algo_option == "rl_ddswa":
                agent = Agent(algo_opt=algo_option, state_size=data_file.num_features*data_file.lane_max, action_size=2*data_file.lane_max)
            

if __name__ == '__main__':
    args = []

    for _train_iter in range(1):
        for _sim_num in range(1, 11):
            args.append([_train_iter, _sim_num])

    pool = Pool(10)
    pool.map(func, args)
