import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import matplotlib.pyplot as plt
import data_file
import math
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SeqModule(nn.Module):
    def __init__(self, lat_size):
        super(SeqModule, self).__init__()
        self.lat_size = lat_size
        self.flatten = nn.Flatten()
        self.bigru = nn.GRU(
            input_size=data_file.num_features * data_file.num_lanes * 4,
            hidden_size=lat_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), data_file.num_veh, -1)

        # print('input',inputs.size())
        
        # Create mask
        # lengths = (inputs != 0).all(dim=1).sum(dim=1).cpu()
        lengths=torch.sum(torch.abs(inputs).sum(dim=2)>0,dim=1)
        # print('lengths',lengths)  # Calculate sequence lengths
        # np.save('data.npy', input)
        # print('saved')
        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.bigru(packed_inputs)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True,total_length=112)

        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        # print('poutput',packed_output)

        return output, (hidden_forward, hidden_backward)
    
class CNN(nn.Module):
    def __init__(self, kernel_num, kernel_size, stride):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(1, kernel_num, kernel_size=kernel_size, stride=stride)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), 1,x.size(1) * x.size(2))
        x = self.conv(x)
        x = self.activation(x)
        return x


class ActorModel(nn.Module):
    def __init__(self):
        super(ActorModel, self).__init__()
        self.layer_conv64 = CNN(64, 128, 128)
        self.layer_conv32 = CNN(32, 64, 64)
        self.layer_conv48 = CNN(48, 96, 96)
        self.layer_conv24 = CNN(24, 48, 48)
        self.layer_conv12 = CNN(12, 24, 24)
        self.layer_conv16_12 = CNN(12, 16, 16)
        self.layer_conv16 = CNN(16, 32, 32)
        self.layer_conv1_12 = CNN(1, 12, 12)
        self.layer_conv8 = CNN(8, 16, 16)
        self.flatten = nn.Flatten()

    def forward(self, seq, latent):
        latent_1 = latent.unsqueeze(1)
        latent = latent_1.repeat(1, data_file.num_veh, 1)
        # print('latent',latent.size())
        state = self.layer_conv64(latent)
        obs = self.layer_conv32(self.layer_conv64(seq))
        # print('seq',seq.size())
        # print(seq)
        # print('state',state.size())
        # print('obs',obs.size())
        # exit()
        # input_1 = self.output_concat([state, obs])
        input_1 = torch.cat((state, obs),1)
        control = self.layer_conv1_12(self.layer_conv12(self.layer_conv24(self.layer_conv48(input_1))))
        control = self.flatten(control)
        print(control.size())
        signal = self.layer_conv16_12(self.layer_conv16(self.layer_conv32(self.layer_conv64(latent_1))))
        signal = signal.view(signal.size(0), -1)
        signal = self.flatten(signal)
        print(signal.size())
        # outputs = self.output_concat([signal, control])
        outputs = torch.cat((signal, control),1)
        return outputs


class CustomLayerNormalization(nn.Module):
    def __init__(self, epsilon=1e-6,axis=1, mask_value=None, center=True, scale=True):
        super(CustomLayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.axis = axis
        self.mask_value = mask_value
        self.center = center
        self.scale = scale

    def forward(self, inputs, mask):
        mask = mask.unsqueeze(1).type_as(inputs)
        # print('i',inputs.size())
        inputs = inputs.squeeze(1)

        masked_sum = torch.sum(inputs * mask, dim=1, keepdim=True)
        # print('ms',masked_sum.size())
        masked_count = torch.sum(mask, dim=-1, keepdim=True)
        # print('mc',masked_count.size())
        masked_mean = masked_sum / masked_count
        # print('mm',masked_mean.size())
        masked_diff = (inputs - masked_mean) * mask
        # print('md',masked_diff.size())
        masked_variance = torch.sum(masked_diff ** 2, dim=-1, keepdim=True) / masked_count
        # print('i-m',(inputs - masked_mean).size())
        normalized_inputs = (inputs - masked_mean) / torch.sqrt(masked_variance + self.epsilon)
        # print('n_i',normalized_inputs.size())

        # mean = torch.mean(inputs, dim=self.axis, keepdim=True)
        # print('mean',mean.size())
        # variance = torch.var(inputs, dim=self.axis, keepdim=True, unbiased=False)
        # print('variance',variance.size())
        # normalized_inputs = (inputs - mean) / torch.sqrt(variance + self.epsilon)
        # print('n_i',normalized_inputs.size())
        # return normalized_inputs.unsqueeze(-1)

class CriticModel(nn.Module):
    def __init__(self, lat_size):
        super(CriticModel, self).__init__()
        self.lay_norm = CustomLayerNormalization()
        self.gru_action = nn.GRU(124, 128,num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
    def forward(self, latent_state, action):
        action = action.unsqueeze(1)

        # Create mask
        mask = (action != -190.392).float()

        action_lengths = (action != -190.392).all(dim=2).sum(dim=1).cpu()
        # Normalize action
        norm_action=action
        #######norm_action = self.lay_norm(action, mask).squeeze()
        ##########norm_action=norm_action.unsqueeze(1)
        print('n_a',norm_action.size())
        # Pack padded sequence
        # packed_action = pack_padded_sequence(norm_action, action_lengths, batch_first=True, enforce_sorted=False)
        # # GRU forward
        # packed_latent_act, _ = self.gru_action(packed_action)
        # latent_act, _ = pad_packed_sequence(packed_latent_act, batch_first=True)
        latent_act,_=self.gru_action(norm_action)
        print('l_a1',latent_act.size())
        # Extract last valid output from GRU
        idx = (action_lengths - 1).view(-1, 1, 1).expand(latent_act.size(0), 1, latent_act.size(2))#.to(latent_act)#.device)
        print('l_a2',latent_act.size())
        # latent_act = latent_act.gather(1, idx)
        latent_act=latent_act.squeeze(dim=1)
        print('l_a',latent_act.size())
        print('l_s',latent_state.size())
        # Concatenate latent_state and latent_act
        concat = torch.cat([latent_state, latent_act], dim=-1)
        print('c',concat.size())
        # Fully connected layers
        # exit()
        Q_val = F.relu(self.fc1(concat))
        Q_val = F.relu(self.fc2(Q_val))
        Q_val = F.relu(self.fc3(Q_val))
        Q_val = self.fc4(Q_val)

        return Q_val

   #################################### NN model is same for the og###################################

class NNModel(nn.Module):
    def __init__(self, lat_size):
        super(NNModel, self).__init__()
        self.RNN = SeqModule(lat_size)
        self.lat_size = lat_size
        self.actor_model = ActorModel()
        self.critic_model = CriticModel(self.lat_size)
        # self.critic_model = CriticModel()

    def forward(self, input, action=None, act_or_cri='act'):
        assert input.shape[0] == data_file.rl_ddpg_samp_size or input.shape[0] == 1, f'wrong dimensions, input_batch: {input.shape[0]}'
        seq, (forward_state, backward_state) = self.RNN(input)
        latent = torch.cat([forward_state, backward_state], dim=-1)
        # print('input',input.size())
        # print('seq_nn',seq.size())
        # print('latent_nn',latent.size())
        if act_or_cri == 'act':
            y_hat = self.actor_model(seq, latent)
            # y_hat = self.actor_model(input)

            print(y_hat.size())
            return y_hat
        elif act_or_cri == 'cri':
            print('a',action.size())
            # print('a',action)
            return self.critic_model(latent, action)
            # return self.critic_model(input, action)
####################################################### Dummy NN model#################################

def float_precision(value, precision):
    #print(fvdgdgghf)
    float128_value = np.float64(value)
    truncated_value = np.round(float128_value, precision)
    result = float(truncated_value)
    return result


# class NNModel(nn.Module):
#     def __init__(self, hidden_dim):
#         super(NNModel, self).__init__()
#         self.actor_model = nn.Sequential(
#             nn.Linear(6272, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 124)  # Assuming 17 is num_actions
#         )
#         self.critic_model = nn.Sequential(
#             nn.Linear(6272 + 124, hidden_dim),  # State + Action
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )

#     def forward(self, state, act_or_cri='act', action=None):
#         if act_or_cri == 'act':
#             return self.actor_model(state)
#         elif act_or_cri == 'cri':
#             return self.critic_model(torch.cat([state, action], dim=-1))
#####################################################################################################

class Buffer:
    def __init__(self, buffer_capacity=700000, batch_size=64, state_size=12, observe_size=None, action_size=5, buff_model=None, buff_model_target=None, gamma=0.99, tau=0.001, critic_optimizer=None, actor_optimizer=None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.obs_size = observe_size
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.model_buff = buff_model
        self.model_target_buff = buff_model_target

        self.gamma = gamma
        self.tau = tau

        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer

        self.state_buffer = np.zeros((self.buffer_capacity, self.state_size))
        self.observe_buffer = np.zeros((self.buffer_capacity, self.state_size))
        self.action_buffer = np.zeros((self.buffer_capacity, self.action_size))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.state_size))
        self.next_observe_buffer = np.zeros((self.buffer_capacity, self.state_size))

    def remember(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        assert len(obs_tuple[0]) == self.state_size
        assert len(obs_tuple[2]) == self.action_size, f'replay assignment error state:{self.state_size},{len(obs_tuple[0])}, action:{self.action_size},{len(obs_tuple[2])}'
        assert len(obs_tuple[4]) == self.state_size, f'replay assignment error'
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[2]
        self.reward_buffer[index] = obs_tuple[3]
        self.next_state_buffer[index] = obs_tuple[4]
        self.buffer_counter += 1

    def update(self, state_batch, obs_batch, action_batch, reward_batch, next_state_batch, next_obs_batch):
        self.model_buff.train()
        self.model_target_buff.eval()

        with torch.no_grad():
            target_actions = self.model_target_buff(next_state_batch, act_or_cri='act')
            # print(next_state_batch)
            # print(self.model_target_buff(next_state_batch, act_or_cri='cri', action=target_actions))
            y = reward_batch + self.gamma * self.model_target_buff(next_state_batch, act_or_cri='cri', action=target_actions)

        critic_value = self.model_buff(state_batch, act_or_cri='cri', action=action_batch)
        critic_loss = torch.mean((y - critic_value) ** 2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actions = self.model_buff(state_batch, act_or_cri='act')
        actor_loss = -self.model_buff(state_batch, act_or_cri='cri', action=actions).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

    def learn(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)
        state_batch = torch.tensor(self.state_buffer[batch_indices], dtype=torch.float32)#.to(device)
        obs_batch = torch.tensor(self.observe_buffer[batch_indices], dtype=torch.float32)#.to(device)
        action_batch = torch.tensor(self.action_buffer[batch_indices], dtype=torch.float32)#.to(device)
        reward_batch = torch.tensor(self.reward_buffer[batch_indices], dtype=torch.float32)#.to(device)
        next_state_batch = torch.tensor(self.next_state_buffer[batch_indices], dtype=torch.float32)#.to(device)
        next_obs_batch = torch.tensor(self.next_observe_buffer[batch_indices], dtype=torch.float32)#.to(device)
        self.update(state_batch, obs_batch, action_batch, reward_batch, next_state_batch, next_obs_batch)

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=5e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.t = 0
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        self.t += self.dt
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + self.std_dev * np.random.normal(size=self.mean.shape) * (1 / self.t))
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class DDPG:
    def __init__(self, sim=0, noise_mean=0, noise_std_dev=0.2, cri_lr=0.001, act_lr=0.0001, disc_factor=0.99, polyak_factor=0.001, buff_size=1000, samp_size=64):
        # self.num_states = 6272
        # self.num_obs = 784
        # self.num_actions = 17
        # self.noise_std_dev = noise_std_dev

        self.num_states = (data_file.num_features*data_file.num_lanes*4)*data_file.num_veh
        self.num_obs = data_file.num_features
        self.num_actions = data_file.num_veh + data_file.num_phases
        self.noise_std_dev = noise_std_dev

        self.ou_noise = OUActionNoise(mean=np.zeros(self.num_actions), std_deviation=float(self.noise_std_dev) * np.ones(self.num_actions))
        self.model = NNModel(64)#.to(device) 64 should be inside the ()
        self.target_model = NNModel(64)#.to(device)  64 should be inside the ()


        self.target_model.load_state_dict(self.model.state_dict())

        self.critic_lr = cri_lr
        self.actor_lr = act_lr

        self.critic_optimizer = optim.Adam(self.model.critic_model.parameters(), lr=self.critic_lr)
        self.actor_optimizer = optim.Adam(self.model.actor_model.parameters(), lr=self.actor_lr)
       
        self.gamma = disc_factor
        self.tau = polyak_factor
        self.buff_size = buff_size
        self.samp_size = samp_size
        self.buffer = Buffer(buffer_capacity=self.buff_size, batch_size=self.samp_size, state_size=self.num_states, observe_size=self.num_obs, action_size=self.num_actions, buff_model=self.model, buff_model_target=self.target_model, gamma=self.gamma, tau=self.tau, critic_optimizer=self.critic_optimizer, actor_optimizer=self.actor_optimizer)

    def update_target(self, target_model, model, tau):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def sigmoid_func(self, np_array):
        temp_array = []
        for t in np_array:
            trunc_val = float_precision(t, 8)   #np.float128(t)
            if trunc_val< -700:
                temp_array.append(0)
            else: #
                temp_array.append(1 / (1 + np.exp(- trunc_val)))
            #temp_array.append(np.exp(trunc_val) / (1 + np.exp(trunc_val)))
            # print(temp_array)
            # exit
            #assert temp_array[-1]<0 and temp_array[-1]>0,f'ori value:{t},trun val :{trunc_val}, sig:{1 / (1 + np.exp(- trunc_val))}'
        return np.asarray(temp_array)

    def policy(self, state, observation, noise_object, num_veh=None):
        state_has_nan = np.isnan(state).any()
        assert not state_has_nan, f'input values-bad : state: {state}, obs: {observation}'
        state = torch.tensor(state, dtype=torch.float32)#.to(device)
        self.model.eval()
        with torch.no_grad():
            sampled_actions = self.model(state, act_or_cri='act').squeeze().cpu().numpy()
        if noise_object is not None:
            noise = noise_object()
            sampled_actions = sampled_actions + noise
        legal_action = self.sigmoid_func(sampled_actions[data_file.num_phases: (data_file.num_veh + data_file.num_phases)])
        legal_action_ph = sampled_actions[:data_file.num_phases]
        legal_action_set = np.append(legal_action_ph, legal_action)
        assert all([_ >= 0 and _ <= 1 for _ in legal_action_set[data_file.num_phases:(data_file.num_veh + data_file.num_phases)]]), f'alpha value not in range :{legal_action_set[data_file.num_phases:(data_file.num_veh + data_file.num_phases)]}, signal:{legal_action_set[:data_file.num_phases]}'
        return [np.squeeze(legal_action_set)]





# class Buffer:
#     def __init__(self, buffer_capacity=700000, batch_size=64, state_size=12, observe_size=None, action_size=5, buff_model=None, buff_model_target=None, gamma=0.99, tau=0.001, critic_optimizer=None, actor_optimizer=None):
        
#         self.state_size = state_size
#         self.action_size = action_size
#         self.obs_size = observe_size
#         self.buffer_capacity = buffer_capacity
#         self.batch_size = batch_size
#         self.buffer_counter = 0

#         self.model_buff = buff_model
#         self.model_target_buff = buff_model_target

#         self.gamma = gamma
#         self.tau = tau

#         self.critic_optimizer = critic_optimizer
#         self.actor_optimizer = actor_optimizer

#         self.state_buffer = np.zeros((self.buffer_capacity, self.state_size))
#         self.observe_buffer = np.zeros((self.buffer_capacity, self.state_size))
#         self.action_buffer = np.zeros((self.buffer_capacity, self.action_size))
#         self.reward_buffer = np.zeros((self.buffer_capacity, 1))
#         self.next_state_buffer = np.zeros((self.buffer_capacity, self.state_size))
#         self.next_observe_buffer = np.zeros((self.buffer_capacity, self.state_size))

#     def remember(self, obs_tuple):
#         index = self.buffer_counter % self.buffer_capacity
#         assert len(obs_tuple[0]) == self.state_size
#         assert len(obs_tuple[2]) == self.action_size, f'replay assignment error state:{self.state_size},{len(obs_tuple[0])}, action:{self.action_size},{len(obs_tuple[2])}'
#         assert len(obs_tuple[4]) == self.state_size, f'replay assignment error'
#         self.state_buffer[index] = obs_tuple[0]
#         self.action_buffer[index] = obs_tuple[2]
#         self.reward_buffer[index] = obs_tuple[3]
#         self.next_state_buffer[index] = obs_tuple[4]
#         self.buffer_counter += 1

#     def update(self, state_batch, obs_batch, action_batch, reward_batch, next_state_batch, next_obs_batch):
#         self.model_buff.train()
#         self.model_target_buff.eval()

#         with torch.no_grad():
#             target_actions = self.model_target_buff(next_state_batch, act_or_cri='act')
#             # print(self.model_target_buff(next_state_batch, act_or_cri='cri', action=target_actions))
#             y = reward_batch + self.gamma * self.model_target_buff(next_state_batch, act_or_cri='cri', action=target_actions)

#         critic_value = self.model_buff(state_batch, act_or_cri='cri', action=action_batch)
#         critic_loss = torch.mean((y - critic_value) ** 2)

#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_optimizer.step()

#         self.actor_optimizer.zero_grad()
#         actions = self.model_buff(state_batch, act_or_cri='act')
#         actor_loss = -self.model_buff(state_batch, act_or_cri='cri', action=actions).mean()
#         actor_loss.backward()
#         self.actor_optimizer.step()

#     def learn(self):
#         record_range = min(self.buffer_counter, self.buffer_capacity)
#         batch_indices = np.random.choice(record_range, self.batch_size)
#         state_batch = torch.tensor(self.state_buffer[batch_indices], dtype=torch.float32)#.to(device)
#         obs_batch = torch.tensor(self.observe_buffer[batch_indices], dtype=torch.float32)#.to(device)
#         action_batch = torch.tensor(self.action_buffer[batch_indices], dtype=torch.float32)#.to(device)
#         reward_batch = torch.tensor(self.reward_buffer[batch_indices], dtype=torch.float32)#.to(device)
#         next_state_batch = torch.tensor(self.next_state_buffer[batch_indices], dtype=torch.float32)#.to(device)
#         next_obs_batch = torch.tensor(self.next_observe_buffer[batch_indices], dtype=torch.float32)#.to(device)
#         self.update(state_batch, obs_batch, action_batch, reward_batch, next_state_batch, next_obs_batch)

# class OUActionNoise:
#     def __init__(self, mean, std_deviation, theta=0.15, dt=5e-2, x_initial=None):
#         self.theta = theta
#         self.mean = mean
#         self.std_dev = std_deviation
#         self.dt = dt
#         self.t = 0
#         self.x_initial = x_initial
#         self.reset()

#     def __call__(self):
#         self.t += self.dt
#         x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + self.std_dev * np.random.normal(size=self.mean.shape) * (1 / self.t))
#         self.x_prev = x
#         return x

#     def reset(self):
#         if self.x_initial is not None:
#             self.x_prev = self.x_initial
#         else:
#             self.x_prev = np.zeros_like(self.mean)

# class DDPG:
#     def __init__(self, sim=0, noise_mean=0, noise_std_dev=0.2, cri_lr=0.001, act_lr=0.0001, disc_factor=0.99, polyak_factor=0.001, buff_size=1000, samp_size=64):
#         # self.num_states = 6272
#         # self.num_obs = 784
#         # self.num_actions = 17
#         # self.noise_std_dev = noise_std_dev

#         self.num_states = (data_file.num_features*data_file.num_lanes*4)*data_file.num_veh
#         self.num_obs = data_file.num_features
#         self.num_actions = data_file.num_veh + data_file.num_phases
#         self.noise_std_dev = noise_std_dev

#         self.ou_noise = OUActionNoise(mean=np.zeros(self.num_actions), std_deviation=float(self.noise_std_dev) * np.ones(self.num_actions))
#         self.model = NNModel(64)#.to(device) 64 should be inside the ()
#         self.target_model = NNModel(64)#.to(device)  64 should be inside the ()


#         self.target_model.load_state_dict(self.model.state_dict())

#         self.critic_lr = cri_lr
#         self.actor_lr = act_lr

#         self.critic_optimizer = optim.Adam(self.model.critic_model.parameters(), lr=self.critic_lr)
#         self.actor_optimizer = optim.Adam(self.model.actor_model.parameters(), lr=self.actor_lr)
       
#         self.gamma = disc_factor
#         self.tau = polyak_factor
#         self.buff_size = buff_size
#         self.samp_size = samp_size
#         self.buffer = Buffer(buffer_capacity=self.buff_size, batch_size=self.samp_size, state_size=self.num_states, observe_size=self.num_obs, action_size=self.num_actions, buff_model=self.model, buff_model_target=self.target_model, gamma=self.gamma, tau=self.tau, critic_optimizer=self.critic_optimizer, actor_optimizer=self.actor_optimizer)

#     def update_target(self, target_model, model, tau):
#         for target_param, param in zip(target_model.parameters(), model.parameters()):
#             target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

#     def sigmoid_func(self, np_array):
#         temp_array = []
#         for t in np_array:
#             trunc_val = float_precision(t, 8)   #np.float128(t)
#             if trunc_val< -700:
#                 temp_array.append(0)
#             else: #
#                 temp_array.append(1 / (1 + np.exp(- trunc_val)))
#             #temp_array.append(np.exp(trunc_val) / (1 + np.exp(trunc_val)))
#             # print(temp_array)
#             # exit
#             #assert temp_array[-1]<0 and temp_array[-1]>0,f'ori value:{t},trun val :{trunc_val}, sig:{1 / (1 + np.exp(- trunc_val))}'
#         return np.asarray(temp_array)

#     def policy(self, state, observation, noise_object, num_veh=None):
#         state_has_nan = np.isnan(state).any()
#         assert not state_has_nan, f'input values-bad : state: {state}, obs: {observation}'
#         state = torch.tensor(state, dtype=torch.float32)#.to(device)
#         self.model.eval()
#         with torch.no_grad():
#             sampled_actions = self.model(state, act_or_cri='act').squeeze().cpu().numpy()
#         if noise_object is not None:
#             noise = noise_object()
#             sampled_actions = sampled_actions + noise
#         legal_action = self.sigmoid_func(sampled_actions[data_file.num_phases: (data_file.num_veh + data_file.num_phases)])
#         legal_action_ph = sampled_actions[:data_file.num_phases]
#         legal_action_set = np.append(legal_action_ph, legal_action)
#         assert all([_ >= 0 and _ <= 1 for _ in legal_action_set[data_file.num_phases:(data_file.num_veh + data_file.num_phases)]]), f'alpha value not in range :{legal_action_set[data_file.num_phases:(data_file.num_veh + data_file.num_phases)]}, signal:{legal_action_set[:data_file.num_phases]}'
#         return [np.squeeze(legal_action_set)]
