import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import sys
#from numba import jit, cuda 
import time
import copy
import numpy as np
from collections import deque
import random
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import pickle
import time
import data_file
import vehicle
import safe_set_map_bisect
import functions
import set_class
import vehicle
import contr_sig_rl_new as contr_sig_rl
import delete_data as delete_data

from multiprocessing import Pool
import logging


def func(_args):

	train_iter = _args[0] # If this value is -1 learning_flag set to zero
	sim = _args[1]     # Argument for calling RL agent, default zero
	arr_rate_array = _args[2]
	arr_rate_ = _args[3]
	train_sim = _args[2]    
 
	# print('arr_rate_array',arr_rate_array)
	# exit()

	algo_option = data_file.algo_option # "comb_opt"
	capture_snapshot_flag = 0
	learning_flag = 0
	max_rep_sim = 1
	if data_file.rl_flag:

		# import tensorflow as tf
		import torch
        ##### Selection of  ddpg and MADDPG   ######
		if data_file.rl_algo_opt == "DDPG":
			from ddpg_torch_with_lenghts import DDPG as Agent    #DDPG_model_seq_v_one import DDPG as Agent
		elif data_file.rl_algo_opt == "MADDPG":
			from maddpg import DDPG as Agent

		wr_coun = 0
		algo_option = "rl_modified_ddswa"

		### algorithm option if data_file.rl_flag is 1 ###
		if train_iter == -1:    # For input "--train" loop 
			learning_flag = 1  
		else:
			learning_flag = 0

		### flag to switch between stream learning and snapshot learning ####
		stream_rl_flag = 1
		### flag to switch between stream learning and snapshot learning ####

		#sim = 1
		ss = [data_file.buff_size, 64] # buffer size and sample size
		actor_lr = 0.0001
		critic_lr = 0.001
		p_factor = 0.0001
		d_factor = 0
		agent = None 

		#### RL agent object creation ####
		if data_file.rl_algo_opt == "DDPG":
			if algo_option == "rl_modified_ddswa":
				agent = Agent(sim, samp_size=ss[1], buff_size=ss[0], act_lr=actor_lr, cri_lr=critic_lr, polyak_factor=p_factor, disc_factor=d_factor)
    
				if learning_flag:          # for input "--train" loop 
					curr_state = None
					prev_state = None

					curr_act = None
					prev_act = None

					curr_rew = None
					prev_rew = None

					curr_obs = None
					prev_obs = None
############################################## Should be 61 ###################################################
					max_rep_sim =100 # for CML 61 episodes of data, each of 500 sec [data collection phase]
###############################################################################################################
			elif algo_option == "rl_ddswa":
				agent = Agent(algo_opt=algo_option, state_size=data_file.num_features*data_file.lane_max, action_size=2*data_file.lane_max)
		elif data_file.rl_algo_opt == "MADDPG": pass

		## load trained model
		if (not learning_flag):
			# agent.model.actor_model.load_weights(f"../data/merged_replay_buffer_with_next_state/train_sim_{train_sim}/trained_weights/actor_weights_itr_{train_iter}")
################################################################ Should be from merge ######################################			
			agent.model.load_state_dict(torch.load(f"./data/train_homo_stream/train_sim_1/sim_data/trained_weights/actor_weights_itr_400.weights.pth"))
########################################################################################################################################   
	############## streamline ######################
	if not data_file.run_coord_on_captured_snap:

		for rep_sim in range(0, max_rep_sim): # through DDPG it will 61 episodes
			if train_iter==-1:
				arr_rate_=random.choice(data_file.arr_rates_to_simulate)
				arr_rate_array = {0:0, 1:arr_rate_, 2:arr_rate_, 3:0, 4:arr_rate_, 5:arr_rate_, 6:0, 7:arr_rate_, 8:arr_rate_, 9:0, 10:arr_rate_, 11:arr_rate_} #arr_rate*np.ones(len(lanes)) #
			print(arr_rate_array)
			# exit()
			#logging.info('arr_rate:{} , episode:{}'.format(arr_rate_array[1],rep_sim))
			time_track = 0  
			cumulative_throuput_count = 0  # cumulative throuput
			throughput_id_in_lane = [0 for _ in data_file.lanes] # variable to help track last vehicle in each lane, a list of zeros, change to 1 if particular one crosses the intersection
			sim_obj = set_class.sets() 
		
			if not data_file.real_time_spawning_flag:  #default 1 #NOT real-time spawning
				file = open(f"./data/compare_files/homogeneous_traffic/arr_{arr_rate_}/sim_obj_num_{sim}", 'rb')
				sim_obj = pickle.load(file)
				file.close()
				total_veh_in_simulation = functions.get_num_of_objects(sim_obj.unspawned_veh) 
			if data_file.real_time_spawning_flag:
				#############lane independent ID ###############
				veh_id_var = 0
				#############lane dependent ID ###############
				dep_veh_id  = [(100*lane) for lane in data_file.lanes] 

				next_spawn_time = [100 + data_file.max_sim_time for _ in data_file.lanes]  
				for lane in data_file.lanes:
					if not (arr_rate_array[lane] == 0):   #if only the arrival rate not zero we add poisson spawning else the previous values not changed
						next_spawn_time[lane] = round(time_track + float(np.random.exponential(1/arr_rate_array[lane],1)),1)


			assert wr_coun ==0, 'replay buffer not updated'

			horizon = {}
			override_lane =  {}
			lane_sig_stat = {0:['R',0], 1:['R',0], 2:['R',0], 3:['R',0], 4:['R',0], 5:['R',0], 6:['R',0], 7:['R',0], 8:['R',0], 9:['R',0], 10:['R',0], 11:['R',0]}
			dict_sig = {1:'R',2:'R',4:'R',5:'R',7:'R',8:'R',10:'R',11:'R'}    
			### override_lane, key - lane, value - vehicle ID of bad agent   ###
			### lane signal and time of previous change  ###

			### start of simulation ###
			while (time_track < data_file.max_sim_time):  
				curr_time = time.time()
				if (not data_file.real_time_spawning_flag):
					if (functions.get_num_of_objects(sim_obj.done_veh) >= total_veh_in_simulation):
						break
				if data_file.rl_algo_opt == "DDPG" and learning_flag and (agent.buffer.buffer_counter > 0) and ((time_track % 1) == 0):
					#print(f'critic:{agent.target_model.variables},actor:{agent.target_model.variables}')
					#exit()
					agent.buffer.learn()

					#print(f'***update:{agent.target_model.variables}')

					#print(f'***weights:{type(agent.target_model.get_weights())},')

					###agent.update_target(agent.target_model.get_weights(), agent.model.get_weights(), agent.tau_)
					agent.update_target(agent.target_model.actor_model, agent.model.actor_model, agent.tau)
					agent.update_target(agent.target_model.critic_model, agent.model.critic_model, agent.tau)



					#agent.update_target(agent.target_model.critic_model.variables, agent.model.critic_model.variables, agent.tau_)
					#agent.update_target(agent.target_model.actor_model.variables, agent.target_model.actor_model.variables, agent.tau_)


					#agent.update_target(agent.target_critic_.variables, agent.critic_model_.variables, agent.tau_)

					#agent.update_target(agent.target_actor_.variables, agent.actor_model_.variables, agent.tau_)
					#agent.update_target(agent.target_critic_.variables, agent.critic_model_.variables, agent.tau_)
				if learning_flag and (time_track % 100) == 0:
					print('uyflf')
					# directory = f"./data/arr_{arr_rate_}/train_homo_stream/train_sim_{sim}/sim_data/trained_weights/"
					directory = f"./data/train_homo_stream/train_sim_{sim}/sim_data/trained_weights/"
					os.makedirs(directory, exist_ok=True)
					filename = f"actor_weights_itr_{int(time_track)}.weights.pth" ## .h5 to .pth
					file_path = os.path.join(directory, filename)
					# agent.model.save_weights(file_path)
					torch.save(agent.model.state_dict(), file_path)


					#agent.model.save_weights(f"../data/arr_{arr_rate_}/train_homo_stream/train_sim_{sim}/sim_data/trained_weights/actor_weights_itr_{int(time_track)}")

				for lane in data_file.lanes: 
					if (data_file.real_time_spawning_flag) and (round(time_track, 1) >= round(next_spawn_time[lane], 1)) and (len(sim_obj.unspawned_veh[lane]) == 0) and (not (arr_rate_array[lane] == 0)):
						#next_spawn_time[lane] = round(time_track + float(np.random.exponential(1/arr_rate_array[lane],1)),1)  #NEW SPAWN TIME FOR NEXT ROBOT
						next_spawn_time[lane] = round(time_track + float(np.random.exponential(1/arr_rate_array[lane],1)[0]), 1)
						new_veh = vehicle.Vehicle(lane, data_file.int_start[lane], 0, data_file.vm[lane], data_file.u_min[lane], data_file.u_max[lane], data_file.L, arr_rate_array)
						

						
						############### veh.ID ####################
						## for lane dependent ID
						#new_veh.id = copy.deepcopy(dep_veh_id[lane])
						#dep_veh_id[lane] += lane  #making the lane dependent ID
						# continuous ID
						#print(type(new_veh.curr_set))
						#exit()
						new_veh.id = copy.deepcopy(veh_id_var)
						veh_id_var += 1
						################ veh.ID #######################
						new_veh.sp_t = copy.deepcopy(time_track)  # time at whic robot veh created 
						new_veh.sig_stat_t = lane_sig_stat[lane][1] 
						new_veh.sig_stat = 0
						#assert new_veh.sig_stat == 0 or new_veh.sig_stat_t  == 1,' error in assigning signal'
						#new_veh.curr_set[time_track] ='un'
						sim_obj.unspawned_veh[lane].append(copy.deepcopy(new_veh))
			    	#################### end of spawning  ################### 	
			
					##### rear end safety validation  ##### 	
					n = len(sim_obj.unspawned_veh[lane])
					########### CHECKING MODULE ###########
					functions.seq_checker(sim_obj.unspawned_veh)
					functions.seq_checker(sim_obj.spawned_veh)
					########### CHECKING MODULE ###########
					v_itr = 0
					if n>1:print(a)

					while v_itr<n:
						v = sim_obj.unspawned_veh[lane][v_itr]
						pre_v = None
						if len(sim_obj.spawned_veh[lane]) > 0:       
							pre_v = sim_obj.spawned_veh[lane][-1]
						if (round(v.sp_t,1) < round(time_track,1)) and (functions.check_init_config(v, pre_v, time_track)): # CHECKS REAR END SAFETY
							v.sp_t = round(time_track,1)	# step at whcih the particular vehicle spawned or entred ROI
							sim_obj.unspawned_veh[lane].popleft()
							#v.curr_set[time_track] ='sp' 
							sim_obj.spawned_veh[lane].append(v)
							n = len(sim_obj.unspawned_veh[lane])
						else:
							break	
				
				########### CHECKING MODULE ###########
				functions.seq_checker(sim_obj.unspawned_veh)
				functions.seq_checker(sim_obj.spawned_veh)
				for _ in data_file.lanes: assert (sorted(list(set([v.id for v in (sim_obj.spawned_veh[_])]))) ==  [v.id for  v in (sim_obj.spawned_veh[_])]), f'ID:{[v.id for  v in (sim_obj.spawned_veh[_])]}, lane:{_},1:{sort(list(set([v.id for v in (sim_obj.spawned_veh[_])])))}, 2:{[[v.id,iter] for  iter, v in enumerate(sim_obj.spawned_veh[_])]}'   #### check all id in a lane in ascending order
				#print(f'ID:{[v.id for  v in (sim_obj.spawned_veh[_])]}, lane:{_},1:{list(set([v.id for v in (sim_obj.spawned_veh[_])]))}, 2:{[v.id for  v in (sim_obj.spawned_veh[_])]} ')
				
				########### CHECKING MODULE ###########
				##### END - spawning & rear end safety validation ##### 
	

				############# update veh and REF dict with signal and time ##################
				#print(f'dic_sig:{dict_sig}, lane_ref_dict:{lane_sig_stat}')
				for lane in data_file.lanes:
					if len(data_file.incompdict[lane])>0:
						if lane_sig_stat[lane][0] !=  dict_sig[lane]:
							lane_sig_stat[lane][0] =  dict_sig[lane]
							lane_sig_stat[lane][1] =  time_track
						n = len(sim_obj.spawned_veh[lane])
						if dict_sig[lane]=='G': sig = 1 
						elif dict_sig[lane]=='R': sig = 0
						for iter in range(n):
							sim_obj.spawned_veh[lane][iter].sig_stat = sig
							sim_obj.spawned_veh[lane][iter].sig_stat_t = lane_sig_stat[lane][1] 
							sim_obj.spawned_veh[lane][iter].global_sig[time_track] = dict_sig
				############# update veh and REF dict  with signal and time  ##################
					

				#print("**********model_summary*********")
				#print(agent.model.summary())
				#print(agent.model.actor_model.summary())
				#print(agent.model.critic_model.summary())
				#print("**********model_summary*********")

				if functions.get_num_of_objects(sim_obj.spawned_veh) > 0 :
					spawned_veh_copy = copy.deepcopy(sim_obj.spawned_veh)
					### RL decision ###
					if learning_flag:
						prev_state = curr_state
						prev_rew = curr_rew
						prev_obs = curr_obs
					sim_obj.spawned_veh, alpha, dict_alpha, signal, dict_sig, state_t, obs_t, action_t, explore_flag = contr_sig_rl.get_alpha_sig(time_track, sim_obj.spawned_veh, agent, algo_option, learning_flag)
					if learning_flag:
						curr_state = copy.deepcopy(state_t)
						prev_act = copy.deepcopy(curr_act)
						curr_act = copy.deepcopy(action_t)
						curr_obs = copy.deepcopy(obs_t)
						curr_rew = 0
					### END - RL decision ###
					#print(agent.model.actor_model.model_summary())



					########### CHECKING MODULE ###########
					functions.seq_checker(sim_obj.spawned_veh)
					if data_file.output =='Signal': 
						assert len(dict_sig) == data_file.num_lanes*4
					elif data_file.output =='Phase': pass
						#assert len(dict_sig) == data_file.num_phases,f'dict:{len(dict_sig)},phase:{data_file.num_phases}'

					assert all ([ v.id == spawned_veh_copy[_][iter].id for _ in data_file.lanes for iter, v in enumerate(sim_obj.spawned_veh[_]) ]),f'spawned set passed from RL WITH ERROR'
					assert all([ _!= None for _ in signal ]), f'signal value is none:{signal}, time:{time_track}'
					assert all([ _!= None for _ in alpha ]), f'alpha value is none:{alpha}, time:{time_track}'
					#assert 
					########### CHECKING MODULE ###########
					
					########### Penalty ###########
					if learning_flag and (functions.get_num_of_objects(sim_obj.spawned_veh)>0):
						curr_rew -= sum([v.priority for _ in data_file.lanes for v in sim_obj.spawned_veh[_]])
						# print('curr_rew',curr_rew)
					else: curr_rew = 0	
					########### Penalty ###########

					#if tprint(f'buf state:{prev_state},{type(prev_state)},len:{len(prev_state)}')
					#exit()

					###### override for a fixed time period T  #############
					###### T time period for bad agent to cross intersection #####
					#print(f' override_lane:{override_lane}')
					for _ in override_lane: 
						#print("fgdgg",_)
						#exit()
						dict_sig[_]='R'
						for lane in data_file.incompdict[_]:
							#print(f'dict:{dict_sig}')
							dict_sig[lane]='R'
					######### override log dictionary ##############

				################## update the veh with signal values #################
				for lane in data_file.lanes:
					n = len(sim_obj.spawned_veh[lane])
					for iter in range(n):
						sim_obj.spawned_veh[lane][iter].global_sig_val[time_track] = signal
						sim_obj.spawned_veh[lane][iter].ovr_stat[time_track] = override_lane # list(override_lane.keys())
						#print(f'sig_values:{(sim_obj.spawned_veh[lane][iter].global_sig_val[time_track])}')

				###### override for a fixed time period T  #############
					
				#if time_track == 25: exit()




				###############################################

				########### control variable estimation ##############
				if functions.get_num_of_objects(sim_obj.spawned_veh) > 0 : 
					for lane in data_file.lanes:
						green_zone = -1*data_file.vm[lane]*data_file.dt + (data_file.vm[lane]**2)/(2*(max(data_file.u_min[lane],-(data_file.vm[lane]/data_file.dt))))  
						if len(data_file.incompdict[lane])>0:
							override_veh = []
							if dict_sig[lane]=='G':
								n = len(sim_obj.spawned_veh[lane])
								for iter in range(n):
									pre_v = None
									success = None
									v = copy.deepcopy(sim_obj.spawned_veh[lane][iter])
									if n >1 and iter >0: pre_v = copy.deepcopy(sim_obj.spawned_veh[lane][iter-1])
									sim_obj.spawned_veh[lane][iter], success = safe_set_map_bisect.green_map(v, pre_v, time_track)#, algo_option, learning_flag, 1, sim, train_iter, train_sim)
									if not learning_flag:functions.storedata(v, train_sim, sim, train_iter) 							
							elif dict_sig[lane]=='R':
								n = len(sim_obj.spawned_veh[lane])
								for iter in range(n):
									v = copy.deepcopy(sim_obj.spawned_veh[lane][iter])
									success = None
									pre_v = None
									if n >1 and iter >0 : pre_v = copy.deepcopy(sim_obj.spawned_veh[lane][iter-1])
									sim_obj.spawned_veh[lane][iter], success = safe_set_map_bisect.red_map(v, pre_v,time_track)# algo_option, learning_flag, 1, sim, train_iter, train_sim)
									if success == False: ##OVERRIDE
										sim_obj.spawned_veh[lane][iter], success = safe_set_map_bisect.green_map(v, pre_v,time_track)# algo_option, learning_flag, 1, sim, train_iter, train_sim)
										override_veh.append(v.id)
									if not learning_flag:functions.storedata(v, train_sim, sim, train_iter) 
								if len(override_veh)>0: override_lane[lane] = override_veh
								#if not learning_flag:functions.storedata(v, train_sim, sim, train_iter) 

					assert len(set(override_lane.keys())) == len(override_lane), f'duplicates present in over_ride lane: {override_lane}'

					#assert all([list(override_lane.keys())[_] not in override_lane  for _ in range(len(override_lane))]),f'duplicates present :{override_lane}'

					############# all incompatible lane_override trajectories #############
	 				######################################################################
	 				##### Note ######### : bad agent shouldn't get this trajectory ########
	  				#######################################################################
					for _ in override_lane:
						print(f'')
						dict_sig[_]='R'
						for lane in data_file.incompdict[_]:
							dict_sig[lane]='R'   #### override
							n = len(sim_obj.spawned_veh[lane])
							for iter in range(n):
								v = copy.deepcopy(sim_obj.spawned_veh[lane][iter])
								success = None
								pre_v = None
								if n >1 and iter >0: pre_v = copy.deepcopy(sim_obj.spawned_veh[lane][iter-1])
								sim_obj.spawned_veh[lane][iter], success= safe_set_map_bisect.red_map(v, pre_v,time_track) 
								assert success != False
								if not learning_flag:functions.storedata(v, train_sim, sim, train_iter) 
						############# lane_override trajectories #############
						#for iter in range(len(sim_obj.spawned_veh[_])):
						#		#print(sim_obj.spawned_veh[lane][iter].id, override_lane[_])
						#		if sim_obj.spawned_veh[_][iter].id not in override_lane[_]:
						#			v = copy.deepcopy(sim_obj.spawned_veh[_][iter])
						#			success = None
						#			pre_v = None
						#			if n >1 and iter >0: pre_v = copy.deepcopy(sim_obj.spawned_veh[_][iter-1])
						#			sim_obj.spawned_veh[_][iter], success= safe_set_map.red_map(v, pre_v,time_track) 
						#			if not learning_flag:functions.storedata(v, train_sim, sim, train_iter) 
				########### all incompatible control variable estimation ##############

				""" 				
				print("After control traj generation")
				for lle in data_file.lanes:  
					print("lane no:",lle,"\n")
					for _iter in range(len(sim_obj.spawned_veh[lle])): print("pos:",sim_obj.spawned_veh[lle][_iter].p_traj, "Vel:",sim_obj.spawned_veh[lle][_iter].v_traj,"u:",sim_obj.spawned_veh[lle][_iter].u_traj , "Id",sim_obj.spawned_veh[lle][_iter].id, end=" ")
				print(f'overide:{override_lane}, signal"{dict_sig}')

 				"""
				#####################################    
    			#if not learn_flag:
	 			#   functions.storedata(veh, tr_sim_num, sim_num, train_iter_num) 			
				#######################################
								

				### update current time###``
				time_track = round((time_track + data_file.dt), 1)
				if learning_flag:
					print(f"arr_rate: {arr_rate_}, rep: {rep_sim}", "current time:", time_track, "sim:", sim, "train_iter:", train_iter,"size_buff",{sys.getsizeof(agent.buffer)},"......", end="\r")
				else:
					print("arr_rate:", arr_rate_,"current time:", time_track, "sim:", sim, "train_sim: ", train_sim, "train_iter:", train_iter, "arr_rate: ", arr_rate_,"********", end="\r")# "heuristic:", data_file.used_heuristic, "................", end="\r")
				### update current time###

				### throuput calculation ###
				for l in data_file.lanes:
					for v in sim_obj.spawned_veh[l]:
						t_ind = functions.find_index(v, time_track)

						if  ((t_ind == None) or (v.p_traj[t_ind] > (v.intsize + data_file.L))) and (v.id >= throughput_id_in_lane[l]):

							throughput_id_in_lane[l] = copy.deepcopy(v.id)
							cumulative_throuput_count += 1
						else:
							break
				### throuput calculation ###

				### removing vehicles which have crossed the region of interest ###
				for l in data_file.lanes:
					n_in_green = len(sim_obj.spawned_veh[l])
					#print("override", override_lane,l)
					
					if  l in override_lane: c = len(override_lane[l])
					else: c = 0
					v_ind = 0
					while v_ind < n_in_green:
						green_while_flag = 0
						v = sim_obj.spawned_veh[l][v_ind]
						t_ind = functions.find_index(v, time_track)
						if (t_ind == None) or (v.p_traj[t_ind] > (v.intsize + v.length) ):   #- v.int_start)):
							#### if agent crosses remove override #####
							if l in override_lane and v.id in override_lane[l]: c -= 1
							curr_rew += 10*v.priority
							## sum([v.priority for _ in data_file.lanes for v in sim_obj.spawned_veh[_]])
							horizon[v.id] = time_track - v.sp_t 
							sim_obj.done_veh[l].append(v)
							sim_obj.spawned_veh[l].popleft()
							n_in_green -= 1
						else:
							break
					if c==0 and l in override_lane: del override_lane[l] 

	


				functions.seq_checker(sim_obj.done_veh)

							#if t_ind != None: print("done pos", time_track,v.p_traj,v.t_ser, v.id)




				###### traj for override set for future lanes ########
				#### code here				
				###### traj for override set for future lanes ########





				""" 
				print("list-after-done")
				for lle in data_file.lanes:  
					print("lane no:",lle,"\n")
					for _iter in range(len(sim_obj.done_veh[lle])): print("done ID-index",sim_obj.done_veh[lle][_iter].p_traj,sim_obj.done_veh[lle][_iter].id,end=" ")
 				"""
				### storing data in buffer
				if (algo_option == "rl_modified_ddswa") and (learning_flag):
					if (not (prev_rew == None)) and (not (len(prev_state) == 0)):

						#print(f'buf state:{prev_state},{type(prev_state)},len:{len(prev_state)}')
						#print(f'buf action:{prev_act},{type(prev_act)},len:{len(prev_act)}')
						#print("state in main:",prev_state, len(prev_state[0]), {curr_state}, {len(curr_state)})#,"obs in main:",prev_obs, len(prev_obs[0]),"rew",prev_rew )
						#print("action in main:",prev_act, len(prev_act[0]))
						agent.buffer.remember((prev_state[0], prev_obs,prev_act, prev_rew, curr_state[0], curr_obs))
						qwe = {}
						qwe["state_buffer"] = agent.buffer.state_buffer
						qwe["observe_buffer"] = agent.buffer.observe_buffer
						qwe["action_buffer"] = agent.buffer.action_buffer
						qwe["reward_buffer"] = agent.buffer.reward_buffer
						qwe["next_state_buffer"] = agent.buffer.next_state_buffer
						qwe["next_observe_buffer"] = agent.buffer.next_observe_buffer

						if rep_sim%5==0 and time_track == data_file.max_sim_time :
							# dbfile = open(f'./data/arr_{arr_rate_}/train_homo_stream/train_sim_{sim}/replay_buffer_sim_{sim}', 'wb')
							dbfile = open(f'./data/train_homo_stream/train_sim_{sim}/replay_buffer_sim_{sim}', 'wb')
							pickle.dump(qwe, dbfile)
							dbfile.close()
							wr_coun =0
				### removed vehicles which have crossed the region of interest ###
						
				##############################



			### end of simulation ###
		
			
    #### streamline ####






if __name__ == '__main__':

	
	with open('train.log', 'w'):	pass # empty loffer for append mode

	arr_rates_to_sim = data_file.arr_rates_to_simulate  #The 10 diff values from 0.01 to 0.1
##########################################################################################################################
	arr_rates_to_sim=[0.1]
###################################################################################################
	args = []


	if data_file.used_heuristic == None:
	
		if data_file.rl_flag:

			train_or_test = str(sys.argv[1])

			if train_or_test == "--train":
				##### for cluster #########
				_arr_rate_ = float(sys.argv[4]) 
				##### for cluster #########
				logging.basicConfig(filename="train.log", format='%(asctime)s %(message)s',filemode='a')
				logger = logging.getLogger()
				logger.setLevel(logging.DEBUG)
				for _train_iter in range(1):
					for _sim_num in range(1, 2):
				# 		#for _arr_rate_ in arr_rates_to_sim:
						arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_} #arr_rate*np.ones(len(lanes)) #
						args.append([-1, _sim_num, arr_rate_array_, _arr_rate_, 0])
				# 			# func(args[-1])
				# for _sim_num in range(1, 2):
					# args.append([-1,_sim_num,0]) # [train_iter,sim,train_sim]
				#try:			
				# pool = Pool(5)
				# pool.map(func, args)
				func(args[0])
				#except Exception as e:
				#	print("Error:", e, file=sys.stderr)
				#	sys.exit(1)  # Exit with non-zero

 
			elif train_or_test == "--test":

				if not data_file.run_coord_on_captured_snap:

					_train_iter_list = [int(sys.argv[2])]

					for _train_iter in _train_iter_list:
						for _sim_num in range(1, 11):  #11
							############################### edited ################ # each policy run at speicific arrival rate for 10 times to increase the samples.
							for _train_sim in list(range(1, 2)):   ##### edited############### 11 $#############################
								for _arr_rate_ in arr_rates_to_sim:
									arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_} #arr_rate*np.ones(len(lanes)) #
									
									# file_path = f"../data/arr_{_arr_rate_}/test_homo_stream/train_sim_{_train_sim}/train_iter_{_train_iter}"								

									file_path = f"./data/test_homo_stream/train_sim_{_train_sim}/train_iter_{_train_iter}/sim_{_sim_num}/sim_{_sim_num}_train_iter_{_train_iter}.png"

									try:
										with open(f"{file_path}") as f:
											f.close()

									except:
										# if len(list(os.listdir(f"{file_path}/pickobj_sim_{_sim_num}"))) == 0:
										args.append([_train_iter, _sim_num, arr_rate_array_, _arr_rate_, _train_sim])
										#print(f"train_sim: {_train_sim}, train_iter: {_train_iter}, sim: {_sim_num}")

										# args.append([_train_iter, _sim_num, arr_rate_array_, _arr_rate_])
										# func(args[-1])
					# # pool = Pool(18)
					# pool = Pool(5)
					# pool.map(func, args)

				else:
					_arr_rate_ = 0.08
					for _sim_num in range(1,4):
						args.append([5000,_sim_num,0,_arr_rate_,8])
						func(args[-1])				




		elif not data_file.run_coord_on_captured_snap:
			for _arr_rate_ in arr_rates_to_sim:
				arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_} #arr_rate*np.ones(len(lanes)) #
				for _sim_num in range(1, 101):
					args.append([0, _sim_num, arr_rate_array_, _arr_rate_, 0])
					func(args[-1])


		else:
			_arr_rate_ = 0.08
			for _sim_num in range(1,4):
				args.append([0,_sim_num,0,_arr_rate_,0])
				func(args[-1])


	else:

		for _train_iter in [0]:
			for _sim_num in range(1, 101): # 100 diff simulations
				for _train_sim in list(range(1)):
					for _arr_rate_ in arr_rates_to_sim:

						arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_} #arr_rate*np.ones(len(lanes)) #
						heuristics_pickobj_save_path = f"./data/{data_file.used_heuristic}/arr_{_arr_rate_}/pickobj_sim_{_sim_num}"

						# if len(os.listdir(f"{heuristics_pickobj_save_path}")) < 290:
						args.append([_train_iter, _sim_num, arr_rate_array_, _arr_rate_, _train_sim])
						
						# else:
						# 	...

						# print(f"train_sim: {_train_sim}, train_iter: {_train_iter}, sim: {_sim_num}")

		pool = Pool(18)
		pool.map(func, args)








