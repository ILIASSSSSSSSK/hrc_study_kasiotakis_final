#!/usr/bin/env python3.6
import rospy
import torch
from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import Twist
from cartesian_state_msgs.msg import PoseTwist
from human_robot_collaborative_learning.srv import *
from human_robot_collaborative_learning.msg import Score
from utils import *
from sensor_msgs.msg import JointState
import math
import numpy as np
import pandas as pd
from scipy.spatial import distance
from tqdm import tqdm
from pydub import AudioSegment
from pydub.playback import play
import threading
import curses
import random
import time
import matplotlib.pyplot as plt
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelConfiguration
from controller_manager_msgs.srv import SwitchController
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelState
from datetime import datetime
import os

class RL_Control:
	def __init__(self):
		self.initialized_agent = rospy.get_param("/rl_control/Game/initialized_agent",False)
		self.train_model = rospy.get_param('rl_control/Game/train_model', False)
		self.transfer_learning = rospy.get_param("rl_control/Game/load_model_transfer_learning", False)
		self.num_of_states=rospy.get_param("rl_control/Game/number_of_states",4)
		self.normalized=rospy.get_param("rl_control/Experiment/normalized",False)
		self.greedy_test = rospy.get_param("rl_control/Experiment/greedy",False)
		
		self.max_vel_x = rospy.get_param("rl_control/Experiment/ee_vel_x_max",0.2)
		self.min_vel_x = rospy.get_param("rl_control/Experiment/ee_vel_x_min",-0.2)
		self.max_vel_y = rospy.get_param("rl_control/Experiment/ee_vel_y_max",0.2)
		self.min_vel_y = rospy.get_param("rl_control/Experiment/ee_vel_y_min",0.2)
		
		self.min_x=rospy.get_param("robot_movement_generation/min_x",-0.356)
		self.min_y=rospy.get_param("robot_movement_generation/min_y",0.162)
		self.max_x=rospy.get_param("robot_movement_generation/max_x",-0.174)
		self.max_y=rospy.get_param("robot_movement_generation/max_y",0.343)

		# PATHS
		self.full_path = rospy.get_param("/rl_control/Game/full_path", "")
		self.load_model_for_training_dir = os.path.join(self.full_path, rospy.get_param("rl_control/Game/load_model_training_dir", ""))
		self.load_ppr_model_dir = os.path.join(self.full_path, rospy.get_param("rl_control/Game/load_ppr_model_dir", ""))
		self.load_model_testing_dir = os.path.join(self.full_path, rospy.get_param("rl_control/Game/load_model_testing_dir", ""))
		self.initialized_agent_dir = os.path.join(self.full_path, rospy.get_param("rl_control/Game/initialized_agent_dir", ""))
		
		if self.train_model:
			self.load_model_for_training = rospy.get_param("rl_control/Game/load_model_training", False)

			if self.load_model_for_training:
				self.agent = get_SAC_agent(observation_space=[self.num_of_states], chkpt_dir=self.load_model_for_training_dir)
				self.agent.load_models()
				rospy.logwarn("Successfully loaded model at {} for training".format(self.load_model_for_training_dir))
			else:
				if self.initialized_agent:
					self.agent = get_SAC_agent(observation_space=[self.num_of_states])
					self.agent.load_baseline_models(self.initialized_agent_dir)
					rospy.logwarn("Successfully loaded model at {} for initialization".format(self.initialized_agent_dir))
					rospy.set_param("/rl_control/Game/initialized_agent",False)
				else:
					self.agent = get_SAC_agent(observation_space=[self.num_of_states])
					rospy.logwarn("User has not specified any model for training. Gonna initialize random agent")
			# PPR loading
			if self.transfer_learning:
				self.expert_agent = get_SAC_agent(observation_space=[self.num_of_states], chkpt_dir=load_ppr_model_dir)
				self.expert_agent.load_models()
				self.ppr_threshold = rospy.get_param("rl_control/Game/ppr_threshold", 0.7)
				rospy.logwarn('Successfully loaded model at {} for transfer learning'.format(load_ppr_model_dir))	
			else:
				rospy.logwarn("User has not loaded any models for transfer learning")
		else:
			rospy.logwarn('User is testing the model {}'.format(self.load_model_for_testing_dir))
			self.agent = get_SAC_agent(observation_space=[self.num_of_states], chkpt_dir=self.load_model_for_testing_dir)
			self.agent.load_models()
			rospy.logwarn("Successfully loaded model at {} for testing".format(self.load_model_for_testing_dir))

		# Game parameters		
		self.goal = rospy.get_param('rl_control/Game/goal', [0, 0])
		self.goal_dis = rospy.get_param('rl_control/Game/goal_distance', 2)
		self.goal_vel = rospy.get_param('rl_control/Game/goal_velocity', 2)
		self.action_duration = rospy.get_param('rl_control/Experiment/action_duration', 0.1)
		self.test_count = 0

		# AUDIO FILES
		audio_dir = os.path.join(self.full_path, 'audio_files')
		start_audio_files = rospy.get_param('rl_control/Game/start_audio', ['', ''])
		self.start_audio = [AudioSegment.from_mp3(os.path.join(audio_dir, file)) for file in start_audio_files]
		win_audio_file = rospy.get_param('rl_control/Game/win_audio', '')
		lose_audio_file = rospy.get_param('rl_control/Game/lose_audio', '')
		self.win_audio = AudioSegment.from_mp3(os.path.join(audio_dir, win_audio_file))
		self.lose_audio = AudioSegment.from_mp3(os.path.join(audio_dir, lose_audio_file))
		
		# Experiment parameters for training
		self.max_episodes = rospy.get_param('rl_control/Experiment/max_episodes', 1000)
		self.num_blocks = rospy.get_param('rl_control/Experiment/num_blocks', 20)
		self.max_timesteps = int(rospy.get_param('rl_control/Experiment/max_duration', 200)/self.action_duration)
		self.start_training_on_episode = rospy.get_param('rl_control/Experiment/start_training_on_episode', 10)
		self.total_update_cycles = rospy.get_param('rl_control/Experiment/total_update_cycles', 10)
		self.randomness_threshold = rospy.get_param('rl_control/Experiment/stop_random_agent', 10)
		self.scheduling = rospy.get_param('rl_control/Experiment/scheduling', 'uniform')
		self.update_cycles = self.total_update_cycles
		self.best_episode_reward = -100 - 1*self.max_timesteps
		self.win_reward = rospy.get_param('rl_control/Experiment/win_reward', 10)
		self.lose_reward = rospy.get_param('rl_control/Experiment/lose_reward', -1)	
		self.reward_history = []
		self.train_timestamps = []
		self.episode_history = []
		self.data_block_history = []
		self.init_pos = []
		self.episode_duration = []
		self.travelled_distance = []
		self.number_of_timesteps = []
		self.column_names=()
		if self.num_of_states==4:
			self.column_names = ("timestamps", "block", "episode", "init_pos", "human_action", "agent_action", "prev_x", "prev_y", "next_x", "next_y", "ee_pos_x_prev", "ee_pos_y_prev", "ee_vel_x_prev","ee_vel_y_prev","ee_pos_x_next", "ee_pos_y_next", "ee_vel_x_next","ee_vel_y_next","cmd_acc_human", "cmd_acc_agent")
		elif self.num_of_states==2:
			self.column_names = ("timestamps", "block", "episode", "init_pos", "human_action", "agent_action", "prev_x", "prev_y", "next_x", "next_y", "ee_pos_x_prev", "ee_pos_y_prev", "ee_pos_x_next", "ee_pos_y_next", "cmd_acc_human", "cmd_acc_agent")
		self.state_info = [self.column_names] 
		self.human_action = None
		self.rest_period = rospy.get_param("rl_control/Game/rest_period", 120)
		self.run_blocks = []
		self.run_steps = []
		self.timestamps = []
		self.run_init_pos = []
		self.human_actions = []
		self.agent_actions = []
		self.ee_pos_x_prev = []
		self.ee_pos_y_prev = []
		self.unnorm_pos_x_prev = []
		self.unnorm_pos_y_prev = []
		self.ee_vel_x_prev = []
		self.ee_vel_y_prev = []
		self.ee_pos_x_next = []
		self.ee_pos_y_next = []
		self.unnorm_pos_x_next = []
		self.unnorm_pos_y_next = []
		self.ee_vel_x_next = []
		self.ee_vel_y_next = []
		self.cmd_acc_x = []
		self.cmd_acc_y = []
		self.expert_action_flag = False
		
		# Experiment parameters for testing
		self.test_max_timesteps = int(rospy.get_param('rl_control/Experiment/test/max_duration', 200)/self.action_duration)
		self.test_max_episodes = rospy.get_param('rl_control/Experiment/test/max_episodes', 1000)
		self.test_interval = rospy.get_param('rl_control/Experiment/test_interval', 10)
		self.test_agent_flag = False
		self.test_best_reward = -100 -1*self.test_max_timesteps
		self.test_reward_history = []
		self.test_timestamps = []
		self.test_data_block_history = []
		self.test_init_pos = []
		self.test_episode_history = []
		self.test_episode_duration = []
		self.test_travelled_distance = []
		self.test_number_of_timesteps = []
		self.test_state_info = [self.column_names] 
		self.ogu_data=[("block", "step", "temp","entropy","entropy_loss","q1_history","q2_history","q1_loss","q2_loss","target_q","policy_loss")]
		
		self.ur3_state_sub = rospy.Subscriber('ur3_cartesian_velocity_controller/ee_state', PoseTwist, self.ee_state_callback)
		if(rospy.get_param("/rl_control/Game/gazebo_simulation",False)):
			self.ur3_state_sub = rospy.Subscriber('ur3_cartesian_velocity_controller_sim/ee_state', PoseTwist, self.ee_state_callback)
			print("I am in simulation")

		self.ur3_velocities_sub = rospy.Subscriber('ur3_cartesian_velocity_controller/command_cart_vel', Twist, self.ee_velocities_callback)
		if(rospy.get_param("/rl_control/Game/gazebo_simulation",False)):
			self.ur3_velocities_sub = rospy.Subscriber('ur3_cartesian_velocity_controller_sim/command_cart_vel', Twist, self.ee_velocities_callback)

		self.human_action_sub = rospy.Subscriber('cmd_vel', Twist, self.human_callback)
		self.agent_action_pub = rospy.Publisher('agent_action_topic', Float64, queue_size=10)
		self.train_pub = rospy.Publisher('train_topic', Bool, queue_size=10)
		self.score_pub = rospy.Publisher('score_topic', Score, queue_size=10)
		self.t_win = threading.Thread(target=play, args=(self.win_audio,))
		self.t_lose = threading.Thread(target=play, args=(self.lose_audio,))
		rospy.sleep(1)

	def load_baseline_weights(self):
		self.agent.load_baseline_models(self.initialized_agent_dir)

	def load_expert_weights(self, expert_number):
		self.agent.load_expert_policy(expert_number)

	def reset(self,position):
		self.timestep = 0
		self.timeout = False
		self.episode_reward = 0
		self.human_actions = []
		self.agent_actions = []
		self.ee_pos_x_prev = []
		self.ee_pos_y_prev = []
		self.unnorm_pos_x_prev = []
		self.unnorm_pos_y_prev = []
		self.ee_vel_y_prev = []
		self.ee_pos_x_next = []
		self.ee_pos_y_next = []
		self.unnorm_pos_x_next = []
		self.unnorm_pos_y_next = []
		self.ee_vel_x_next = []
		self.ee_vel_y_next = []
		self.cmd_acc_x = []
		self.cmd_acc_y = []
		self.run_blocks = []
		self.run_steps = []
		self.timestamps = []
		self.run_init_pos = []

		# Reduce PPR threshold every time we reset the game from "Train" mode
		if self.transfer_learning and not self.test_agent_flag:
			self.ppr_threshold -= 0.01
		rospy.wait_for_service('reset')

		try:
			reset_game = rospy.ServiceProxy('reset', Reset)
			rospy.loginfo('Resetting the game')
			reset_game(position=position)
			rospy.loginfo('Game reset. Start episode')
		except rospy.ServiceException as e:
			rospy.logerr("Service call failed: %s"%e)

	# def reset_from_different_position(self,position):
	# 	self.timestep = 0
	# 	self.timeout = False
	# 	self.episode_reward = 0
	# 	self.human_actions = []
	# 	self.agent_actions = []
	# 	self.ee_pos_x_prev = []
	# 	self.ee_pos_y_prev = []
	# 	self.unnorm_pos_x_prev = []
	# 	self.unnorm_pos_y_prev = []
	# 	self.ee_vel_y_prev = []
	# 	self.ee_pos_x_next = []
	# 	self.ee_pos_y_next = []
	# 	self.unnorm_pos_x_next = []
	# 	self.unnorm_pos_y_next = []
	# 	self.ee_vel_x_next = []
	# 	self.ee_vel_y_next = []
	# 	self.cmd_acc_x = []
	# 	self.cmd_acc_y = []
	# 	self.run_blocks = []
	# 	self.run_steps = []


	# 	# TODO: Link this with CPP file to start from different position
	# 	try:
	# 		reset_game = rospy.ServiceProxy('reset', Reset)
	# 		rospy.loginfo('Resetting the game')
	# 		reset_game(position=position)
	# 		rospy.loginfo('Game reset. Start episode')
	# 	except rospy.ServiceException as e:
	# 		rospy.logerr("Service call failed: %s"%e)


	def e_greedy(self, randomness_request):
		if randomness_request <= self.randomness_threshold:
			# Pure exploration
			if self.rand_int==True:
				self.agent_action = np.random.randint(self.agent.n_actions)
				#print("I am in randint")
			else:
				if self.test_agent_flag:
					if not self.greedy_test:
						self.agent_action=self.agent.actor.sample_act(self.observation)
						#print("i am in test and sample")
					else:
						self.agent_action=self.agent.actor.greedy_act(self.observation)
						#print("i am in test and greedy")
				else:
					self.agent_action=self.agent.actor.sample_act(self.observation)
					#print("i am in train and sample")


		else:
			#if you are in training
			if not self.test_agent_flag:
				# Explore with actions_prob
				self.agent_action = self.agent.actor.sample_act(self.observation)
				#print("I am in training and sample")
			
			#my code: in testing have greedy action or sampled action
			else:
				if self.greedy_test:
					self.agent_action = self.agent.actor.greedy_act(self.observation)
					#print("I am in greedy")
				else:
					self.agent_action= self.agent.actor.sample_act(self.observation)
					#print(self.agent_action)
					#print("I am in test and sample")
			#end of my code

			
		if self.test_agent_flag:
			self.save_models = False
		else:
			self.save_models = (randomness_request > self.randomness_threshold)

	def compute_agent_action(self, block_number):
		self.expert_action_flag = False
		if self.test_agent_flag:
			self.agent_action = self.agent.actor.greedy_act(self.observation)
		else:
			if self.transfer_learning and block_number != 1: 
				self.ppr_request = np.random.randint(100)/100
				if self.ppr_request < self.ppr_threshold:
					self.expert_action_flag = True
					self.agent_action = self.expert_agent.actor.sample_act(self.observation)
				else:
					self.agent_action = self.agent.actor.sample_act(self.observation)
				self.save_models = True
			else:
				self.agent_action = self.agent.actor.sample_act(self.observation)
				self.save_models = True


		agent_action_msg = Float64()
		agent_action_msg.data = self.agent_action
		self.agent_action_pub.publish(agent_action_msg)

	def compute_reward(self):
		pos_x_ee=0
		pos_y_ee=0
		vel_x=0
		vel_y=0

		if(rospy.get_param("/rl_control/Game/gazebo_simulation",False)):
			vel_x=self.ur3_vel.linear.x
			vel_y=self.ur3_vel.linear.y
		else:
			vel_x=self.ur3_state.twist.linear.x
			vel_y=self.ur3_state.twist.linear.y
		if (distance.euclidean([self.ur3_state.pose.position.x, self.ur3_state.pose.position.y], self.goal) <= self.goal_dis and 
			distance.euclidean([vel_x, vel_y], [0, 0]) <= self.goal_vel):
			return self.win_reward
		
		return self.lose_reward

	def check_if_game_ended(self, block_number):
		pos_x_ee=0
		pos_y_ee=0
		vel_x_ee=0
		vel_y_ee=0
		if(rospy.get_param("/rl_control/Game/gazebo_simulation",False)):
			vel_x_ee=self.ur3_vel.linear.x 
			vel_y_ee=self.ur3_vel.linear.y
		else:
			vel_x_ee=self.ur3_state.twist.linear.x
			vel_y_ee=self.ur3_state.twist.linear.y


		pos_x_ee=self.ur3_state.pose.position.x
		pos_y_ee=self.ur3_state.pose.position.y
		
		if (distance.euclidean([pos_x_ee, pos_y_ee], self.goal) <= self.goal_dis and 
			distance.euclidean([vel_x_ee, vel_y_ee], [0, 0]) <= self.goal_vel) or self.timeout:
		
		
			score_msg = Score()
			score_msg.score.data = 150 + self.episode_reward
			score_msg.block.data = block_number
			if self.timeout:
				score_msg.status.data = "Timeout"
				rospy.loginfo('Episode ended with timeout')
				t_lose = threading.Thread(target=play, args=(self.lose_audio,))
				t_lose.start()
			else:
				score_msg.status.data = "Success"
				rospy.loginfo('Episode ended with goal reached')
				t_win = threading.Thread(target=play, args=(self.win_audio,))
				t_win.start()
			print("Score Message: ",score_msg)
			self.score_pub.publish(score_msg)
			return True
		return False
	
	def start_clock(self, block_number):
		score_msg = Score()
		score_msg.score.data = -1
		score_msg.block.data = block_number
		score_msg.status.data = "Start"
		print(score_msg)
		self.score_pub.publish(score_msg)

	def ee_state_callback(self, msg):
		self.ur3_state = msg

	def ee_velocities_callback(self, msg):
		self.ur3_vel=msg

	def get_state(self):
		vel_x=0
		vel_y=0

		if(rospy.get_param("/rl_control/Game/gazebo_simulation",False)):
			vel_x=self.ur3_vel.linear.x
			vel_y=self.ur3_vel.linear.y
		else:
			vel_x=self.ur3_state.twist.linear.x
			vel_y=self.ur3_state.twist.linear.y

		pos_x = self.ur3_state.pose.position.x
		pos_y = self.ur3_state.pose.position.y
		
		
		if (self.num_of_states==4):
		 return np.array([((pos_x-self.min_x)/(self.max_x-self.min_x)), ((pos_y-self.min_y)/(self.max_y-self.min_y)), ((vel_x-self.min_vel_x)/(self.max_vel_x-self.min_vel_x)), ((vel_y-self.min_vel_y)/(self.max_vel_y-self.min_vel_y))]), pos_x, pos_y
		elif (self.num_of_states==2):
		 return np.array([((pos_x-self.min_x)/(self.max_x-self.min_x)), ((pos_y-self.min_y)/(self.max_y-self.min_y))]), pos_x, pos_y

	def save_experience(self, interaction):
		self.agent.memory.add(*interaction)

	def human_callback(self, msg):
		self.human_action = msg.linear.x / 5

	def grad_updates(self, i_block):
		update_cycles = int(self.update_cycles)
		start_grad_updates = rospy.get_time()
		rospy.loginfo('Performing {} updates'.format(update_cycles))
		for i in tqdm(range(update_cycles)):
			#self.agent.learn()
			self.agent.learn(i, i_block) #THIS episode number shoes the percentage we take from the replay buffer each time an offline update happens
			self.agent.soft_update_target()
		end_grad_updates = rospy.get_time()

		return end_grad_updates - start_grad_updates

	def run(self, i_episode, block_number, init_position):
		rospy.loginfo('Episode: {}'.format(i_episode))
		start_time = rospy.Time.now().to_sec()
		count = 0
		while rospy.Time.now().to_sec() - start_time <= 4:
			if count < 3:
				play(self.start_audio[0])
				count += 1
				rospy.sleep(0.5)
			else:
				play(self.start_audio[1])
				#rospy.sleep(0.1)

		self.start_clock(block_number)
		tmp_time = 0
		total_travelled_distance = 0
		while not rospy.is_shutdown():
			self.timestep += 1
			
			if self.timestep == self.max_timesteps:
				self.timeout = True
			
			self.observation, pos_x, pos_y = self.get_state()
			# tstamp = str(datetime.now().strftime("%H:%M:%S.%f")[:-3])
			tst = datetime.now()
			tstamp = int(tst.timestamp() * 1000)
			if rospy.get_time() - tmp_time > self.action_duration:
				tmp_time = rospy.get_time()
				self.compute_agent_action(block_number)
			
			rospy.sleep(self.action_duration)
			
			self.observation_, pos_x_, pos_y_  = self.get_state()
			self.reward = self.compute_reward()
			self.episode_reward += self.reward
			self.done = self.check_if_game_ended(block_number)

			if self.human_action is None:
				self.human_action = 0
			cmd_acc_human = self.human_action / 5.0
			if cmd_acc_human == 0.4:
				cmd_acc_human = -0.2
			cmd_acc_agent = self.agent_action / 5.0
			if cmd_acc_agent == 0.4:
				cmd_acc_agent = -0.2

			
			self.human_actions.append(self.human_action)
			self.agent_actions.append(self.agent_action)
			self.ee_pos_x_prev.append(self.observation[0])
			self.ee_pos_y_prev.append(self.observation[1])
			self.timestamps.append(tstamp)
			self.unnorm_pos_x_prev.append(pos_x)
			self.unnorm_pos_y_prev.append(pos_y)
			self.run_blocks.append(i_episode)
			self.run_steps.append(block_number)
			self.run_init_pos.append(init_position)

			if self.num_of_states==4:
				self.ee_vel_x_prev.append(self.observation[2])
				self.ee_vel_y_prev.append(self.observation[3])

			self.ee_pos_x_next.append(self.observation_[0])
			self.ee_pos_y_next.append(self.observation_[1])
			self.unnorm_pos_x_next.append(pos_x_)
			self.unnorm_pos_y_next.append(pos_y_)

			if self.num_of_states==4:
				self.ee_vel_x_next.append(self.observation_[2])
				self.ee_vel_y_next.append(self.observation_[3])

			self.cmd_acc_x.append(cmd_acc_human)
			self.cmd_acc_y.append(cmd_acc_agent)
			if self.timestep == 1:
				self.start_time = rospy.get_time()
		
			if not self.test_agent_flag:
				self.save_experience([self.observation, self.agent_action, self.reward, self.observation_, self.done])
			
			total_travelled_distance += distance.euclidean([self.observation_[0]*(self.max_x-self.min_x)+self.min_x,self.observation_[1]*(self.max_y-self.min_y)+self.min_y], [self.observation[0]*(self.max_x-self.min_x)+self.min_x,self.observation[1]*(self.max_y-self.min_y)+self.min_y]) 
			
			if self.done:
				self.end_time = rospy.get_time()
				break

		
		new_state_info=[]
		if self.num_of_states==4:
			new_state_info = list(zip(self.timestamps, self.run_blocks, self.run_steps, self.run_init_pos, self.human_actions, self.agent_actions, self.unnorm_pos_x_prev, self.unnorm_pos_y_prev, self.unnorm_pos_x_next, self.unnorm_pos_y_next, self.ee_pos_x_prev, self.ee_pos_y_prev, self.ee_vel_x_prev, self.ee_vel_y_prev, self.ee_pos_x_next, self.ee_pos_y_next, self.ee_vel_x_next, self.ee_vel_y_next, self.cmd_acc_x, self.cmd_acc_y))
		elif self.num_of_states==2:
			new_state_info = list(zip(self.timestamps, self.run_blocks, self.run_steps, self.run_init_pos, self.human_actions, self.agent_actions, self.unnorm_pos_x_prev, self.unnorm_pos_y_prev, self.unnorm_pos_x_next, self.unnorm_pos_y_next,  self.ee_pos_x_prev, self.ee_pos_y_prev, self.ee_pos_x_next, self.ee_pos_y_next, self.cmd_acc_x, self.cmd_acc_y))
		else:
			print("Wrong dimension of states")
			exit()

		if self.test_agent_flag or block_number == 1 or block_number == 14:
			self.test_reward_history.append(self.episode_reward)
			self.test_episode_history.append(i_episode)
			self.test_timestamps.append(tstamp)
			self.test_data_block_history.append(block_number)
			self.test_init_pos.append(init_position)
			self.test_episode_duration.append(self.end_time - self.start_time)
			self.test_travelled_distance.append(total_travelled_distance)
			self.test_number_of_timesteps.append(self.timestep)
			self.test_state_info.extend(new_state_info) 
			self.test_state_info.append((0,)*len(self.test_state_info[0]))
			if self.test_best_reward < self.episode_reward:
				self.test_best_reward = self.episode_reward
		else:
			self.reward_history.append(self.episode_reward)
			self.episode_history.append(i_episode)
			self.train_timestamps.append(tstamp)
			self.data_block_history.append(block_number)
			self.init_pos.append(init_position)
			self.episode_duration.append(self.end_time - self.start_time)
			self.travelled_distance.append(total_travelled_distance)
			self.number_of_timesteps.append(self.timestep)
			self.state_info.extend(new_state_info)  
			self.state_info.append((0,)*len(self.state_info[0]))
			if self.best_episode_reward < self.episode_reward:
				self.best_episode_reward = self.episode_reward

	def test_different_positions(self, block_number):
		# start_position_0 = [-0.341, 0.300, 0.173] # down left
		# start_position_1 = [-0.341, 0.200, 0.173] # down right
		# start_position_2 = [-0.180, 0.300, 0.173] # up left 
		# start_position_3 = [-0.180, 0.200, 0.173] # up right
		# new_start_pos = [start_position_0, start_position_1, start_position_2, start_position_3]
		v = [4,4,5,5,6,6,7,7]
		random.shuffle(v)

		self.test_agent_flag = True
		for i in range(1, self.test_max_episodes+1):
			self.test_count += 1
			position = v[i-1]
			self.reset(position)
			self.run(i, block_number, position)

	def train(self, block_number):
		v = [0,0,1,1,2,2,3,3]
		random.shuffle(v)
		self.test_agent_flag = False
		
		for i in range(1, self.test_max_episodes+1):
			position = v[i-1]
			self.reset(position)
			self.run(i, block_number, position)

	def test(self, block_number):
		v = [0,0,1,1,2,2,3,3]
		random.shuffle(v)
		if block_number == 1 or block_number == 14: # Baseline - sample actions
			self.test_agent_flag = False
		else:
			self.test_agent_flag = True
		
		for i in range(1, self.test_max_episodes+1):
			self.test_count += 1
			position = v[i-1]
			self.reset(position)
			self.run(i, block_number, position)

	def train_models(self, i_block):
		self.agent.entropy_history = []
		self.agent.block_history = []
		self.agent.step_history = []
		self.agent.entropy_loss_history = []
		self.agent.temperature_history = []
		self.agent.q1_history=[]
		self.agent.q2_history=[]
		self.agent.policy_loss_history=[]
		self.agent.q1_loss_history=[]
		self.agent.q2_loss_history=[]
		self.agent.targetqhistory=[]

		start_training_time = rospy.get_time()
		train_msg = Bool()
		train_msg.data = True
		self.train_pub.publish(train_msg)
		self.compute_update_cycles()
		if self.update_cycles > 0:
			grad_updates_duration = self.grad_updates(i_block)
			temperature=list(zip(
				  self.agent.block_history,
				  self.agent.step_history,
				  self.agent.temperature_history,
				  self.agent.entropy_history,
				  self.agent.entropy_loss_history,
				  self.agent.q1_history,
				  self.agent.q2_history,
				  self.agent.q1_loss_history,
				  self.agent.q2_loss_history,
				  self.agent.targetqhistory,
				  self.agent.policy_loss_history,))
			self.ogu_data.extend(temperature) 


		remaining_wait_time = self.rest_period - (rospy.get_time() - start_training_time)
		start_remaining_time = rospy.get_time()
	
		while rospy.get_time() - start_remaining_time < remaining_wait_time:
			pass

		train_msg = Bool()
		train_msg.data = False
		self.train_pub.publish(train_msg)
		

	def compute_update_cycles(self):
		if self.scheduling == 'uniform':
			if (rospy.get_param("/rl_control/Experiment/update_per_game",False)): 
				self.update_cycles = math.ceil(self.total_update_cycles / math.ceil(self.max_episodes / self.agent.update_interval))/10
			else:
				self.update_cycles = math.ceil(self.total_update_cycles / math.ceil(self.max_episodes / self.agent.update_interval))
		elif self.scheduling == 'descending':
			self.update_cycles /= 2
		else:
			raise Exception("Choose a valid scheduling procedure")
	
	def initiale_offline_update(self):
		train_msg = Bool()
		train_msg.data = True
		self.train_pub.publish(train_msg)

		# Existing offline update logic
		start_training_time = rospy.get_time() 
		self.compute_update_cycles()  # Compute the number of updates to perform
		if self.update_cycles > 0:
			grad_updates_duration = self.grad_updates(0)  # Perform the updates
			self.agent.save_models()  # Save the updated model
   
		remaining_wait_time = self.rest_period - (rospy.get_time() - start_training_time)
		start_remaining_time = rospy.get_time()
		while rospy.get_time() - start_remaining_time < remaining_wait_time:
			pass

		train_msg = Bool()
		train_msg.data = False
		self.train_pub.publish(train_msg)

# def wait_for_keypress(): 
#     stdscr = curses.initscr()
#     curses.noecho()
#     stdscr.nodelay(True)
#     stdscr.refresh()
#     print("Press any key to continue...")
#     try:
#         while True:
#             key = stdscr.getch()
#             if key != -1:
#                 return
#     finally:
#         curses.endwin()

def wait_for_keypress():
	stdscr = curses.initscr()
	curses.noecho()
	curses.cbreak()
	stdscr.keypad(True)
	stdscr.addstr("Continue the game? Press Y for Yes, N for No: ")
	stdscr.refresh()

	try:
		while True:
			c = stdscr.getch()
			if c in [ord('Y'), ord('y')]:
				return True
			elif c in [ord('N'), ord('n')]:
				return False
			else:
				stdscr.addstr(1, 0, "Invalid key. Please press Y or N: ")
				stdscr.refresh()
	finally:
		curses.nocbreak()
		stdscr.keypad(False)
		curses.echo()
		curses.endwin()


def game_loop(game, data_dir):
	if game.train_model:
		rospy.loginfo('\n ******* Mode: Baseline ******* \n')
		game.test(1)										# test with random agent initial games first 'start_training_on_episode' games
		save_baseline_data(game, 1, data_dir)
		game.agent.save_models(1)
		rospy.set_param("/rl_control/Game/save_init_ag",True) 	# show that you saved the initial agents
		wait_for_keypress()
		
		for i_block in range(2, game.num_blocks + 2):
			if i_block % 2 == 0:
				game.train(i_block)
				game.train_models(i_block)
				game.agent.save_models(i_block)
				save_train_data(game, i_block, data_dir)
				game.agent.memory.save_buffer('/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Paper_Results/buffers/expert_buffer', i_block)
			else:
				game.test(i_block)
				save_test_data(game, i_block, data_dir)
				print("-------------------------")
				print(i_block)
				print("-------------------------")
				if not wait_for_keypress():
					rospy.loginfo("Game terminated by user choice.")
					break  # Exit the for loop
				# if i_block == 7 or i_block == 13:
				# 	print("-------------------------")
				# 	print("ENTERED")
				# 	wait_for_keypress()
				# 	print("EXIT")
				# 	print("-------------------------")
	else:
		rospy.loginfo('Testing')
		game.test()

def position_loop(game, data_dir): 
	game.test_different_positions(17)
	save_test_data(game, "different_initial_positions", data_dir)

def baseline_loop(game, data_dir): 
	game.test(14)
	save_test_data(game, "baseline_repeat", data_dir)
	wait_for_keypress()

def expert_loop(game, data_dir, expert_number):
	if expert_number == 1:
		game.test(15)
		save_test_data(game, "expert_1", data_dir)
	else:
		game.test(16)
		save_test_data(game, "expert_2", data_dir)

def spawn_marker(model_name, x, y):
	with open('/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/scripts/ee_marker.sdf', 'r') as f:
		model_xml = f.read()

	spawn_model_prox = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
	pose = Pose()
	pose.position.x = x
	pose.position.y = y
	pose.position.z = 0.86
	spawn_model_prox(model_name, model_xml, '', pose, "robot")

def move_model(x, y):
	state_msg = ModelState()
	state_msg.model_name = 'simple_cylinder'  # Change to your model's name
	state_msg.pose.position.x = x
	state_msg.pose.position.y = y
	state_msg.pose.position.z = 0.4  # adjust height
	state_msg.pose.orientation.x = 0
	state_msg.pose.orientation.y = 0
	state_msg.pose.orientation.z = 0
	state_msg.pose.orientation.w = 1

	pub.publish(state_msg)

if __name__ == "__main__":
	load_model_for_training = rospy.get_param("rl_control/Game/load_model_training", False)
	_, data_dir, plot_dir = get_save_dir(load_model_for_training)
	game = RL_Control()

	if game.normalized==True:
		rospy.logwarn("The user has selected normalized features")
	else:
		rospy.logwarn("User has selected the real values of the features")

	start_experiment_time = rospy.get_time()
	if (rospy.get_param("/rl_control/Game/gazebo_simulation",False)):
		spawn_marker('simple_cylinder',-0.333,-0.346)
	   
	################################# PHASE 1-3 | HAC ##########################################################
	game_loop(game, data_dir)
	end_experiment_time = rospy.get_time()
	game.reset(0)
	save_data(game, data_dir)
	plot_statistics(game, plot_dir)
	rospy.loginfo("Phase 1-3 of the Experimented Ended!!")
	rospy.loginfo("Total experiment duration: {} mins".format((end_experiment_time - start_experiment_time)/60))
	############################################################################################################


	################################# PHASE 4A | Human Learning #####################################
	# We load the weights of the baseline and run 8 games
	game.load_baseline_weights()
	baseline_loop(game, data_dir)
	rospy.loginfo("Phase 4a: Human Learning Phase Ended!!")
	#################################################################################################


	################################# PHASE 4B | Overfitting to Expert Behavior #####################
	# We load the policy of two expert players, and let the novice player play 8 games with them (X2)
	game.load_expert_weights(expert_number=1)
	expert_loop(game, data_dir, expert_number=1)
	rospy.loginfo("Phase 4b: Finished Playing with the Expert Policy!")
	# game.load_expert_weights(expert_number=2)
	# expert_loop(game, data_dir, expert_number=2)
	# rospy.loginfo("Phase 4b of the Experimented Ended!!")
	#################################################################################################


	####################### PHASE 4C | Learning from Different Initial Positions ####################
	# # We continue with the same user weights and load the game from new random initial position
	# position_loop(game, data_dir)
	# rospy.loginfo("Phase 4c of the Experimented Ended!!")
	#################################################################################################