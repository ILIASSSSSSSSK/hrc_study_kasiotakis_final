import rospy
from sac_discrete_agent import DiscreteSACAgent
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

now = datetime.now()
YEAR = now.strftime("%Y")
MONTH = now.strftime("%m")
DAY = now.strftime("%d")
date = str(DAY) + str(MONTH) + str(YEAR)


def get_SAC_agent(observation_space, chkpt_dir=""):
	rospy.init_node('rl_control')
	buffer_max_size = rospy.get_param("rl_control/SAC/buffer_max_size", 10000)
	update_interval = rospy.get_param("rl_control/Experiment/learn_every_n_episodes", 10)
	scale = rospy.get_param("rl_control/Experiment/reward_scale", 2)
	n_actions = rospy.get_param("rl_control/Experiment/number_of_agent_actions", 3)

	if chkpt_dir == "":
		participant = rospy.get_param('rl_control/Game/participant_name', 'thanasis')
		total_number_updates = rospy.get_param('rl_control/Experiment/total_update_cycles', 1000)
		action_duration = int(rospy.get_param('rl_control/Experiment/action_duration', 100)*1000)
		training_scheduling = rospy.get_param('rl_control/Experiment/scheduling', 'uniform')
		transfer_learning = rospy.get_param("rl_control/Game/load_model_transfer_learning", False)
		save_chkpt_dir = "{}K_every{}_{}_{}ms_{}_no_TL".format(int(total_number_updates/1000), update_interval, training_scheduling, action_duration, participant) if not transfer_learning else "{}K_every{}_{}_{}ms_{}".format(int(total_number_updates/1000), update_interval, training_scheduling, action_duration, participant)
		
		save_chkpt_dir = os.path.join(rospy.get_param("rl_control/Game/full_path", "tmp"), 'rl_models/' + save_chkpt_dir)
		i=1
		while os.path.exists(save_chkpt_dir + '_' + str(i)):
			i += 1
		chkpt_dir = save_chkpt_dir + '_' + str(i)
	
	return DiscreteSACAgent(input_dims=observation_space, n_actions=n_actions, 
		chkpt_dir=chkpt_dir, buffer_max_size=buffer_max_size, 
		update_interval=update_interval, reward_scale=scale)

def get_save_dir(load_model_for_training=False):
	full_path = os.path.join(rospy.get_param("/rl_control/Game/full_path", ""), "Ilias_Experiments", "games_info")

	if not load_model_for_training:
		# total_number_updates = rospy.get_param('rl_control/Experiment/total_update_cycles', 1000)
		# update_interval = rospy.get_param('rl_control/Experiment/learn_every_n_episodes', 10)
		# action_duration = int(rospy.get_param('rl_control/Experiment/action_duration', 100)*1000)
		# training_scheduling = rospy.get_param('rl_control/Experiment/scheduling', 'uniform')
		# transfer_learning = rospy.get_param("rl_control/Game/load_model_transfer_learning", False)
		#save_chkpt_dir = "{}K_every{}_{}_{}ms_{}_no_TL".format(int(total_number_updates/1000), update_interval, training_scheduling, action_duration, participant) if not transfer_learning else "{}K_every{}_{}_{}ms_{}".format(int(total_number_updates/1000), update_interval, training_scheduling, action_duration, participant)
		#save_chkpt_dir = "{}K_every{}_{}_{}ms_{}_no_TL".format(int(total_number_updates/1000), update_interval, training_scheduling, action_duration, participant) if not transfer_learning else "{}K_every{}_{}_{}ms_{}".format(int(total_number_updates/1000), update_interval, training_scheduling, action_duration, participant)
		
		participant = rospy.get_param('rl_control/Game/participant_name', 'thanasis')
		save_chkpt_dir = "{}_{}".format(participant, date)
		save_chkpt_dir = os.path.join(full_path, save_chkpt_dir)
		
		# Create new folder if the same {NAME}_{DATE} exists
		data_dir = os.path.join(save_chkpt_dir, 'data')
		plot_dir = os.path.join(save_chkpt_dir, 'plots')
		if os.path.exists(save_chkpt_dir):
			i = 1
			if os.path.exists(save_chkpt_dir + '_' + str(i)):
				while os.path.exists(save_chkpt_dir + '_' + str(i)):
					i += 1
				save_chkpt_dir = save_chkpt_dir + '_' + str(i)
		else:
			os.makedirs(os.path.join(full_path, save_chkpt_dir))
			os.makedirs(os.path.join(full_path, data_dir))
			os.makedirs(os.path.join(full_path, plot_dir))
	else:
		save_chkpt_dir = os.path.join(full_path, rospy.get_param("rl_control/Game/load_model_training_dir", ""))
		data_dir = os.path.join(save_chkpt_dir, 'data')
		plot_dir = os.path.join(save_chkpt_dir, 'plots')

	return save_chkpt_dir, data_dir, plot_dir


def save_data(game, data_dir):
	participant = rospy.get_param('rl_control/Game/participant_name', 'thanasis')
	with open(data_dir+'/train_data.csv', 'ab') as outputFile:
		headers = [['timestamp'], ['block'], ['episode'], ['start_pos'], ['expert'],['Rewards'], ['Episodes Duration in Seconds'], ['Travelled Distance'], ['Episodes Duration in Timesteps']]		
		np.savetxt(outputFile, list(zip(*headers)), delimiter=',', fmt='%s')
		data = [game.train_timestamps, game.data_block_history, game.episode_history, game.init_pos, len(game.reward_history) * [game.expert_action_flag],game.reward_history, game.episode_duration, game.travelled_distance, game.number_of_timesteps]		
		np.savetxt(outputFile, list(zip(*data)), delimiter=',')

	with open(data_dir+'/test_data.csv', 'ab') as outputFile:
		headers = [['timestamp'], ['block'], ['episode'], ['start_pos'], ['Rewards'], ['Episodes Duration in Seconds'], ['Travelled Distance'], ['Episodes Duration in Timesteps']]
		np.savetxt(outputFile, list(zip(*headers)), delimiter=',', fmt='%s')
		data = [game.test_timestamps, game.test_data_block_history, game.test_episode_history, game.test_init_pos, game.test_reward_history, game.test_episode_duration, game.test_travelled_distance, game.test_number_of_timesteps]
		np.savetxt(outputFile, list(zip(*data)), delimiter=',')

	with open(data_dir+'/rl_train_data.csv', 'ab') as outputFile:
		np.savetxt(outputFile, [game.state_info[0]], delimiter=',', fmt='%s')
		np.savetxt(outputFile, game.state_info[1:], delimiter=',')

	with open(data_dir+'/rl_test_data.csv', 'ab') as outputFile:
		np.savetxt(outputFile, [game.test_state_info[0]], delimiter=',', fmt='%s')
		np.savetxt(outputFile, game.test_state_info[1:], delimiter=',')

	with open(data_dir+'/entropy.csv', 'ab') as outputFile:
		np.savetxt(outputFile, [game.ogu_data[0]], delimiter=',', fmt='%s')
		np.savetxt(outputFile, game.ogu_data[1:], delimiter=',')

def save_train_data(game, i_block, data_dir):
	participant = rospy.get_param('rl_control/Game/participant_name', 'thanasis')
	with open(data_dir+'/train_data_block_' + str(i_block) + '.csv', 'ab') as outputFile:
		headers = [['timestamp'], ['block'], ['episode'], ['start_pos'], ['expert'],['Rewards'], ['Episodes Duration in Seconds'], ['Travelled Distance'], ['Episodes Duration in Timesteps']]		
		np.savetxt(outputFile, list(zip(*headers)), delimiter=',', fmt='%s')
		data = [game.train_timestamps, game.data_block_history, game.episode_history, game.init_pos, len(game.reward_history) * [game.expert_action_flag],game.reward_history, game.episode_duration, game.travelled_distance, game.number_of_timesteps]		
		np.savetxt(outputFile, list(zip(*data)), delimiter=',')

	with open(data_dir+'/rl_train_data_block_' + str(i_block) + '.csv', 'ab') as outputFile:
		np.savetxt(outputFile, [game.state_info[0]], delimiter=',', fmt='%s')
		np.savetxt(outputFile, game.state_info[1:], delimiter=',')

	with open(data_dir+'/entropy_block_' + str(i_block) + '.csv', 'ab') as outputFile:
		np.savetxt(outputFile, [game.ogu_data[0]], delimiter=',', fmt='%s')
		np.savetxt(outputFile, game.ogu_data[1:], delimiter=',')

def save_test_data(game, i_block, data_dir):
	participant = rospy.get_param('rl_control/Game/participant_name', 'thanasis')
	with open(data_dir+'/test_data_block_' + str(i_block) + '.csv', 'ab') as outputFile:
		headers = [['timestamp'], ['block'], ['episode'], ['start_pos'], ['Rewards'], ['Episodes Duration in Seconds'], ['Travelled Distance'], ['Episodes Duration in Timesteps']]
		np.savetxt(outputFile, list(zip(*headers)), delimiter=',', fmt='%s')
		data = [game.test_timestamps, game.test_data_block_history, game.test_episode_history, game.test_init_pos, game.test_reward_history, game.test_episode_duration, game.test_travelled_distance, game.test_number_of_timesteps]
		np.savetxt(outputFile, list(zip(*data)), delimiter=',')

	with open(data_dir+'/rl_test_data_block_' + str(i_block) + '.csv', 'ab') as outputFile:
		np.savetxt(outputFile, [game.test_state_info[0]], delimiter=',', fmt='%s')
		np.savetxt(outputFile, game.test_state_info[1:], delimiter=',')

def save_baseline_data(game, i_block, data_dir):
	participant = rospy.get_param('rl_control/Game/participant_name', 'thanasis')
	print("Saving baseline data...")
	with open(data_dir+'/baseline_data.csv', 'ab') as outputFile:
		headers = [['timestamp'], ['block'], ['episode'], ['start_pos'], ['Rewards'], ['Episodes Duration in Seconds'], ['Travelled Distance'], ['Episodes Duration in Timesteps']]
		np.savetxt(outputFile, list(zip(*headers)), delimiter=',', fmt='%s')
		data = [game.test_timestamps, game.test_data_block_history, game.test_episode_history, game.test_init_pos, game.test_reward_history, game.test_episode_duration, game.test_travelled_distance, game.test_number_of_timesteps]
		np.savetxt(outputFile, list(zip(*data)), delimiter=',')

	with open(data_dir+'/rl_baseline_data.csv', 'ab') as outputFile:
		np.savetxt(outputFile, [game.test_state_info[0]], delimiter=',', fmt='%s')
		np.savetxt(outputFile, game.test_state_info[1:], delimiter=',')

def plot_statistics(game, plot_dir):
	fig = plt.figure()
	ax = plt.axes()
	fig.suptitle('Rewards over episodes')
	ax.plot(np.arange(1, len(game.reward_history)+1), game.reward_history)
	ax.set_xlabel('Episodes(N)')
	ax.set_ylabel('Rewards')
	ax.grid()
	plt.savefig(plot_dir+'/rewards.png')

	fig = plt.figure()
	ax = plt.axes()
	fig.suptitle('Episodes duration')
	ax.plot(np.arange(1, len(game.episode_duration)+1), game.episode_duration)
	ax.set_xlabel('Episodes(N)')
	ax.set_ylabel('Duration(sec)')
	ax.grid()
	plt.savefig(plot_dir+'/time_duration.png')

	fig = plt.figure()
	ax = plt.axes()
	fig.suptitle('Travelled Distance')
	ax.plot(np.arange(1, len(game.travelled_distance)+1), game.travelled_distance)
	ax.set_xlabel('Episodes(N)')
	ax.set_ylabel('Travelled(m)')
	ax.grid()
	plt.savefig(plot_dir+'/travelled_distance.png')

	fig = plt.figure()
	ax = plt.axes()
	fig.suptitle('Number of Timesteps')
	ax.plot(np.arange(1, len(game.number_of_timesteps)+1), game.number_of_timesteps)
	ax.set_xlabel('Episodes(N)')
	ax.set_ylabel('Timesteps(M)')
	ax.grid()
	plt.savefig(plot_dir+'/number_of_timesteps.png')

	fig = plt.figure()
	ax = plt.axes()
	fig.suptitle('Rewards over episodes')
	ax.plot(np.arange(1, len(game.test_reward_history)+1), game.test_reward_history)
	ax.set_xlabel('Episodes(N)')
	ax.set_ylabel('Rewards')
	ax.grid()
	plt.savefig(plot_dir+'/test_rewards.png')

	fig = plt.figure()
	ax = plt.axes()
	fig.suptitle('Episodes duration')
	ax.plot(np.arange(1, len(game.test_episode_duration)+1), game.test_episode_duration)
	ax.set_xlabel('Episodes(N)')
	ax.set_ylabel('Duration(sec)')
	ax.grid()
	plt.savefig(plot_dir+'/test_time_duration.png')

	fig = plt.figure()
	ax = plt.axes()
	fig.suptitle('Travelled Distance')
	ax.plot(np.arange(1, len(game.test_travelled_distance)+1), game.test_travelled_distance)
	ax.set_xlabel('Episodes(N)')
	ax.set_ylabel('Travelled(m)')
	ax.grid()
	plt.savefig(plot_dir+'/test_travelled_distance.png')

	fig = plt.figure()
	ax = plt.axes()
	fig.suptitle('Number of Timesteps')
	ax.plot(np.arange(1, len(game.test_number_of_timesteps)+1), game.test_number_of_timesteps)
	ax.set_xlabel('Episodes(N)')
	ax.set_ylabel('Timesteps(M)')
	ax.grid()
	plt.savefig(plot_dir+'/test_number_of_timesteps.png')

	#plt.show()
