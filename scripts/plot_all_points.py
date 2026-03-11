import numpy
import pandas as pd
import matplotlib.pyplot as plt

block_number = 7


def plot_wins(method,fig,axs,color1="red",color2="red",name=""):
	count=0
	for i in method:
		df=pd.read_csv(i)
		reward=df["Rewards"]+150
		reward_wins=df["Rewards"]
		norm_dist=df["Travelled Distance"]*df["Episodes Duration in Seconds"]/30

		wins=[]
		avg_reward=[]
		avg_norm_dist=[]
		for j in range(0, block_number+1):
			avg_reward.append(numpy.average(reward[j:((j+1) * block_number)]))
			avg_norm_dist.append(numpy.average(norm_dist[j:((j+1) * block_number)]))
			w=0
			for l in range(block_number):
				print(j*block_number + l)
				if reward_wins[j*block_number + l]>-150:
					w+=1
			wins.append(w)
		print(avg_reward)
		print(wins)
		print(avg_norm_dist)
		if count==0:
			axs[0].plot(range(0,8),avg_reward, 'o-',color=color1,label=name)
		else:
			axs[0].plot(range(0,8),avg_reward, 'o-',color=color1)
		axs[0].set_title("Averge reward per Block")
		axs[0].set(xlabel='Block', ylabel='Reward')
		axs[0].set_ylim(0,165)
		#plt.title('Averge reward per Block')
		

		if count==0:
			axs[1].plot(range(0,8),wins, 'o-',color=color2,label=name)
		else:
			axs[1].plot(range(0,8),wins, 'o-',color=color2)
		axs[1].set_ylim(0,11)
		axs[1].set_title("Wins per Block")
		axs[1].set(xlabel='Block', ylabel='Wins')

		if count==0:
			axs[2].plot(range(0,8),avg_norm_dist, 'o-',color=color2,label=name)
		else:
			axs[2].plot(range(0,8),avg_norm_dist, 'o-',color=color2)
		axs[2].set_ylim(0,2)
		axs[2].set_title("Average Normalized Distance per Block")
		axs[2].set(xlabel='Block', ylabel='Normalized Distance')
		count=1
	axs[0].legend()
	axs[1].legend()
	axs[2].legend()

# method_8_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_51/data/test_data.csv",
# "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_52/data/test_data.csv","/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_54/data/test_data.csv",
# "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_55/data/test_data.csv",]

# method_7_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_50/data/test_data.csv",
# "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_49/data/test_data.csv",
# "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_48/data/test_data.csv",
# "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_47/data/test_data.csv",]
"""
robot_christos = ["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Paper_Results/games_info/100K_every8_uniform_200ms_Christos_01_LfD_TL_2/data/test_data_block_17.csv"]
robot_maria =["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Paper_Results/games_info/100K_every8_uniform_200ms_Maria_01_LfD_TL_2/data/test_data_block_17.csv"]
robot_dimitris = ["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Paper_Results/games_info/100K_every8_uniform_200ms_Dimitris_01_LfD_TL_2/data/test_data_block_17.csv"]

fig,axs=plt.subplots(1,3)
#plot_wins(method_8_data,fig=fig,axs=axs,name="method 8")
plot_wins(robot_dimitris,fig=fig,axs=axs,color1="red",color2="red",name="Dimitris")
#plt.show()
plot_wins(robot_christos,fig=fig,axs=axs,color1="blue",color2="blue",name="Christos")
#plt.show()
plot_wins(robot_maria,fig=fig,axs=axs,color1="green",color2="green",name="Maria")
plt.show()
"""