import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
method_1_data=[
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_8/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_7/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/entropy.csv",    
]
method_2_data=[
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_2/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_1/data/entropy.csv",
]

method_3_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_25/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_26/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_27/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_28/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_29/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_30/data/entropy.csv"]

method_4_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_31/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_32/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_33/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_34/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_35/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_36/data/entropy.csv"]

method_5_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_41/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_37/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_38/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_39/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_40/data/entropy.csv"]

method_6_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_42/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_43/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_44/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_45/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_46/data/entropy.csv"]

method_7_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_50/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_49/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_48/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_47/data/entropy.csv",]

method_8_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_51/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_52/data/entropy.csv","/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_54/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_55/data/entropy.csv",]

method_9_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_60/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_57/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_58/data/entropy.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_59/data/entropy.csv",]

real_robot_method_7=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_56/data/entropy.csv","/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_56/data/entropy.csv"]

def smooth_values(t,smooth_step=50):
	t_smoothed=[]
	for i in range(0,len(t)):
		end=i+smooth_step-1
		#print(i)
		#print(end)
		#print(t[i:end])
		t_smoothed.append(np.mean(t[i:end]))
	print(len(t_smoothed))
	return t_smoothed

def plot_a_tempereture(method,color="blue",name="",temp_or_policy_loss="temp"):
	if len(method)==1:
		df=pd.read_csv(method[0])
		temp=df[temp_or_policy_loss]
		steps=range(0,len(temp))
		plt.plot(steps, temp, color='tab:blue', label='stim')
		plt.xlabel("#steps")
		plt.ylabel(temp_or_policy_loss)
		if temp_or_policy_loss=="temp":
			plt.title('tempreture hyperparameter per step')
		else:
			plt.title('policy loss per step')
		plt.grid(True)
		if temp_or_policy_loss=="temp":
			plt.ylim(0,1)
		plt.xlim(0,100000)
		plt.show()
	else:
		temp=[]
		len_t1=0
		len_t2=0
		for i in method:
			df=pd.read_csv(i)
			t=df[temp_or_policy_loss].to_list()
			print(len(t))
			temp.append(t)
			print(len(t))
		#find minimum and max lenght of each file
		i_min=100000
		i_max=0
		#this variable checks if all equal
		equal_len=False
		for i in temp:
			if i_min>len(i):
				i_min=len(i)
			if i_max<len(i):
				i_max=len(i)
		print(i_min)
		print(i_max)
		if i_min==i_max:
			equal_len=True
		if equal_len==False:
			for i in range(len(temp)):
				#number of the first elements that are going to be ignored
				ignore=len(temp[i])-i_min
				temp[i]=temp[i][ignore:]

		
		avg_temp=np.mean(np.array(temp), axis=0)
		std_temp = np.std(np.array(temp), axis=0)
		
		if equal_len==False:
			std_temp =np.insert(std_temp, 0, 0)
			avg_temp=np.insert(avg_temp, 0, 1)
		steps=range(0,len(avg_temp))
		plt.plot(steps, avg_temp, color=color, label=name)
		plt.fill_between(steps, avg_temp - std_temp, avg_temp + std_temp, color=color, alpha=0.3)
		plt.xlabel("#steps")
		if temp_or_policy_loss=="temp":
			plt.ylabel("temp")
		else:
			plt.ylabel("policy loss")
		if temp_or_policy_loss=="temp":
			plt.title('tempreture hyperparameter per step')
		else:
			plt.title('policy loss per step')
		plt.grid(True)
		if temp_or_policy_loss=="temp":
			plt.ylim(0,1)
		else:
			plt.ylim(-2,50)
		plt.xlim(0,100000)
		plt.legend()


    
def plot_q_values_and_loss(method,fig="",axs="",color="tab:blue",name="",q_or_entropy="q",smoothing=True):
	if len(method)==1:
		df=pd.read_csv(method[0])
		q1=[]
		q2=[]
		qtarget=[]
		q1_loss=[]
		q2_loss=[]
		q1_legend='q1'
		q1_legend_loss='q1 loss'
		y_label="q value"
		diag_title="q value per step"
		y_label2="q loss"
		diag_title2="q loss per step"
		y_limits=[-50,1]
		y_limits_loss=[0,50]
		q1=df["q1_history"]
		q2=df["q2_history"]
		qtarget=df["target_q"]
		q1_loss=df["q1_loss"]
		q2_loss=df["q2_loss"]
		
		if q_or_entropy=="q":
			if smoothing:

				q1=smooth_values(q1)
				q2=smooth_values(q2)
				qtarget=smooth_values(qtarget)
				q1_loss=smooth_values(q1_loss)
				q2_loss=smooth_values(q2_loss)

			
		else:
			q1=df["entropy"]
			q1_loss=df["entropy_loss"]
			q1_legend='entropy'
			q1_legend_loss='entropy loss'
			if smoothing:

				q1=smooth_values(q1)
				q1_loss=smooth_values(q1_loss)
			
			y_label="entropy"
			diag_title="entropy per step"
			y_label2="entropy loss"
			diag_title2="entropy loss per step"
			y_limits=[0,1.2]
			y_limits_loss=[-3,1.5]

		steps=range(0,len(q1))
		plt.subplot(1, 2, 1)
		steps=range(0,len(q1))
		plt.plot(steps, q1, color='tab:blue', label=q1_legend)
		if q_or_entropy=="q":
			plt.plot(steps, q2, color='tab:red', label='q2')
			plt.plot(steps, qtarget, color='tab:green', label='q_target')
		plt.xlabel("#steps")
		plt.ylabel(y_label)
		plt.title(diag_title)
		plt.ylim(y_limits[0],y_limits[1])
		plt.grid(True)
		plt.legend()
		plt.subplot(1, 2, 2)
		steps=range(0,len(q1_loss))
		plt.plot(steps, q1_loss, color='tab:blue', label=q1_legend_loss)
		if q_or_entropy=="q":
			plt.plot(steps, q2_loss, color='tab:red', label='q2 loss')
		plt.xlabel("#steps")
		plt.ylabel(y_label2)
		plt.ylim(y_limits_loss[0],y_limits_loss[1])
		plt.title(diag_title2)
		plt.grid(True)
		plt.legend()
		plt.suptitle(name)
		plt.tight_layout()
		plt.subplots_adjust(top=0.85) 
		plt.show()	
	else:

		q1, q2, q_target = [], [], []
		q1_loss, q2_loss = [], []
		q1_legend='q1'
		q1_legend_loss='q1 loss'
		y_label="q value"
		diag_title="q value per step"
		y_label2="q loss"
		diag_title2="q loss per step"


		y_limits=[-50,1]
		y_limits_loss=[0,50]

		if q_or_entropy!="q":
			q1_legend=name
			q1_legend_loss=name
			y_label="entropy"
			diag_title="entropy per step"
			y_label2="entropy loss"
			diag_title2="entropy loss per step"
			y_limits=[0,1.2]
			y_limits_loss=[-3,1.5]

		for i in method:
			df = pd.read_csv(i)
			if q_or_entropy=="q":
				if smoothing==False:
					q1.append(df["q1_history"].to_list())

				else:
					q=smooth_values(df["q1_history"])
					q1.append(q)

			else:
				if smoothing==False:
					q1.append(df["entropy"].to_list())
				else:
					q=smooth_values(df["entropy"])
					q1.append(q)
			if smoothing==False:
				q2.append(df["q2_history"].to_list())
				q_target.append(df["target_q"].to_list())
			else:
				q=smooth_values(df["q2_history"])
				q2.append(q)
				q=smooth_values(df["target_q"])
				q_target.append(q)
			if q_or_entropy=="q":
				if smoothing==False:
					q1_loss.append(df["q1_loss"].to_list())
				else:
					q=smooth_values(df["q1_loss"])
					q1_loss.append(q)
			else:
				if smoothing==False:
					q1_loss.append(df["entropy_loss"].to_list())
				else:
					q=smooth_values(df["entropy_loss"])
					q1_loss.append(q)
			if smoothing==False:
				q2_loss.append(df["q2_loss"].to_list())
			else:
				q=smooth_values(df["q2_loss"])
				q2_loss.append(q)

		# Find min and max lengths
		lengths = [len(arr) for arr in q1]
		i_min, i_max = min(lengths), max(lengths)
		print(i_min)
		print(i_max)

		# Compute 1st step averages/stds safely
		q1_1st_av = np.mean([row[0] for row in q1 if len(row) > 0])
		q2_1st_av = np.mean([row[0] for row in q2 if len(row) > 0])
		q_t_1st_av = np.mean([row[0] for row in q_target if len(row) > 0])
		q1l_1st_av = np.mean([row[0] for row in q1_loss if len(row) > 0])
		q2l_1st_av = np.mean([row[0] for row in q2_loss if len(row) > 0])

		q1_1st_std = np.std([row[0] for row in q1 if len(row) > 0])
		q2_1st_std = np.std([row[0] for row in q2 if len(row) > 0])
		q_t_1st_std = np.std([row[0] for row in q_target if len(row) > 0])
		q1l_1st_std = np.std([row[0] for row in q1_loss if len(row) > 0])
		q2l_1st_std = np.std([row[0] for row in q2_loss if len(row) > 0])

		equal_len = (i_min == i_max)

		if not equal_len:
			for i in range(len(q1)):
				ignore = len(q1[i]) - i_min
				q1[i] = q1[i][ignore:]
				q2[i] = q2[i][ignore:]
				q_target[i] = q_target[i][ignore:]
				q1_loss[i] = q1_loss[i][ignore:]
				q2_loss[i] = q2_loss[i][ignore:]

		# Convert lists to arrays
		q1_arr = np.array(q1)
		q2_arr = np.array(q2)
		qt_arr = np.array(q_target)
		q1l_arr = np.array(q1_loss)
		q2l_arr = np.array(q2_loss)

		# Compute mean and std across runs
		avg_q1, std_q1 = np.mean(q1_arr, axis=0), np.std(q1_arr, axis=0)
		avg_q2, std_q2 = np.mean(q2_arr, axis=0), np.std(q2_arr, axis=0)
		avg_qtarget, std_qtarget = np.mean(qt_arr, axis=0), np.std(qt_arr, axis=0)
		avg_q1_loss, std_q1_loss = np.mean(q1l_arr, axis=0), np.std(q1l_arr, axis=0)
		avg_q2_loss, std_q2_loss = np.mean(q2l_arr, axis=0), np.std(q2l_arr, axis=0)

		if not equal_len:
			# Insert first-step stats at beginning
			avg_q1 = np.insert(avg_q1, 0, q1_1st_av)
			std_q1 = np.insert(std_q1, 0, q1_1st_std)
			avg_q2 = np.insert(avg_q2, 0, q2_1st_av)
			std_q2 = np.insert(std_q2, 0, q2_1st_std)
			avg_qtarget = np.insert(avg_qtarget, 0, q_t_1st_av)
			std_qtarget = np.insert(std_qtarget, 0, q_t_1st_std)
			avg_q1_loss = np.insert(avg_q1_loss, 0, q1l_1st_av)
			std_q1_loss = np.insert(std_q1_loss, 0, q1l_1st_std)
			avg_q2_loss = np.insert(avg_q2_loss, 0, q2l_1st_av)
			std_q2_loss = np.insert(std_q2_loss, 0, q2l_1st_std)

		# Plotting
		
		if (fig=="")and(axs==""):
			fig, axs = plt.subplots(1, 2, figsize=(12, 5))
		steps = range(len(avg_q1))
		axs[0].plot(steps, avg_q1, color=color, label=q1_legend)
		axs[0].fill_between(steps, avg_q1 - std_q1, avg_q1 + std_q1, color=color, alpha=0.3)
		
		if q_or_entropy=="q":
			axs[0].plot(steps, avg_q2, color='tab:red', label='q2')
			axs[0].fill_between(steps, avg_q2 - std_q2, avg_q2 + std_q2, color='tab:red', alpha=0.3)
			axs[0].plot(steps, avg_qtarget, color='tab:green', label='q_target')
			axs[0].fill_between(steps, avg_qtarget - std_qtarget, avg_qtarget + std_qtarget, color='tab:green', alpha=0.3)
		axs[0].set_xlabel("#steps")
		axs[0].set_ylabel(y_label)
		axs[0].set_title(diag_title)
		axs[0].set_ylim(y_limits[0], y_limits[1])
		axs[0].set_xlim(0,100000)
		axs[0].legend()
		axs[0].grid(True)

		steps = range(len(avg_q1_loss))
		axs[1].plot(steps, avg_q1_loss, color=color, label=q1_legend_loss)
		axs[1].fill_between(steps, avg_q1_loss - std_q1_loss, avg_q1_loss + std_q1_loss, color=color, alpha=0.3)
		if q_or_entropy=="q":
			axs[1].plot(steps, avg_q2_loss, color='tab:red', label='q2 loss')
			axs[1].fill_between(steps, avg_q2_loss - std_q2_loss, avg_q2_loss + std_q2_loss, color='tab:red', alpha=0.3)
		axs[1].set_xlabel("#steps")
		axs[1].set_ylabel(y_label2)
		axs[1].set_ylim(y_limits_loss[0], y_limits_loss[1])
		axs[1].set_xlim(0,100000)
		axs[1].set_title(diag_title2)
		axs[1].legend()
		axs[1].grid(True)

		fig.suptitle(name)
		fig.tight_layout()
		fig.subplots_adjust(top=0.85)
		
csvf=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_60/data/entropy.csv"]
plot_a_tempereture(csvf)
methods=[method_3_data,method_4_data,method_5_data,method_6_data,method_2_data,method_1_data,method_7_data,method_8_data,method_9_data,real_robot_method_7]
methods_names=["Non Normalized data 98K, per 10, 4 states","Non Normalized data 98K, per 10, 4 states (loosing)","Non Normalized data 98K, per 10, 2 states","Non Normalized data 98K, per 1, 2 states","Non Normalized data 35K, per 1, 2 states","Normalized data 35K, per 1, 4 states","Normalized data 98K, per 10, 4 states","Normalized data 98K, per 10, 4 states, greedy","Normalized data 98K, per 10, 2 states", "real robot, normalized data, per 10, 4 states"]
colors = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan'
]

for i in range(len(methods)):
    plot_a_tempereture(methods[i],colors[i],methods_names[i])
plt.show()
for i in range(len(methods)):
   plot_a_tempereture(methods[i],colors[i],methods_names[i],temp_or_policy_loss="policy_loss")
plt.show()
plot_q_values_and_loss(["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_56/data/entropy.csv"],name="real robot method 7",q_or_entropy="q")
plot_q_values_and_loss(["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_56/data/entropy.csv"],name="real robot method 7",q_or_entropy="entropy")
#plot_q_values_and_loss(["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_56/data/entropy.csv"],name="real robot method 7",q_or_entropy="entropy")
fig1, axs1 = plt.subplots(1, 2, figsize=(12, 5))
for i in range(len(methods)):	
	plot_q_values_and_loss(methods[i],name=methods_names[i],q_or_entropy="entropy",color=colors[i],fig=fig1,axs=axs1,smoothing=True)	
plt.show()
for i in range(len(methods)):
	plot_q_values_and_loss(methods[i],name=methods_names[i],smoothing=True)
plt.show()


