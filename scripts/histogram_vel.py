import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
"""
method_1_data=[
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_8/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_7/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/rl_test_data.csv",    
]
method_2_data=[
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_2/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_1/data/rl_test_data.csv",
]

method_3_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_25/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_26/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_27/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_28/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_29/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_30/data/rl_test_data.csv"]

method_4_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_31/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_32/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_33/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_34/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_35/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_36/data/rl_test_data.csv"]
"""
"""
method_5_data=[
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_41/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_37/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_38/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_39/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_40/data/rl_test_data.csv"]
"""
"""
method_6_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_42/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_43/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_44/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_45/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_46/data/rl_test_data.csv"]

method_7_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_50/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_49/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_48/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_47/data/rl_test_data.csv",]

method_8_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_51/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_52/data/rl_test_data.csv","/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_54/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_55/data/rl_test_data.csv",]
"""
"""
method_9_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_60/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_57/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_58/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_59/data/rl_test_data.csv",]
"""
"""
method_10_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_69/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_68/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_67/data/rl_test_data.csv",
]
"""

method_sim_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/49K_every5_uniform_200ms_Christos_half_01_LfD_TL_41/data/rl_test_data.csv"]
real_robot_method_7=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_56/data/rl_test_data.csv"]
real_robot_method_11=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_78/data/rl_test_data.csv"]
real_robot_method_12=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/49K_every5_uniform_200ms_itsmetheexpert_LfD_TL_1/data/rl_test_data.csv"]

#methods_unnorm=[method_3_data,method_4_data]
methods_norm=[method_sim_data]#method_7_data,method_8_data,method_10_data,method_1_data]
methods_real_norm=[real_robot_method_11,real_robot_method_12]

def unnormalize(data1,x_max,x_min):
	data1_un=[]
	for i in range(len(data1)):
		data1_un.append(data1[i]*(x_max-x_min)+x_min)
	return data1_un

def read_data_from_methods(methods):
	velocity_x=[]
	velocity_y=[]
	for i in methods:
		for j in i:
			df=pd.read_csv(j)
			print(j)
			print("\n")
			velocity_x.append(df["ee_vel_x_next"])
			velocity_y.append(df["ee_vel_y_next"])
	velocity_x_flat=[item for sublist in velocity_x for item in sublist]
	velocity_y_flat=[item for sublist in velocity_y for item in sublist]
	return velocity_x_flat,velocity_y_flat

#velocity_x_sim_un,velocity_y_sim_un=read_data_from_methods(methods_unnorm)

velocity_x_sim_n,velocity_y_sim_n=read_data_from_methods(methods_norm)

velocity_x_r_n1,velocity_y_r_n1=read_data_from_methods([real_robot_method_7])
velocity_x_r_n2,velocity_y_r_n2=read_data_from_methods(methods_real_norm)

vel_sim_x_un1=unnormalize(velocity_x_sim_n,0.2,-0.2)#0.012,-0.011)
vel_sim_y_un1=unnormalize(velocity_y_sim_n,0.2,-0.2)#0.012,-0.011)

vel_r_x_un1=unnormalize(velocity_x_r_n1,0.012,-0.011)
vel_r_y_un1=unnormalize(velocity_y_r_n1,0.012,-0.011)

vel_r_x_un2=unnormalize(velocity_x_r_n2,0.2,-0.2)
vel_r_y_un2=unnormalize(velocity_y_r_n2,0.2,-0.2)

vel_x_sim_un=vel_sim_x_un1#+velocity_x_sim_un
vel_y_sim_un=vel_sim_y_un1#+velocity_y_sim_un

vel_x_r_un=vel_r_x_un1+vel_r_x_un2
vel_y_r_un=vel_r_y_un1+vel_r_y_un2

plt.subplot(2, 2, 1)

# Creating a customized histogram with a density plot
sns.histplot(vel_x_sim_un, bins=30, kde=True, color='lightgreen', edgecolor='red')

# Adding labels and title
plt.xlabel('ee_vel_x')
plt.ylabel('Density')
plt.title(' Histogram of ee_vel_x with Density Plot in simulation')

plt.subplot(2, 2, 2)

# Creating a customized histogram with a density plot
sns.histplot(vel_y_sim_un, bins=30, kde=True, color='blue', edgecolor='cyan')

# Adding labels and title
plt.xlabel('ee_vel_y')
plt.ylabel('Density')
plt.title('Histogram of ee_vel_y with Density Plot in simulation')
# Display the plot
plt.subplot(2, 2, 3)

# Creating a customized histogram with a density plot
sns.histplot(vel_x_r_un, bins=30, kde=True, color='red', edgecolor='orange')

# Adding labels and title
plt.xlabel('ee_vel_x')
plt.ylabel('Density')
plt.title(' Histogram of ee_vel_x with Density Plot in reality')

plt.subplot(2, 2, 4)

# Creating a customized histogram with a density plot
sns.histplot(vel_y_r_un, bins=30, kde=True, color='yellow', edgecolor='cyan')

# Adding labels and title
plt.xlabel('ee_vel_y')
plt.ylabel('Density')
plt.title('Histogram of ee_vel_y with Density Plot in reality')
# Display the plot
plt.show()
