import pandas as pd
max_vel_x=0.2
max_vel_y=0.2
min_vel_x=-0.2
min_vel_y=-0.2
file="/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Ilias_Experiments/games_info/Ilias_news6_2_21102025/data/rl_test_data_block_3.csv"
df=pd.read_csv(file)
for i in range(len(df["acc_x_prev"])):
	if ((df["acc_x_prev"][i]>1)and(df["acc_x"][i]>0.3))or((df["acc_x_prev"][i]<1)and(df["acc_x"][i]<-0.3)):
		print(df["acc_x"][i],i)
print("comapre")
for i in range(len(df["acc_x_prev"])-1):
	if (abs(df["acc_x_"][i]-df["cmd_acc_agent"][i])>0.01):
		print(df["acc_x_"][i],df["cmd_acc_agent"][i],i)		
"""
print("check with velocities")
for i in range(len(df["ee_vel_x_prev"])):
	if not(df["ee_vel_x_next"][i]==df["ee_vel_y_next"][i]==df["ee_vel_x_prev"][i]==df["ee_vel_y_prev"][i]==df["ee_pos_x_next"][i]==df["ee_pos_x_next"][i]==0):
		acc_x_pred=(df["ee_vel_x_next"][i]*(max_vel_x-min_vel_x)+min_vel_x-(df["ee_vel_x_prev"][i]*(max_vel_x-min_vel_x)+min_vel_x))/0.2
		acc_y_pred=(df["ee_vel_y_next"][i]*(max_vel_y-min_vel_y)+min_vel_y-(df["ee_vel_y_prev"][i]*(max_vel_y-min_vel_y)+min_vel_y))/0.2
		if (acc_x_pred!=df["acc_x_"][i])and(abs(acc_x_pred-df["acc_x_"][i])>0.01):
			print("x",acc_x_pred,df["acc_x_"][i],i)
		elif (acc_x_pred!=df["acc_y_"][i])and(abs(acc_y_pred-df["acc_y_"][i])>0.01):
			print("y",acc_y_pred,df["acc_y_"][i],i)

"""