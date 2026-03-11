import pandas as pd

max_vel_x=0.2
max_vel_y=0.2
min_vel_x=-0.2
min_vel_y=-0.2

max_x=-0.162
min_x=-0.350
max_y=0.348
min_y=0.159

dt=0.2

def check_if_right(file):
	df=pd.read_csv(file)
	norm_vel_x_prev=df["ee_vel_x_prev"]
	norm_vel_y_prev=df["ee_vel_y_prev"]
	norm_vel_x_next=df["ee_vel_x_next"]
	norm_vel_y_next=df["ee_vel_y_next"]

	vel_x_prev=norm_vel_x_prev*(max_vel_x-min_vel_x)+min_vel_x
	vel_y_prev=norm_vel_y_prev*(max_vel_y-min_vel_y)+min_vel_y
	vel_x_next=norm_vel_x_next*(max_vel_x-min_vel_x)+min_vel_x
	vel_y_next=norm_vel_y_next*(max_vel_y-min_vel_y)+min_vel_y
	
	acc_x=df["cmd_acc_agent"]
	acc_y=df["cmd_acc_human"]

	prev_x=df["prev_x"]
	next_x=df["next_x"]
	prev_y=df["prev_y"]
	next_y=df["next_y"]

	episode=df["episode"]

	error_counter=0

	for i in range(len(prev_x)):
		if episode[i]==0:
			continue
		if (i-1)>=0:
			if episode[i-1]==0:
				continue
		if i-1<0:
			continue

		

		next_x_approx=prev_x[i]+vel_x_prev[i]*dt+0.5*acc_x[i]*dt*dt

		if next_x_approx > max_x:
			next_x_approx=max_x
		elif next_x_approx<min_x:
			next_x_approx=min_x


    	# error=difference/actual
		if (100*abs((next_x_approx-next_x[i])/next_x[i]))>15:
			print(i,next_x[i],next_x_approx,100*abs((next_x_approx-next_x[i])/next_x[i]))
			error_counter+=1
	print(error_counter,len(vel_x_prev))

	
	approx_acc_x=(vel_x_next-vel_x_prev)/dt
	approx_acc_y=(vel_y_next-vel_y_prev)/dt
	print("Acceleration")
	
	for i in range(len(approx_acc_x)):
		if episode[i]==0:
			continue
		if (i-1)>=0:
			if episode[i-1]==0:
				continue
		if i-1<0:
			continue		
		if acc_x[i]!=0:
			if (100*abs(approx_acc_x[i]-acc_x[i])/acc_x[i])>35:
				print(i,approx_acc_x[i],acc_x[i],100*abs(approx_acc_x[i]-acc_x[i])/acc_x[i])
		elif (acc_x[i]==0) and (approx_acc_x[i]!=0):
			if (approx_acc_x[i]-acc_x[i])>0.01:
				print(i,acc_x[i],approx_acc_x[i])


	
	
check_if_right("/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Paper_Results/games_info/AXG_12092025/data/rl_test_data_block_13.csv")