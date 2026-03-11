import pandas as pd
import matplotlib.pyplot as plt


csv_file_1 = "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_47/data/rl_test_data.csv"
csv_file_2 = "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_56/data/rl_test_data.csv"
csv_file_1_time="/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_47/data/test_data.csv"
csv_file_2_time="/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_56/data/test_data.csv"
column_name ='ee_vel_y_next' 
action='agent_action'
time='Episodes Duration in Seconds'
predicted_column1=[]
predicted_column2=[]

df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)

data1 = df1[column_name]
data2 = df2[column_name]
print(data1)
data1n = df1[column_name]
data2n = df2[column_name]
print(data1n)
time1=df1.iloc[:,2]#df1[time]
time2=df2.iloc[:,2]#df2[time]


time1=sum(time1)
time2=sum(time2)

time_step_1=0.2#time1/(len(data1))
time_step_2=0.2#time1/(len(data2))

a1=df1[action]
a2=df2[action]

x_max=0.012
x_min=-0.011

#unnormalize
data1_un=[]
data2_un=[]

for i in range(len(data1)):
    data1_un.append(data1[i]*(x_max-x_min)+x_min)
for i in range(len(data2)):
    data2_un.append(data2[i]*(x_max-x_min)+x_min)

#estimated velocity

vo=0
counter=1
for i in range(len(data1)):
    a=0
    if a1[i]==0:
        a=0
    elif a1[i]==1:
        a=0.2
    elif a1[i]==2:
        a=-0.2
    if abs(vo+time_step_1*a)<=0.2:
        predicted_column1.append(vo+time_step_1*a)
    elif (vo+time_step_1*a)>0.2:
        predicted_column1.append(0.2)
    elif (vo+time_step_1*a)<-0.2:
        predicted_column1.append(-0.2)

    vo=vo+time_step_1*a
    if data1n[i]==0:
        vo=0
        print('a')
        print(counter)
        print(i)
        counter+=1


vo=0
counter=1
for i in range(len(data2)):
    a=0

    if a2[i]==0:
        a=0

    elif a2[i]==1:
        a=0.2
    elif a2[i]==2:
        a=-0.2
    if abs(vo+time_step_2*a)<=0.2:
        predicted_column2.append(vo+time_step_2*a)
    elif (vo+time_step_2*a)>0.2:
        predicted_column2.append(0.2)
    elif (vo+time_step_2*a)<-0.2:
        predicted_column2.append(-0.2)
    vo=vo+time_step_2*a
    if data2n[i]==0:
        vo=0
        print(counter)
        counter+=1
plt.figure(figsize=(10, 6))
plt.plot(data2_un, label='variation 7, real_robot')
plt.plot(data1_un, label='variation 7, simulation')
#plt.plot(predicted_column2, label='variation 7, real_robot, estimation')
#plt.plot(predicted_column1, label='variation 7, simulation,estimation')
plt.xlabel('timestep')
plt.ylabel(column_name)
plt.title(f'Comparison of {column_name} between simulation and real robot')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
