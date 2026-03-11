import pandas as pd 
#df=pd.read_csv("/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_15/data/rl_test_data.csv")
files=[]
for f in range(1,27):
	files.append("/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_"+str(f)+"/data/rl_test_data.csv")


df=pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

ee_pos_x="ee_pos_x_prev"
ee_pos_y="ee_pos_y_prev"
ee_vel_x="ee_vel_x_prev"
ee_vel_y="ee_vel_y_prev"
max_posx=df[ee_pos_x].max()
max_posy=df[ee_pos_y].max()
max_velx=df[ee_vel_x].max()
max_vely=df[ee_vel_y].max()

min_posx=df[ee_pos_x].min()
min_posy=df[ee_pos_y].min()
min_velx=df[ee_vel_x].min()
min_vely=df[ee_vel_y].min()

mean_posx=df[ee_pos_x].mean()
mean_posy=df[ee_pos_y].mean()
mean_velx=df[ee_vel_x].mean()
mean_vely=df[ee_vel_y].mean()

print("ee_pos_x max: %f"%max_posx)
print("ee_pos_y max: %f"%max_posy)
print("ee_pos_x min: %f"%min_posx)
print("ee_pos_y min: %f"%min_posy)
print("ee_vel_x max: %f"%max_velx)
print("ee_vel_y max: %f"%max_vely)
print("ee_vel_x min: %f"%min_velx)
print("ee_vel_y min: %f"%min_vely)


print("ee_pos_x mean: %f"%mean_posx)
print("ee_pos_y mean: %f"%mean_posy)

print("ee_vel_x mean: %f"%mean_velx)
print("ee_vel_y mean: %f"%mean_vely)
