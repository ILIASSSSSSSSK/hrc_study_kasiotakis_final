import pandas as pd

predicted_accelerations_x=[]
predicted_accelaerations_y=[]

df=pd.read_csv("/home/kassiotakis/Desktop/Manos Old files/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert04entropy_LfD_TL_1/data/rl_test_data.csv")
dt=0.2

predicted_accelerations_x=(df["ee_vel_x_next"]-df["ee_vel_x_prev"])/dt
predicted_accelerations_y=(df["ee_vel_y_next"]-df["ee_vel_y_prev"])/dt

print(predicted_accelerations_x)
print(df["cmd_acc_agent"])
print(predicted_accelerations_y)
print(df["cmd_acc_human"])