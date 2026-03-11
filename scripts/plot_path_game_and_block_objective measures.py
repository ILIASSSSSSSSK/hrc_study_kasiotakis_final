import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.io as pio

data_normalized=True
normalized=True
max_x= -0.174 #-0.179
min_x= -0.356 #-0.359
max_y= 0.343
min_y= 0.162
ee_vel_x_max= 0.012
ee_vel_y_max= 0.07
ee_vel_x_min= -0.011
ee_vel_y_min=-0.014
#select the experiment 
file_rl_data=[#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_1/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_2/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/rl_test_data.csv"
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_16/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_19/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_37/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_38/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_39/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_40/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_41/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_31/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_32/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_33/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_34/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_35/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_36/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_46/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_45/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_44/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_43/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_42/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_2/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_1/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_68/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_69/data/rl_test_data.csv",
]
file_data=[#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_1/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_2/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/test_data.csv"
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_16/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_19/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_37/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_38/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_39/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_40/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_41/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_31/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_32/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_33/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_34/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_35/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_36/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_46/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_45/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_44/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_43/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_42/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_2/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_1/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_68/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_69/data/test_data.csv",
]

#comparison between different methods 
method_1_data=[
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_8/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_7/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/test_data.csv",    
]
method_2_data=[
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_2/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_1/data/test_data.csv",
]

method_3_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_25/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_26/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_27/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_28/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_29/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_30/data/test_data.csv"]

method_4_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_31/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_32/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_33/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_34/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_35/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_36/data/test_data.csv"]

method_5_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_41/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_37/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_38/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_39/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_40/data/test_data.csv"]

method_6_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_42/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_43/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_44/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_45/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_46/data/test_data.csv"]

method_7_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_50/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_49/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_48/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_47/data/test_data.csv",]

method_8_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_51/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_52/data/test_data.csv","/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_54/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_55/data/test_data.csv",]

method_9_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_60/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_57/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_58/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_59/data/test_data.csv",]
#real_robot_method_7=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/49K_every5_uniform_200ms_Christos_half_01_LfD_TL_41/data/test_data.csv","/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/49K_every5_uniform_200ms_Christos_half_01_LfD_TL_41/data/test_data.csv"]
real_robot_method_7=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_56/data/test_data.csv","/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_56/data/test_data.csv"]
real_robot_method_7_rl=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/49K_every5_uniform_200ms_Christos_half_01_LfD_TL_41/data/rl_test_data.csv","/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/49K_every5_uniform_200ms_Christos_half_01_LfD_TL_41/data/rl_test_data.csv"]

method_9_rl_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_60/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_57/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_58/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_59/data/rl_test_data.csv",]

method_8_rl_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_51/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_51/data/rl_test_data.csv","/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_54/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_55/data/rl_test_data.csv",
]

method_7_rl_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_50/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_49/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_48/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_47/data/rl_test_data.csv",]

method_1_rl_data=[
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_8/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_7/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/rl_test_data.csv",    
]
method_2_rl_data=[
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_2/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/35K_every10_uniform_200ms_itsmetheexpert_LfD_TL_1/data/rl_test_data.csv",
]

method_3_rl_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_25/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_26/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_27/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_28/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_29/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_30/data/rl_test_data.csv"]

method_4_rl_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_31/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_32/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_33/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_34/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_35/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_36/data/rl_test_data.csv"]

method_5_rl_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_41/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_37/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_38/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_39/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_40/data/rl_test_data.csv"]

method_6_rl_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_42/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_43/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_44/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_45/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_46/data/rl_test_data.csv"]

method_10_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Ilias_Experiments/games_info/Ilias_new_07102025/data/test_data_block_13.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Ilias_Experiments/games_info/Ilias_new2_14102025/data/test_data_block_13.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Ilias_Experiments/games_info/Ilias_new3_14102025/data/test_data_block_13.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Ilias_Experiments/games_info/Ilias_new4_14102025/data/test_data_block_13.csv"#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_69/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_68/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_67/data/test_data.csv",
]

method_10_rl_data=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Ilias_Experiments/games_info/Ilias_new_07102025/data/rl_test_data_block_13.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Ilias_Experiments/games_info/Ilias_new2_14102025/data/rl_test_data_block_13.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Ilias_Experiments/games_info/Ilias_new3_14102025/data/rl_test_data_block_13.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/Ilias_Experiments/games_info/Ilias_new4_14102025/data/rl_test_data_block_13.csv"#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_69/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_68/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_67/data/rl_test_data.csv",
]
#plot graph of trajectory for selected game and block

def plot_path_graph(files,block=2,game=10):
 num_games= len(files)
 rows,cols=0,0
 if len(files)==1:
 	rows,cols=1,1
 else:
 	if (len(files)==3) or (len(files)==2):
 		rows,cols=1,len(files)
 	elif(len(files)==4):
 		rows,cols=2,2
 	elif(len(files)==7):
 		rows=3
 		cols=3
 	else:
 		if len(files)%2==0:
 			cols=int(len(files)/2)
 			rows=cols
 		else:
 			cols=int(len(files)/2)
 			rows=int(len(files)/2)+1
 			if len(files)==9:
 				cols=3
 				rows=3

 fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
 if (cols==1) and (rows==1):
 	axes=[axes]
 else:
 	axes = axes.flatten()
 

 #select block (consider thet you start from 1 (ex 2nd block: 2))
 block=block
 #select the game you want to plot (consider thet you start from 1 (ex 2nd game: 2)) 
 game=game
 for j,file in enumerate(files):
 	df=pd.read_csv(file)
 	ee_pos_x_prev="ee_pos_x_prev" 
 	ee_pos_y_prev="ee_pos_y_prev"
 	ee_pos_x_next="ee_pos_x_next"
 	ee_pos_y_next="ee_pos_y_next"
 	
 	if data_normalized:
 	 	df[ee_pos_x_prev] = df[ee_pos_x_prev] * (max_x-min_x)+min_x
 	 	df[ee_pos_y_prev] = df[ee_pos_y_prev] *(max_y-min_y)+min_y
 	 	df[ee_pos_x_next] = df[ee_pos_x_next] * (max_x-min_x)+min_x
 	 	df[ee_pos_y_next] = df[ee_pos_y_next] *(max_y-min_y)+min_y
 	
 	g=0
 	all_games_indexes=[0]
 	positions_x=[]
 	positions_y=[]
 	for i in range(len(df[ee_pos_y_prev])):
 		if ((not normalized)and((df[ee_pos_x_prev][i]==(0.0))and(df[ee_pos_x_next][i]==(0.0))and(df[ee_pos_y_prev][i]==(0.0))and(df[ee_pos_y_next][i]==(0.0))))or((normalized)and((df[ee_pos_x_prev][i]==(0.0*(max_x-min_x)+min_x))and(df[ee_pos_x_next][i]==(0.0* (max_x-min_x)+min_x))and(df[ee_pos_y_prev][i]==(0.0* (max_y-min_y)+min_y))and(df[ee_pos_y_next][i]==(0.0* (max_y-min_y)+min_y)))):
 			all_games_indexes.append(i)
 	game_id=(block-1)*10+(game-1)
 	for i in range(all_games_indexes[game_id]+1,all_games_indexes[game_id+1]):
 		positions_x.append(df[ee_pos_x_prev][i])
 		positions_y.append(df[ee_pos_y_prev][i])
 		if i==(all_games_indexes[game_id+1]-1):
 			positions_x.append(df[ee_pos_x_next][i])
 			positions_y.append(df[ee_pos_y_next][i])
 	goal=[-0.265, 0.251]
 	goal_distance=0.01
 	#the first point is the final of the last game so ignore it (start from 1, not 0)
 	ax=axes[j]
 	
 	line=Line2D(positions_x[1:],positions_y[1:],color='blue')
 	ax.add_line(line)
 	#ax.scatter(positions_x[1:],positions_y[1:],color='blue',label='Intermediate Points')
 	ax.scatter(positions_x[1],positions_y[1],color='pink',label='Starting Point')
 	ax.scatter(positions_x[-1],positions_y[-1],color='green',label='Final Point')
 	ax.scatter(-0.346, 0.333,color='red')
 	ax.scatter(-0.345, 0.172,color='red')
 	ax.scatter(-0.185, 0.332,color='red')
 	ax.scatter(-0.184, 0.172,color='red')
 	ax.scatter(goal[0],goal[1],color='purple',label='Goal')
 	circle=patches.Circle(goal,goal_distance,color='purple',fill=False,linewidth=1)
 	ax.add_patch(circle)
 	ax.set_aspect('equal')
 	ax.set_xlabel('X')
 	ax.set_ylabel('Y')
 	ax.legend(fontsize="5",loc='upper right')
 	title=file.replace("/data/rl_test_data.csv","")
 	title2=""
 	if "expert" in title:
 		title2="Expert Experiment #"+title[-2]+title[-1]
 	elif "LfD_TL" in file:
 		title2="TL Experiment #"+title[-2]+title[-1]
 	elif "LfD_no_TL" in file:
 		title2="no_TL Experiment #"+title[-2]+title[-1]


 	ax.set_title(title2+(' Path for game %i in block %i'%(game,block)))
 	ax.set_xlim(-0.38,0)
 	ax.set_ylim(0.13,0.35)
 # Hide extra subplots (if any)
 for j in range(i + 1, len(axes)):
 	fig.delaxes(axes[j])

 # Adjust layout to prevent overlap
 
 plt.tight_layout()
 plt.show()

def plot_wins(files,compare,color="green",color_hex="#1f77b4",method="hello"):
	#calculate wins for each block
	wins_per_file=[]
	
	for file in files:
		
		df1=pd.read_csv(file)
		game_rew=df1["Rewards"]
		wins_per_block=[]

		for i in range(0,len(game_rew),10):
			wins=0
			for j in range(10):
				if game_rew[j+i]!=(-150):
					wins+=1
			wins_per_block.append(wins)
		
		wins_per_file.append(wins_per_block)
	
	wins_block_diagram=[]
	blocks=[]
	wins=np.zeros((len(wins_per_file),8))
	for i in range(len(wins_per_file)):
		for j in range(8):
			wins[i][j]=wins_per_file[i][j]
			wins_block_diagram.append(wins[i,j])
			blocks.append(j)            
	df2=pd.DataFrame({'Wins': wins_block_diagram, 'Blocks': blocks})
	
	avg_wins=np.mean(wins, axis=0)
	std_wins = np.std(wins, axis=0)
	print("Average Wins: ")
	print(avg_wins)
	print("Std Wins: ")
	print(std_wins)
	blocks=range(0,8)
	if compare:
		return df2["Wins"], df2["Blocks"], avg_wins, std_wins, blocks
	plt.plot(blocks, avg_wins, color=color, label='stim')
	plt.fill_between(blocks, avg_wins - std_wins, avg_wins + std_wins, color=color, alpha=0.3)
	plt.xlabel("Block")
	plt.ylabel("#wins")
	plt.title('Wins per Block')
	plt.grid(True)
	plt.ylim(0,11)
	plt.show()
	title="Average Wins per Block of "+method
	fig = px.box(df2, x="Blocks", y="Wins",title=title,color_discrete_sequence=[color_hex])
	# Define layout with y-axis range
	fig.update_layout(yaxis_range=[0, 11])
	plot(fig,filename='fig1.html',auto_open=True)

	#pio.show(fig)

def plot_rewards_and_norm_dist(files,compare,color_hex="#1f77b4",method=""):
	#norm_dist=tot_dist*tot_time/max_time
	
	rewards_per_file=[]
	rew_blocks_for_all=[]
	dist_per_file=[]
	dur_per_file=[]
	norm_dist_block_for_all=[]
	for file in files:
		df1=pd.read_csv(file)
		game_rew=df1["Rewards"]
		tot_dist=df1["Travelled Distance"]
		tot_time=df1["Episodes Duration in Seconds"]
		rewards_per_file.extend(game_rew)
		dist_per_file.extend(tot_dist)
		dur_per_file.extend(tot_time)
		
	
	max_time = 30.0
	print("Max time in secs: ")
	print(max_time)
	
	for i in range(0,80,10):
		rew_per_bl=[]
		norm_dist_per_bl=[]
		for j in range(0,len(rewards_per_file),80):
			for l in range(0,10):
				rew_per_bl.append(rewards_per_file[l+i+j]+150)
				norm_dist_per_bl.append(dist_per_file[l+i+j]*dur_per_file[l+i+j]/max_time)
				
		rew_blocks_for_all.append(rew_per_bl)
		norm_dist_block_for_all.append(norm_dist_per_bl)
		
	avg_rewards=[]
	std_rewards=[]
	avg_norm_dist=[]
	std_norm_dist=[]
	df3=[]
	df4=[]
	if len(files)==1:
		block_id=[]
		block=0
		for i in range(len(rew_blocks_for_all)):
			avg_r=np.mean(rew_blocks_for_all[i])
			std_r=np.std(rew_blocks_for_all[i])
			avg_nd=np.mean(norm_dist_block_for_all[i])
			std_nd=np.std(norm_dist_block_for_all[i])
			for j in range(len(rew_blocks_for_all[i])):
				block_id.append(block)
			avg_rewards.append(avg_r)
			std_rewards.append(std_r)
			avg_norm_dist.append(avg_nd)
			std_norm_dist.append(std_nd)
			block+=1
		#print(rew_blocks_for_all)
		#print(np.array(rew_blocks_for_all).flatten().tolist())
		df3=pd.DataFrame({'Rewards': np.array(rew_blocks_for_all).flatten().tolist(), 'Blocks': block_id})
		df4=pd.DataFrame({'Norm_Dist': np.array(norm_dist_block_for_all).flatten().tolist(), 'Blocks': block_id})
		if (not compare):
			fig3=px.box(df3, x="Blocks", y="Rewards",title="Average Reward per Block")
			fig3.update_layout(yaxis_range=[0, 165])
			plot(fig3,filename='fig2.html',auto_open=True)
		
			fig2 = px.box(df4, x="Blocks", y="Norm_Dist",title="Average Normalized Distance per Block")
			fig2.update_layout(yaxis_range=[0, 2])
			plot(fig2,filename='fig3.html',auto_open=True)

	else:
		rewards_block_diagram=[]
		norm_dist_block_diagram=[]
		block=0
		block_id=[]
		for i in range(len(rew_blocks_for_all)):
			avgs=[]
			avgs_nd=[]
			for j in range(0,len(rew_blocks_for_all[0]),10):								
				avgs.append(np.mean(rew_blocks_for_all[i][j:j+10]))
				avgs_nd.append(np.mean(norm_dist_block_for_all[i][j:j+10]))
				rewards_block_diagram.append(np.mean(rew_blocks_for_all[i][j:j+10]))
				norm_dist_block_diagram.append(np.mean(norm_dist_block_for_all[i][j:j+10]))
				block_id.append(block)
		    
			avg_rewards.append(np.mean(avgs))
			std_rewards.append(np.std(avgs))
			avg_norm_dist.append(np.mean(avgs_nd))
			std_norm_dist.append(np.std(avgs_nd))
			block+=1
		df3=pd.DataFrame({'Rewards': rewards_block_diagram, 'Blocks': block_id})
		df4=pd.DataFrame({'Norm_Dist': norm_dist_block_diagram, 'Blocks': block_id})
		if (not compare):
			fig3=px.box(df3, x="Blocks", y="Rewards",title="Average Reward per Block of "+method,color_discrete_sequence=[color_hex])
			fig3.update_layout(yaxis_range=[0, 165])
			plot(fig3,filename='fig2.html',auto_open=True)
		
			fig2 = px.box(df4, x="Blocks", y="Norm_Dist",title="Average Normalized Distance per Block of "+method,color_discrete_sequence=[color_hex])
			fig2.update_layout(yaxis_range=[0, 3])
			plot(fig2,filename='fig3.html',auto_open=True)
		
			





	print("")
	print("Average rewards: ")
	print(avg_rewards)
	print('Std: ')
	print(std_rewards)
	print("")
	print("Average Normalised Distance: ")
	print(avg_norm_dist)
	print("Std Normalised Distance: ")
	print(std_norm_dist)


	blocks=range(0,8)
	if compare:
		return df3["Rewards"],df3["Blocks"],df4["Norm_Dist"],df4["Blocks"],avg_rewards,std_rewards,avg_norm_dist,std_norm_dist
	plt.plot(blocks, avg_rewards, color='tab:blue', label='stim')
	plt.fill_between(blocks, np.array(avg_rewards) - np.array(std_rewards), np.array(avg_rewards) + np.array(std_rewards), color='tab:blue', alpha=0.3)
	plt.xlabel("Block")
	plt.ylabel("Reward")
	plt.title('Reward per Block')
	plt.grid(True)
	plt.ylim(0,165)
	plt.show()

	plt.plot(blocks, avg_norm_dist, color='tab:blue', label='stim')
	plt.fill_between(blocks, np.array(avg_norm_dist) - np.array(std_norm_dist), np.array(avg_norm_dist) + np.array(std_norm_dist), color='tab:blue', alpha=0.3)
	plt.xlabel("Block")
	plt.ylabel("Normalised Distance")
	plt.title('Normalised Distance per Block')
	plt.grid(True)
	plt.ylim(0,3)
	plt.show()

def plot_heatmap_with_coverage(batch_number, filepaths, steps_filepaths, games_per_batch=8, threshold=0, max_x=-0.18, min_x=-0.349, max_y=0.330, min_y=0.170, smoothing_sigma=0.3, ax=None):
    all_x_coords = []
    all_y_coords = []

    # Helper function to get game data
    def get_game_data(game_number, test_data, rl_data):
        if game_number < 0 or game_number >= len(test_data):
            raise ValueError("Invalid game number.")
        start_index = 0 if game_number == 0 else int(np.sum(test_data[:game_number, -1])) + game_number
        num_rows = int(test_data[game_number, -1])

        game_data = rl_data[start_index:start_index+num_rows, :]
        return game_data

    # Helper function to get batch data
    def get_batch_data(batch_number, test_data, rl_data, games_per_batch):
        start_game = batch_number * games_per_batch
        
        end_game = start_game + games_per_batch
        x_coords = []
        y_coords = []
        for game_num in range(start_game, end_game):
            game_data = get_game_data(game_num, test_data, rl_data)
            x_coords.extend(game_data[:, 2])
            y_coords.extend(game_data[:, 3])
        return x_coords, y_coords
    
    # Iterate over each participant
    for test_data, rl_data in zip(filepaths, steps_filepaths):
        x_coords, y_coords = get_batch_data(batch_number, test_data, rl_data, games_per_batch)
        all_x_coords.extend(x_coords)
        all_y_coords.extend(y_coords)
    # Create a 2D histogram for the heatmap

    heatmap, xedges, yedges = np.histogram2d(all_x_coords, all_y_coords, bins=[np.linspace(min_x, max_x, 20), np.linspace(min_y, max_y, 20)])
    total_bins = np.prod(heatmap.shape)
    filled_bins = np.nansum(heatmap > 0)
    coverage = filled_bins / total_bins
    heatmap_normalized = heatmap / np.max(heatmap)

    # Apply Gaussian smoothing to the heatmap
    #smoothed_heatmap = gaussian_filter(heatmap, smoothing_sigma)
    smoothed_heatmap = gaussian_filter(heatmap_normalized, smoothing_sigma)

    # Plotting the smoothed heatmap
    if ax is None:
       	plt.figure(figsize=(8, 8), facecolor='white')

    im=ax.imshow(smoothed_heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='YlGn', aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Counts')
    if ax is None:
        plt.colorbar(label='Counts')
        plt.title(f"Smoothed Heatmap of Positions in Batch {batch_number} ")
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        #plt.gca().set_facecolor('black')
    


    print(f"Coverage for Batch {batch_number}: {coverage:.2%}")

    return coverage  # Optionally return the coverage value

expert_test_data=[]
expert_steps_test_data=[]
expert_test_data2=[]
expert_steps_test_data2=[]
expert_test_data6=[]
expert_steps_test_data6=[]
expert_test_data3=[]
expert_steps_test_data3=[]
expert_test_data4=[]
expert_steps_test_data4=[]
expert_test_data5=[]
expert_steps_test_data5=[]
expert_test_data7=[]
expert_steps_test_data7=[]

expert_test_data8=[]
expert_steps_test_data8=[]
expert_test_data9=[]
expert_steps_test_data9=[]
expert_test_data_real_robot=[]
expert_steps_test_data_real_robot=[]

expert_test_data10=[]
expert_steps_test_data10=[]
"""
plot_path_graph(file_rl_data)

#plot just for one run or for one method. No comparison
plot_wins(real_robot_method_7,False)
plot_rewards_and_norm_dist(real_robot_method_7,False)
"""
#compare 2 different methods
wins_box_per_method=[]
blocks_box_per_method=[]
avg_wins_per_method=[]
std_wins_per_method=[]
blocks_per_method=[]
Rewards_box_per_method=[]
Blocks_box_r_per_method=[]
Norm_Dist_box_per_method=[]
Blocks_box_n_per_method=[]
avg_rewards_per_method=[]
std_rewards_per_method=[]
avg_norm_dist_per_method=[]
std_norm_dist_per_method=[]

methods_data=[
#method_3_data,method_4_data,method_5_data,method_6_data,method_2_data,method_1_data,
method_7_data,
#method_8_data,method_9_data,

#method_10_data,
real_robot_method_7]
for i in range(len(methods_data)):
	wins_box,blocks_box, avg_wins, std_wins, blocks=plot_wins(methods_data[i],True)
	Rewards_box,Blocks_box_r,Norm_Dist_box,Blocks_box_n,avg_rewards,std_rewards,avg_norm_dist,std_norm_dist=plot_rewards_and_norm_dist(methods_data[i],True)
	wins_box_per_method.append(wins_box)
	blocks_box_per_method.append(blocks_box)
	avg_wins_per_method.append(avg_wins)
	std_wins_per_method.append(std_wins)
	blocks_per_method.append(blocks)
	Rewards_box_per_method.append(Rewards_box)
	Blocks_box_r_per_method.append(Blocks_box_r)
	Norm_Dist_box_per_method.append(Norm_Dist_box)
	Blocks_box_n_per_method.append(Blocks_box_n)
	avg_rewards_per_method.append(avg_rewards)
	std_rewards_per_method.append(std_rewards)
	avg_norm_dist_per_method.append(avg_norm_dist)
	std_norm_dist_per_method.append(std_norm_dist)



methods_names=[#"Non Normalized data 98K, per 10, 4 states","Non Normalized data 98K, per 10, 4 states (loosing)",
#"Non Normalized data 98K, per 10, 2 states","Non Normalized data 98K, per 1, 2 states","Non Normalized data 35K, per 1, 2 states"
#,"Normalized data 35K, per 1, 4 states",
"Normalized data 98K, per 10, 4 states",
#"Normalized data 98K, per 10, 4 states, greedy",
#"Normalized data 98K, per 10, 2 states",
#"normalized data, per 10, 4 states, action from NN" ,
"real robot, normalized data, per 10, 4 states"]
colours=[
    #'#1f77b4',  # tab:blue
    #'#ff7f0e',  # tab:orange
    #'#2ca02c',  # tab:green
    #'#d62728',  # tab:red
    #'#9467bd',  # tab:purple
    #'#8c564b',  # tab:brown
    '#e377c2',  # tab:pink
    #'#7f7f7f',  # tab:gray
    #'#bcbd22',  # tab:olive
    #'#ffff14',	#tab:yellow
    '#17becf',  # tab:cyan

]

"""
for i in range(len(methods_data)):
	plot_wins(methods_data[i],False,color_hex=colours[i],method=methods_names[i])
	plot_rewards_and_norm_dist(methods_data[i],False,color_hex=colours[i],method=methods_names[i])
"""

data_plot_wins = []
data_plot_rewards = []
data_plot_norm_dist = []

for i in range(len(methods_names)):
    for reward, block in zip(Rewards_box_per_method[i], Blocks_box_r_per_method[i]):
        data_plot_rewards.append({
            "Reward": reward,
            "Block": block,
            "Method": methods_names[i]
        })
    for win, block in zip(wins_box_per_method[i], blocks_box_per_method[i]):
    	    data_plot_wins.append({
            	"Wins": win,
            	"Block": block,
            	"Method": methods_names[i]
        })
    for norm_dist, block in zip(Norm_Dist_box_per_method[i], Blocks_box_n_per_method[i]):
        data_plot_norm_dist.append({
            "Norm_Dist": norm_dist,
            "Block": block,
            "Method": methods_names[i]
        })
df_wins = pd.DataFrame(data_plot_wins)
df_rew = pd.DataFrame(data_plot_rewards)
df_nd = pd.DataFrame(data_plot_norm_dist)
fig_wins2 = px.box(
    df_wins,
    x="Block",
    y="Wins",
    color="Method",
    color_discrete_sequence=colours,
    title="Average wins per block"
)
fig_rew = px.box(
    df_rew,
    x="Block",
    y="Reward",
    color="Method",
    color_discrete_sequence=colours,
    title="Average reward per block"
)
fig_nd = px.box(
    df_nd,
    x="Block",
    y="Norm_Dist",
    color="Method",
    color_discrete_sequence=colours,
    title="Average Normalized Distance per block"
)
fig_wins2.update_layout(
    yaxis=dict(
        title="Average Wins",
        range=[0, 11],
        autorange=False,
        dtick=1
    )
)
fig_rew.update_layout(
    yaxis=dict(
        title="Average Reward",
        range=[0, 165],
        autorange=False
        
    )
)
fig_nd.update_layout(
    yaxis=dict(
        title="Average Normalized Distance",
        range=[0, 2],
        autorange=False,
        dtick=1
    )
)
plot(fig_wins2,filename='fig_wins2.html',auto_open=True)
plot(fig_rew,filename='fig_rew.html',auto_open=True)
plot(fig_nd,filename='fig_nd.html',auto_open=True)
real_robot_method_7=["/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/49K_every5_uniform_200ms_Christos_half_01_LfD_TL_41/data/test_data.csv","/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/49K_every5_uniform_200ms_Christos_half_01_LfD_TL_41/data/test_data.csv"]
for expert_test_data_file, expert_steps_test_data_file in zip(method_1_data, method_1_rl_data):
    a=np.loadtxt(expert_steps_test_data_file, delimiter=',', skiprows=1)
    #if positions are normalized, take the original
    for i in range(len(a)):
    	a[i][2]=(max_x-min_x)*a[i][2]+min_x
    	a[i][3]=(max_y-min_y)*a[i][3]+min_y
    	a[i][6]=(max_x-min_x)*a[i][6]+min_x
    	a[i][7]=(max_y-min_y)*a[i][7]+min_y
    expert_test_data.append(np.loadtxt(expert_test_data_file, delimiter=',', skiprows=1))
    expert_steps_test_data.append(a)
for expert_test_data_file2, expert_steps_test_data_file2 in zip(method_2_data, method_2_rl_data):
    a=np.loadtxt(expert_steps_test_data_file2, delimiter=',', skiprows=1)
    expert_test_data2.append(np.loadtxt(expert_test_data_file2, delimiter=',', skiprows=1))
    expert_steps_test_data2.append(a)
for expert_test_data_file6, expert_steps_test_data_file6 in zip(method_6_data, method_6_rl_data):
    a=np.loadtxt(expert_steps_test_data_file6, delimiter=',', skiprows=1)
    expert_test_data6.append(np.loadtxt(expert_test_data_file6, delimiter=',', skiprows=1))
    expert_steps_test_data6.append(a)

batches_to_collect = [0, 3, 5]
col_labels = ["Baseline", "Block 3", "Block 6"]
row_labels = ["Normalized 35K, 4s, per 1","Non Normalized 35K, 2s, per 1","Non Normalized 98K, 2s, per 1"]
num_rows=len(row_labels)
num_cols=len(col_labels)
fig = plt.figure(figsize=(14, 14))
for row in range(num_rows):
    for col in range(num_cols):
        subplot_idx = row * num_cols + col + 1
        ax = fig.add_subplot(num_rows, num_cols, subplot_idx)

        # Determine the group based on the row (0 for Experts, 1 for TL, 2 for No TL)
        group_idx = row

        # Get the batch number based on the column
        batch_number = batches_to_collect[col]

        # Construct a title based on the labels and batch number
        title = f"{row_labels[group_idx]} - {col_labels[col]}"

        if group_idx == 0:
            # Expert
            plot_heatmap_with_coverage(batch_number, expert_test_data, expert_steps_test_data, ax=ax)
        elif group_idx == 1:
            # TL Participant
            plot_heatmap_with_coverage(batch_number, expert_test_data2, expert_steps_test_data2, ax=ax)
        elif group_idx == 2:
            # No TL Participant
            plot_heatmap_with_coverage(batch_number, expert_test_data6, expert_steps_test_data6, ax=ax)
        

        ax.set_title(title)

# Adjust the layout
plt.tight_layout()

# Show the figure
plt.show()
for expert_test_data_file3, expert_steps_test_data_file3 in zip(method_3_data, method_3_rl_data):
    a=np.loadtxt(expert_steps_test_data_file3, delimiter=',', skiprows=1)
    expert_test_data3.append(np.loadtxt(expert_test_data_file3, delimiter=',', skiprows=1))
    expert_steps_test_data3.append(a)
for expert_test_data_file4, expert_steps_test_data_file4 in zip(method_4_data, method_4_rl_data):
    a=np.loadtxt(expert_steps_test_data_file4, delimiter=',', skiprows=1)
    expert_test_data4.append(np.loadtxt(expert_test_data_file4, delimiter=',', skiprows=1))
    expert_steps_test_data4.append(a)
for expert_test_data_file5, expert_steps_test_data_file5 in zip(method_5_data, method_5_rl_data):
    a=np.loadtxt(expert_steps_test_data_file5, delimiter=',', skiprows=1)
    expert_test_data5.append(np.loadtxt(expert_test_data_file5, delimiter=',', skiprows=1))
    expert_steps_test_data5.append(a)
batches_to_collect = [0, 3, 5]
col_labels = ["Baseline", "Block 3", "Block 6"]
row_labels = ["Non Normalized 98K, 4s, per 10","Non Normalized 98K, 4s, per 10(loosing)","Non Normalized 98K, 2s, per 10"]
num_rows=len(row_labels)
num_cols=len(col_labels)
fig = plt.figure(figsize=(14, 14))
for row in range(num_rows):
    for col in range(num_cols):
        subplot_idx = row * num_cols + col + 1
        ax = fig.add_subplot(num_rows, num_cols, subplot_idx)

        # Determine the group based on the row (0 for Experts, 1 for TL, 2 for No TL)
        group_idx = row

        # Get the batch number based on the column
        batch_number = batches_to_collect[col]

        # Construct a title based on the labels and batch number
        title = f"{row_labels[group_idx]} - {col_labels[col]}"

        if group_idx == 0:
            # Expert
            plot_heatmap_with_coverage(batch_number, expert_test_data3, expert_steps_test_data3, ax=ax)
        elif group_idx == 1:
            # TL Participant
            plot_heatmap_with_coverage(batch_number, expert_test_data4, expert_steps_test_data4, ax=ax)
        elif group_idx == 2:
            # No TL Participant
            plot_heatmap_with_coverage(batch_number, expert_test_data5, expert_steps_test_data5, ax=ax)
        

        ax.set_title(title)

# Adjust the layout
plt.tight_layout()

# Show the figure
plt.show()
for expert_test_data_file7, expert_steps_test_data_file7 in zip(method_7_data, method_7_rl_data):
    a=np.loadtxt(expert_steps_test_data_file7, delimiter=',', skiprows=1)
    #if positions are normalized, take the original
    for i in range(len(a)):
    	a[i][2]=(max_x-min_x)*a[i][2]+min_x
    	a[i][3]=(max_y-min_y)*a[i][3]+min_y
    	a[i][6]=(max_x-min_x)*a[i][6]+min_x
    	a[i][7]=(max_y-min_y)*a[i][7]+min_y
    expert_test_data7.append(np.loadtxt(expert_test_data_file7, delimiter=',', skiprows=1))
    expert_steps_test_data7.append(a)

for expert_test_data_file8, expert_steps_test_data_file8 in zip(method_8_data, method_8_rl_data):
    a=np.loadtxt(expert_steps_test_data_file8, delimiter=',', skiprows=1)
    #if positions are normalized, take the original
    for i in range(len(a)):
    	a[i][2]=(max_x-min_x)*a[i][2]+min_x
    	a[i][3]=(max_y-min_y)*a[i][3]+min_y
    	a[i][6]=(max_x-min_x)*a[i][6]+min_x
    	a[i][7]=(max_y-min_y)*a[i][7]+min_y
    expert_test_data8.append(np.loadtxt(expert_test_data_file8, delimiter=',', skiprows=1))
    expert_steps_test_data8.append(a)

for expert_test_data_file9, expert_steps_test_data_file9 in zip(method_9_data, method_9_rl_data):
    a=np.loadtxt(expert_steps_test_data_file9, delimiter=',', skiprows=1)
    #if positions are normalized, take the original
    for i in range(len(a)):
    	a[i][2]=(max_x-min_x)*a[i][2]+min_x
    	a[i][3]=(max_y-min_y)*a[i][3]+min_y
    	a[i][6]=(max_x-min_x)*a[i][6]+min_x
    	a[i][7]=(max_y-min_y)*a[i][7]+min_y
    expert_test_data9.append(np.loadtxt(expert_test_data_file9, delimiter=',', skiprows=1))
    expert_steps_test_data9.append(a)

for expert_test_data_file_real_robot, expert_steps_test_data_file_real_robot in zip(real_robot_method_7, real_robot_method_7_rl):
    a=np.loadtxt(expert_steps_test_data_file_real_robot, delimiter=',', skiprows=1)
    #if positions are normalized, take the original
    for i in range(len(a)):
    	a[i][2]=(max_x-min_x)*a[i][2]+min_x
    	a[i][3]=(max_y-min_y)*a[i][3]+min_y
    	a[i][6]=(max_x-min_x)*a[i][6]+min_x
    	a[i][7]=(max_y-min_y)*a[i][7]+min_y
    expert_test_data_real_robot.append(np.loadtxt(expert_test_data_file_real_robot, delimiter=',', skiprows=1))
    expert_steps_test_data_real_robot.append(a)

for expert_test_data_file_10, expert_steps_test_data_file_10 in zip(method_10_data, method_10_rl_data):
    a=np.genfromtxt(expert_steps_test_data_file_10, delimiter=',', skip_header=1, dtype=None, encoding='utf-8')
    #if positions are normalized, take the original
    for i in range(len(a)):
    	a[i][2]=(max_x-min_x)*a[i][2]+min_x
    	a[i][3]=(max_y-min_y)*a[i][3]+min_y
    	a[i][6]=(max_x-min_x)*a[i][6]+min_x
    	a[i][7]=(max_y-min_y)*a[i][7]+min_y
    expert_test_data10.append(np.loadtxt(expert_test_data_file_10, delimiter=',', skiprows=1))
    expert_steps_test_data10.append(a)

col_labels = ["Baseline", "Block 3", "Block 6"]
row_labels = ["Normalized 98K, 4s, per 10","Normalized 98K, 4s, per 10, greedy","Normalized 98K, 2s, per 10"]
num_rows=len(row_labels)
num_cols=len(col_labels)
fig = plt.figure(figsize=(14, 14))
for row in range(num_rows):
    for col in range(num_cols):
        subplot_idx = row * num_cols + col + 1
        ax = fig.add_subplot(num_rows, num_cols, subplot_idx)

        # Determine the group based on the row (0 for Experts, 1 for TL, 2 for No TL)
        group_idx = row

        # Get the batch number based on the column
        batch_number = batches_to_collect[col]

        # Construct a title based on the labels and batch number
        title = f"{row_labels[group_idx]} - {col_labels[col]}"

        if group_idx == 0:
            # Expert
            plot_heatmap_with_coverage(batch_number, expert_test_data7, expert_steps_test_data7, ax=ax)
        elif group_idx == 1:
            # TL Participant
            plot_heatmap_with_coverage(batch_number, expert_test_data8, expert_steps_test_data8, ax=ax)
        elif group_idx == 2:
            # No TL Participant
            plot_heatmap_with_coverage(batch_number, expert_test_data9, expert_steps_test_data9, ax=ax)
        

        ax.set_title(title)
col_labels = ["Baseline", "Block 3", "Block 6"]
row_labels = ["Normalized 75K, 4s, per 8","Normalized 75K, 4s, per 8, NN's action"]
num_rows=len(row_labels)
num_cols=len(col_labels)
fig = plt.figure(figsize=(12, 4))
for row in range(num_rows):
    for col in range(num_cols):
        subplot_idx = row * num_cols + col + 1
        ax = fig.add_subplot(num_rows, num_cols, subplot_idx)

        # Determine the group based on the row (0 for Experts, 1 for TL, 2 for No TL)
        group_idx = row

        # Get the batch number based on the column
        batch_number = batches_to_collect[col]

        # Construct a title based on the labels and batch number
        title = f"{row_labels[group_idx]} - {col_labels[col]}"

        if group_idx == 0:
            # Expert
            plot_heatmap_with_coverage(batch_number, expert_test_data_real_robot, expert_steps_test_data_real_robot, games_per_batch=5,ax=ax)
        elif group_idx == 1:
            # TL Participant
            plot_heatmap_with_coverage(batch_number, expert_test_data10, expert_steps_test_data10, ax=ax)
        elif group_idx == 2:
            # No TL Participant
            plot_heatmap_with_coverage(batch_number, expert_test_data9, expert_steps_test_data9, ax=ax)
        

        ax.set_title(title)

# Adjust the layout
plt.tight_layout()

# Show the figure
plt.show()
