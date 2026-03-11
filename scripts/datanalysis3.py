import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load data for multiple participants # for the first 3 plots WINS REWARDS TRAVELLED DISTANCE we can have more than 1 filepath 
TL_test_filepaths = [
    "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_7/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_8/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_9/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_10/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_11/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_12/data/test_data.csv"
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/test_data.csv'
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/test_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/test_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/test_data.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXR_LfD_TL_1/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXM_LfD_TL_2/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_GXL_LfD_TL_1/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_MXS_LfD_TL_2/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_XXK_LfD_TL_1/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXP_LfD_TL_1/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_SXT_LfD_TL_1/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_PXG_LfD_TL_1/data/test_data.csv'
]
TL_train_filepaths = [
     "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_7/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_8/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_9/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_10/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_11/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_12/data/test_data.csv"
     #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/data.csv'
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/data.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXR_LfD_TL_1/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXM_LfD_TL_2/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_GXL_LfD_TL_1/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_MXS_LfD_TL_2/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_XXK_LfD_TL_1/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXP_LfD_TL_1/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_SXT_LfD_TL_1/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_PXG_LfD_TL_1/data/data.csv'
]
No_TL_test_filepaths = [
    "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_7/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_8/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_9/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_10/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_11/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_12/data/test_data.csv"
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/test_data.csv'
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/test_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/test_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/test_data.csv'

    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXI_no_TL_1/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXM_no_TL_1/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_VXC_no_TL_2/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXS_no_TL_1/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_FXS_no_TL_1/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXT_no_TL_1/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXAXG_no_TL_1/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXG_no_TL_1/data/test_data.csv'
]
No_TL_train_filepaths = [
    "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_7/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_8/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_9/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_10/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_11/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_12/data/test_data.csv"
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/data.csv'
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/data.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXI_no_TL_1/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXM_no_TL_1/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_VXC_no_TL_2/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXS_no_TL_1/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_FXS_no_TL_1/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXT_no_TL_1/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXAXG_no_TL_1/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXG_no_TL_1/data/data.csv'
]

TL_steps_test_filepaths = [
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_7/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_8/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_9/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_10/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_11/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_12/data/rl_test_data.csv"
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/rl_test_data.csv'
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/rl_test_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/rl_test_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/rl_test_data.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXR_LfD_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXM_LfD_TL_2/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_GXL_LfD_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXEXN_LfD_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_MXS_LfD_TL_2/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_XXK_LfD_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXP_LfD_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_SXT_LfD_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_PXG_LfD_TL_1/data/rl_test_data.csv'
]
TL_steps_train_filepaths = [
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/rl_data.csv',
    "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_7/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_8/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_9/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_10/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_11/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_12/data/rl_test_data.csv"
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/rl_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/rl_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/rl_data.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXR_LfD_TL_1/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXM_LfD_TL_2/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_GXL_LfD_TL_1/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXEXN_LfD_TL_1/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_MXS_LfD_TL_2/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_XXK_LfD_TL_1/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXP_LfD_TL_1/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_SXT_LfD_TL_1/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_PXG_LfD_TL_1/data/rl_data.csv'
]
No_TL_steps_test_filepaths = [
    "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_7/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_8/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_9/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_10/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_11/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_12/data/rl_test_data.csv"
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/rl_test_data.csv'
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/rl_test_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/rl_test_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/rl_test_data.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXI_no_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXM_no_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_VXC_no_TL_2/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXS_no_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_FXS_no_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXT_no_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXAXG_no_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXG_no_TL_1/data/rl_test_data.csv'
]
No_TL_steps_train_filepaths = [
    "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_7/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_8/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_9/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_10/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_11/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_12/data/rl_test_data.csv"
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/rl_data.csv'
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/rl_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/rl_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/rl_data.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXI_no_TL_1/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXM_no_TL_1/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_VXC_no_TL_2/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXS_no_TL_1/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_FXS_no_TL_1/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXT_no_TL_1/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXAXG_no_TL_1/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXG_no_TL_1/data/rl_data.csv'
]
experts_test_filepaths = [
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expertdimitrisentropy_LfD_TL_4/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_EXPERT80ep_LfD_TL_1/data/test_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/test_data.csv',
    "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_7/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_8/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_9/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_10/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_11/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_12/data/test_data.csv"
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/test_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/test_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/test_data.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert05entropygood_LfD_TL_2/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert01ntropy_LfD_TL_2/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert01ntropy_LfD_TL_1/data/test_data.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/initialized_agents/ExpertStavrouFinal/data/test_data.csv'

    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert04entropy_LfD_TL_1/data/test_data.csv',
    
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert06entropy_LfD_TL_2/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/20K_every10_uniform_200ms_expert0.75we4000_LfD_TL_1/data/test_data.csv',
    ##'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.75we_LfD_TL_1/data/test_data.csv',
    ####'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.98we_LfD_TL_1/data/test_data.csv'


]
experts_train_filepaths = [
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expertdimitrisentropy_LfD_TL_4/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_EXPERT80ep_LfD_TL_1/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert05entropygood_LfD_TL_2/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert01ntropy_LfD_TL_2/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert01ntropy_LfD_TL_1/data/data.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/initialized_agents/ExpertStavrouFinal/data/data.csv'
    
    "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_7/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_8/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_9/data/test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_10/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_11/data/test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_12/data/test_data.csv"
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/test_data.csv'
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/test_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/test_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/test_data.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert04entropy_LfD_TL_1/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert06entropy_LfD_TL_2/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/20K_every10_uniform_200ms_expert0.75we4000_LfD_TL_1/data/data.csv',
    ##'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.75we_LfD_TL_1/data/data.csv',
    ##'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.98we_LfD_TL_1/data/data.csv'
]
experts_steps_test_filepaths = [
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expertdimitrisentropy_LfD_TL_4/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_EXPERT80ep_LfD_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert05entropygood_LfD_TL_2/data/rl_test_data.csv'
    "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_7/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_8/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_9/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_10/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_11/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_12/data/rl_test_data.csv"
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/rl_test_data.csv'
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/rl_test_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/rl_test_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/rl_test_data.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert01ntropy_LfD_TL_2/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert01ntropy_LfD_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert04entropy_LfD_TL_1/data/rl_test_data.csv',
    ####'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/initialized_agents/ExpertStavrouFinal/data/rl_data.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert06entropy_LfD_TL_2/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/20K_every10_uniform_200ms_expert0.75we4000_LfD_TL_1/data/rl_test_data.csv',
    ##'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.75we_LfD_TL_1/data/rl_test_data.csv',
    ####'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.98we_LfD_TL_1/data/rl_test_data.csv'


]
experts_steps_train_filepaths = [
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expertdimitrisentropy_LfD_TL_4/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_EXPERT80ep_LfD_TL_1/data/rl_data.csv',
    "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_7/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_8/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_9/data/rl_test_data.csv",
#"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_10/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_11/data/rl_test_data.csv",
"/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_12/data/rl_test_data.csv"
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_6/data/rl_data.csv'
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_3/data/rl_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_4/data/rl_data.csv',
    #'/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/games_info/98K_every10_uniform_200ms_itsmetheexpert_LfD_TL_5/data/rl_data.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert05entropygood_LfD_TL_2/data/rl_data.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert01ntropy_LfD_TL_2/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert01ntropy_LfD_TL_1/data/rl_data.csv',
    ###'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/initialized_agents/ExpertStavrouFinal/data/rl_data.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert04entropy_LfD_TL_1/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert06entropy_LfD_TL_2/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/20K_every10_uniform_200ms_expert0.75we4000_LfD_TL_1/data/rl_data.csv'
    ##'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.75we_LfD_TL_1/data/rl_data.csv',
    ##'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.98we_LfD_TL_1/data/rl_data.csv',

]
experts_entropy_filepaths = [
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expertdimitrisentropy_LfD_TL_4/data/entropy.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_EXPERT80ep_LfD_TL_1/data/entropy.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert05entropy_LfD_TL_2/data/entropy.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert01ntropy_LfD_TL_2/data/entropy.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert01ntropy_LfD_TL_1/data/entropy.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert04entropy_LfD_TL_1/data/entropy.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/20K_every10_uniform_200ms_expert0.75we4000_LfD_TL_1/data/entropy.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert06entropy_LfD_TL_2/data/entropy.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.75we_LfD_TL_1/data/entropy.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.98we_LfD_TL_1/data/entropy.csv',
]
TL_test_data=[]
TL_train_data=[]
TL_steps_test_data=[]
TL_steps_train_data=[]
NO_TL_test_data=[]
NO_TL_train_data=[]
NO_TL_steps_test_data=[]
NO_TL_steps_train_data=[]
expert_test_data=[]
expert_train_data=[]
expert_steps_test_data=[]
expert_steps_train_data=[]
expert_entropy_data=[]

# Correcting the data loading loop
for expert_test_data_file, expert_train_data_file, expert_steps_test_data_file, expert_steps_train_data_file in zip(experts_test_filepaths, experts_train_filepaths, experts_steps_test_filepaths, experts_steps_train_filepaths):
    expert_test_data.append(np.loadtxt(expert_test_data_file, delimiter=',', skiprows=1))
    expert_train_data.append(np.loadtxt(expert_train_data_file, delimiter=',', skiprows=1))
    expert_steps_test_data.append(np.loadtxt(expert_steps_test_data_file, delimiter=',', skiprows=1))
    expert_steps_train_data.append(np.loadtxt(expert_steps_train_data_file, delimiter=',', skiprows=1))
    #expert_entropy_data.append(np.loadtxt(expert_entropy_file, delimiter=',', skiprows=1))

for entropy_filepath in experts_entropy_filepaths:
    data = np.loadtxt(entropy_filepath, delimiter=',', skiprows=1)
    expert_entropy_data.append(data)




for TL_test_data_files, TL_train_data_files, NO_TL_test_data_files, NO_TL_train_data_files in zip(TL_test_filepaths,TL_train_filepaths, No_TL_test_filepaths,No_TL_train_filepaths):
    TL_test_data.append(np.loadtxt(TL_test_data_files, delimiter=',', skiprows=1))
    TL_train_data.append(np.loadtxt(TL_train_data_files, delimiter=',', skiprows=1))
    NO_TL_test_data.append(np.loadtxt(NO_TL_test_data_files, delimiter=',', skiprows=1))
    NO_TL_train_data.append(np.loadtxt(NO_TL_train_data_files, delimiter=',', skiprows=1))

for TL_steps_test_data_files, TL_steps_train_data_files, NO_TL_steps_test_data_files, NO_TL_steps_train_data_files in zip(TL_steps_test_filepaths,TL_steps_train_filepaths, No_TL_steps_test_filepaths,No_TL_steps_train_filepaths):
    TL_steps_test_data.append(np.loadtxt(TL_steps_test_data_files, delimiter=',', skiprows=1))
    TL_steps_train_data.append(np.loadtxt(TL_steps_train_data_files, delimiter=',', skiprows=1))
    NO_TL_steps_test_data.append(np.loadtxt(NO_TL_steps_test_data_files, delimiter=',', skiprows=1))
    NO_TL_steps_train_data.append(np.loadtxt(NO_TL_steps_train_data_files, delimiter=',', skiprows=1))



TL_train_data = [participant_data[:, 1:] for participant_data in TL_train_data]
NO_TL_train_data = [participant_data[:, 1:] for participant_data in NO_TL_train_data]
expert_train_data= [participant_data[:, 1:] for participant_data in expert_train_data]


def calculate_wins(data_group):
    all_participant_wins = []
    all_participant_rewards = []
    for participant_data in data_group:
        win_counts = []  
        rewards_per_block = []
        for i in range(0, participant_data.shape[0], 10):
            block = participant_data[i:i+10, 0]
            block_rewards = 150 + block
            block_reward = np.mean(block_rewards)
            rewards_per_block.append(block_reward)
            win_count = np.sum(block != -150)
            win_counts.append(win_count)
        all_participant_rewards.append(rewards_per_block)    
        all_participant_wins.append(win_counts)
    return all_participant_wins, all_participant_rewards

def calculate_mean_normalized_distances(data_group):
    all_participant_mean_normalized_distances = []
    for participant_data in data_group:
        mean_normalized_distances_per_participant = []
        for i in range(0, participant_data.shape[0], 10):
            block = participant_data[i:i+10]
            print(block)
            max_duration = np.max(block[:, 1])
            #max_duration=30.0
            normalized_distances_block = (block[:, 1] / max_duration) * block[:, 2]
            mean_normalized_distance = np.mean(normalized_distances_block)
            mean_normalized_distances_per_participant.append(mean_normalized_distance)
        all_participant_mean_normalized_distances.append(mean_normalized_distances_per_participant)
    return all_participant_mean_normalized_distances

def calculate_stats(metrics_list):
    min_blocks = min(len(metrics) for metrics in metrics_list)
    truncated_metrics = [metrics[:min_blocks] for metrics in metrics_list]
    mean_metrics = np.mean(truncated_metrics, axis=0)
    std_dev_metrics = np.std(truncated_metrics, axis=0, ddof=0)  # Set ddof to 0
    return mean_metrics, std_dev_metrics


def plot_metrics(metrics_list, title, label, y_label):
    mean_metrics, std_dev_metrics = calculate_stats(metrics_list)
    blocks = np.arange(1, len(mean_metrics) + 1)
    plt.errorbar(blocks, mean_metrics, yerr=std_dev_metrics, fmt='o-', label=label)
    plt.xlabel('Block Number')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
#################
# Calculate wins, rewards, and mean normalized distances for each group
TL_test_wins, TL_test_rewards = calculate_wins(TL_test_data)
TL_train_wins, TL_train_rewards = calculate_wins(TL_train_data)
NO_TL_test_wins, NO_TL_test_rewards = calculate_wins(NO_TL_test_data)
NO_TL_train_wins, NO_TL_train_rewards = calculate_wins(NO_TL_train_data)

TL_test_distance = calculate_mean_normalized_distances(TL_test_data)
TL_train_distance = calculate_mean_normalized_distances(TL_train_data)
NO_TL_test_distance = calculate_mean_normalized_distances(NO_TL_test_data)
NO_TL_train_distance = calculate_mean_normalized_distances(NO_TL_train_data)
# Plotting Wins, Rewards, and Traveled Distances for each group

# Calculate wins and rewards for the expert data
expert_wins_test, expert_rewards_test = calculate_wins(expert_test_data)
expert_wins_train, expert_rewards_train = calculate_wins(expert_train_data)

# Calculate distances for the expert data
expert_distance_test = calculate_mean_normalized_distances(expert_test_data)
expert_distance_train = calculate_mean_normalized_distances(expert_train_data)

def plot_combined_metrics(tl_data, no_tl_data, expert_data, title, label1, label2, label3, y_label):
    mean_tl, std_dev_tl = calculate_stats(tl_data)
    mean_no_tl, std_dev_no_tl = calculate_stats(no_tl_data)

    if expert_data:  # Check if expert data is provided
        mean_expert, std_dev_expert = calculate_stats(expert_data)
        blocks = np.arange(1, len(mean_expert) + 1)
        plt.errorbar(blocks, mean_expert, yerr=std_dev_expert, fmt='o-', label=label3)
    print(mean_tl)
    print(mean_no_tl)
    #print(mean_expert)

    blocks = np.arange(1, len(mean_tl) + 1)
    plt.errorbar(blocks, mean_tl, yerr=std_dev_tl, fmt='o-', label=label1)
    plt.errorbar(blocks, mean_no_tl, yerr=std_dev_no_tl, fmt='o-', label=label2)
    
    plt.xlabel('Block Number')
    plt.ylabel(y_label)
    plt.title(title)
    
    # Move the legend outside the figure to the right
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Modify the x-axis tick labels
    plt.xticks(blocks, ['Baseline'] + list(blocks[1:]))  # Replace the first label with 'Baseline'
    
    plt.grid(True)

# Assuming you have calculated the expert wins, rewards, and distances for both test and train data
######################################################################BOTH GROUPS PLOTS###############################
# Plotting Combined Wins for TL, No TL, and Expert Test Groups
print(expert_wins_test)
plt.figure(figsize=(12, 6))

plot_combined_metrics(TL_test_wins, NO_TL_test_wins, False, 'Wins per Test Block ', 'TL', 'No TL', '', 'Wins')
#plt.show()

# Plotting Combined Rewards for TL, No TL, and Expert Test Groups
plt.figure(figsize=(12, 6))

plot_combined_metrics(TL_test_rewards, NO_TL_test_rewards, expert_rewards_test, 'Rewards per Test Block ', 'TL', 'No TL', 'Expert ', 'Rewards')
#plt.show()

# Plotting Combined Normalized Traveled Distance for TL, No TL, and Expert Test Groups
plt.figure(figsize=(12, 6))
plot_combined_metrics(TL_test_distance, NO_TL_test_distance, expert_distance_test, 'Normalized Travelled Distance per Test Block ', 'TL', 'No TL', 'Expert ', 'Normalized Travelled Distance ')
#plt.show()

######################################TIME########################
"""
def calculate_group_total_time(data_group):
    group_total_time = []
    for participant_data in data_group:
        total_time_per_participant = np.sum(participant_data[:, 1])
        group_total_time.append(total_time_per_participant)
    return group_total_time

# Calculate group mean total time and standard deviation for each group in seconds
mean_TL_test_total_time_secs = np.mean(calculate_group_total_time(TL_test_data))
std_TL_test_total_time_secs = np.std(calculate_group_total_time(TL_test_data), ddof=0)  # Set ddof to 0 for population std deviation
mean_NO_TL_test_total_time_secs = np.mean(calculate_group_total_time(NO_TL_test_data))
std_NO_TL_test_total_time_secs = np.std(calculate_group_total_time(NO_TL_test_data), ddof=0)

mean_TL_train_total_time_secs = np.mean(calculate_group_total_time(TL_train_data))
std_TL_train_total_time_secs = np.std(calculate_group_total_time(TL_train_data), ddof=0)
mean_NO_TL_train_total_time_secs = np.mean(calculate_group_total_time(NO_TL_train_data))
std_NO_TL_train_total_time_secs = np.std(calculate_group_total_time(NO_TL_train_data), ddof=0)

# Convert mean and std to minutes
mean_TL_test_total_time_mins = mean_TL_test_total_time_secs / 60
std_TL_test_total_time_mins = std_TL_test_total_time_secs / 60
mean_NO_TL_test_total_time_mins = mean_NO_TL_test_total_time_secs / 60
std_NO_TL_test_total_time_mins = std_NO_TL_test_total_time_secs / 60

mean_TL_train_total_time_mins = mean_TL_train_total_time_secs / 60
std_TL_train_total_time_mins = std_TL_train_total_time_secs / 60
mean_NO_TL_train_total_time_mins = mean_NO_TL_train_total_time_secs / 60
std_NO_TL_train_total_time_mins = std_NO_TL_train_total_time_secs / 60

# Print the calculated group means and standard deviations in minutes
print("TL Test Mean Total Time (mins):", mean_TL_test_total_time_mins)
print("TL Test Std Total Time (mins):", std_TL_test_total_time_mins)
print("NO TL Test Mean Total Time (mins):", mean_NO_TL_test_total_time_mins)
print("NO TL Test Std Total Time (mins):", std_NO_TL_test_total_time_mins)

print("TL Train Mean Total Time (mins):", mean_TL_train_total_time_mins)
print("TL Train Std Total Time (mins):", std_TL_train_total_time_mins)
print("NO TL Train Mean Total Time (mins):", mean_NO_TL_train_total_time_mins)
print("NO TL Train Std Total Time (mins):", std_NO_TL_train_total_time_mins)

"""
###########################COMBINED######################
"""
def plot_combined_bar_and_print_means(metrics_test, metrics_train, title, ylabel):
    # Ensure the number of batches (5 batches in this case)
    num_batches = 5

    # Initialize lists for mean metrics and standard deviations
    mean_metrics = []
    std_metrics = []

    # Initialize a list to store the mean for each block for each participant
    participant_means = []

    # Extract the first test block for Baseline and calculate its mean and std
    baseline_test_block = [participant[0] for participant in metrics_test]
    mean_metrics.append(np.mean(baseline_test_block))
    std_metrics.append(np.std(baseline_test_block, ddof=1))
    participant_means.append(baseline_test_block)

    # Add pairs of train and test blocks for each batch
    for i in range(num_batches):
        # Add training block if available
        if i < len(metrics_train):
            train_block = [participant[i] for participant in metrics_train]
            mean_metrics.append(np.mean(train_block))
            std_metrics.append(np.std(train_block, ddof=1))
            participant_means.append(train_block)
        # Add subsequent test block if available
        if i < len(metrics_test) - 1:
            test_block = [participant[i + 1] for participant in metrics_test]
            mean_metrics.append(np.mean(test_block))
            std_metrics.append(np.std(test_block, ddof=1))
            participant_means.append(test_block)

    # Print the mean for each block for each participant
    for participant_id, means in enumerate(zip(*participant_means)):
        print(f"Participant {participant_id + 1}: {means}")

    # Creating the bar plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(mean_metrics))  # the label locations
    plt.bar(x, mean_metrics, yerr=std_metrics, capsize=5)
    plt.xlabel('Blocks/Games')
    plt.ylabel(ylabel)
    plt.title(title)

    # Generate labels
    labels = ['Baseline']
    for i in range(1, len(mean_metrics)):
        labels.append(f'Train{i}' if i % 2 == 1 else f'Test{i//2 + 1}')

    plt.xticks(x, labels, rotation=45)  # Set x-ticks with the generated labels

    plt.grid(True)
    #plt.show()

# Example usage
plot_combined_bar_and_print_means(TL_test_distance, TL_train_distance, 'TL Group Travelled Distance', 'Distance')
plot_combined_bar_and_print_means(NO_TL_test_distance, NO_TL_train_distance, 'No TL Group Travelled Distance', 'Distance')

plt.show()

"""
#############HEATMAPS##################

from scipy.ndimage import gaussian_filter

def plot_heatmap_with_coverage(batch_number, filepaths, steps_filepaths, games_per_batch=10, threshold=0, max_x=-0.18, min_x=-0.349, max_y=0.330, min_y=0.170, smoothing_sigma=0.3, ax=None):
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
        print(start_game)
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
        plt.figure(figsize=(8, 6), facecolor='white')
    im=ax.imshow(smoothed_heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='YlGn', aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    #cbar.set_label('Counts')
    if ax is None:
        plt.colorbar(label='Counts')
        plt.title(f"Smoothed Heatmap of Positions in Batch {batch_number} with Threshold {threshold}")
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        #plt.gca().set_facecolor('black')

    print(f"Coverage for Batch {batch_number}: {coverage:.2%}")

    return coverage  # Optionally return the coverage value

num_rows = 3
num_cols = 3

# Batches to collect (0, 3, 5 for Experts, TL, and No TL)
batches_to_collect = [0, 3, 5]

# Create a figure with the desired size
fig = plt.figure(figsize=(12, 12))

# Titles for the subplots
# Titles for the subplots


# Create a list of labels for rows and columns
row_labels = ["0.4*E Expert", "TL", "No TL"]
col_labels = ["Baseline", "Block 3", "Block 6"]

# Iterate through rows and columns to create subplots
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
            plot_heatmap_with_coverage(batch_number, TL_test_data, TL_steps_test_data, ax=ax)
        elif group_idx == 2:
            # No TL Participant
            plot_heatmap_with_coverage(batch_number, NO_TL_test_data, NO_TL_steps_test_data, ax=ax)

        ax.set_title(title)

# Adjust the layout
plt.tight_layout()

# Show the figure
plt.show()
