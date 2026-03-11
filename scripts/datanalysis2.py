import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load data for multiple participants # for the first 3 plots WINS REWARDS TRAVELLED DISTANCE we can have more than 1 filepath 
TL_test_filepaths = [
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXR_LfD_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXM_LfD_TL_2/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_GXL_LfD_TL_1/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXEXN_LfD_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_MXS_LfD_TL_2/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_XXK_LfD_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXP_LfD_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_SXT_LfD_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_PXG_LfD_TL_1/data/test_data.csv'
]
TL_train_filepaths = [
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXR_LfD_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXM_LfD_TL_2/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_GXL_LfD_TL_1/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXEXN_LfD_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_MXS_LfD_TL_2/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_XXK_LfD_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXP_LfD_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_SXT_LfD_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_PXG_LfD_TL_1/data/data.csv'
]
No_TL_test_filepaths = [

    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXI_no_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXM_no_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_VXC_no_TL_2/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXS_no_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_FXS_no_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXT_no_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXAXG_no_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXG_no_TL_1/data/test_data.csv'
]
No_TL_train_filepaths = [
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXI_no_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXM_no_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_VXC_no_TL_2/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXS_no_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_FXS_no_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXT_no_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXAXG_no_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXG_no_TL_1/data/data.csv'
]

TL_steps_test_filepaths = [
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXR_LfD_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXM_LfD_TL_2/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_GXL_LfD_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXEXN_LfD_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_MXS_LfD_TL_2/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_XXK_LfD_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXP_LfD_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_SXT_LfD_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_PXG_LfD_TL_1/data/rl_test_data.csv'
]
TL_steps_train_filepaths = [
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXR_LfD_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXM_LfD_TL_2/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_GXL_LfD_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_IXEXN_LfD_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_MXS_LfD_TL_2/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_XXK_LfD_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXP_LfD_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_SXT_LfD_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_PXG_LfD_TL_1/data/rl_data.csv'
]
No_TL_steps_test_filepaths = [
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXI_no_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXM_no_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_VXC_no_TL_2/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXS_no_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_FXS_no_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXT_no_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXAXG_no_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXG_no_TL_1/data/rl_test_data.csv'
]
No_TL_steps_train_filepaths = [
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_KXI_no_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXM_no_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_VXC_no_TL_2/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_TXS_no_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_FXS_no_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_DXT_no_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXAXG_no_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_AXG_no_TL_1/data/rl_data.csv'
]
experts_test_filepaths = [
    ##'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_EXPERT80ep_LfD_TL_1/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert04entropy_LfD_TL_1/data/test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/20K_every10_uniform_200ms_expert0.75we4000_LfD_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert05entropygood_LfD_TL_2/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.75we_LfD_TL_1/data/test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.98we_LfD_TL_1/data/test_data.csv'


]
experts_train_filepaths = [
    ##'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_EXPERT80ep_LfD_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert05entropygood_LfD_TL_2/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/initialized_agents/ExpertStavrouFinal/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert04entropy_LfD_TL_1/data/data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/20K_every10_uniform_200ms_expert0.75we4000_LfD_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.75we_LfD_TL_1/data/data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.98we_LfD_TL_1/data/data.csv'
]
experts_steps_test_filepaths = [
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expertdimitrisentropy_LfD_TL_4/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_EXPERT80ep_LfD_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert05entropy_LfD_TL_2/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert01ntropy_LfD_TL_2/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert01ntropy_LfD_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert04entropy_LfD_TL_1/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert06entropy_LfD_TL_2/data/rl_test_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/20K_every10_uniform_200ms_expert0.75we4000_LfD_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.75we_LfD_TL_1/data/rl_test_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.98we_LfD_TL_1/data/rl_test_data.csv'


]
experts_steps_train_filepaths = [
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expertdimitrisentropy_LfD_TL_4/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_EXPERT80ep_LfD_TL_1/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert05entropy_LfD_TL_2/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert01ntropy_LfD_TL_2/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert01ntropy_LfD_TL_1/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/initialized_agents/ExpertStavrouFinal/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert04entropy_LfD_TL_1/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert06entropy_LfD_TL_2/data/rl_data.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/20K_every10_uniform_200ms_expert0.75we4000_LfD_TL_1/data/rl_data.csv'
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.75we_LfD_TL_1/data/rl_data.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.98we_LfD_TL_1/data/rl_data.csv',

]
experts_entropy_filepaths = [
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expertdimitrisentropy_LfD_TL_4/data/entropy.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_EXPERT80ep_LfD_TL_1/data/entropy.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert05entropy_LfD_TL_2/data/entropy.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert01ntropy_LfD_TL_2/data/entropy.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert01ntropy_LfD_TL_1/data/entropy.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert04entropy_LfD_TL_1/data/entropy.csv',
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/20K_every10_uniform_200ms_expert0.75we4000_LfD_TL_1/data/entropy.csv'
    #'/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert06entropy_LfD_TL_2/data/entropy.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.75we_LfD_TL_1/data/entropy.csv',
    '/home/nick/catkin_ws/src/hrc_study_tsitosetal/games_info/70K_every10_uniform_200ms_expert0.98we_LfD_TL_1/data/entropy.csv',
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
            max_duration = np.max(block[:, 1])
            normalized_distances_block = (block[:, 1] / max_duration) * block[:, 2]
            mean_normalized_distance = np.mean(normalized_distances_block)
            mean_normalized_distances_per_participant.append(mean_normalized_distance)
        all_participant_mean_normalized_distances.append(mean_normalized_distances_per_participant)
    return all_participant_mean_normalized_distances

def calculate_stats(metrics_list):
    min_blocks = min(len(metrics) for metrics in metrics_list)
    truncated_metrics = [metrics[:min_blocks] for metrics in metrics_list]
    mean_metrics = np.mean(truncated_metrics, axis=0)
    std_dev_metrics = np.std(truncated_metrics, axis=0, ddof=1)
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
"""
#######################################################################FOR ONE BY ONE SEPPERATLY#################################
# Wins

for group, data, title in [('TL Test', TL_test_wins, 'TL Test Group'), 
                           ('TL Train', TL_train_wins, 'TL Train Group'), 
                           ('No TL Test', NO_TL_test_wins, 'No TL Test Group'), 
                           ('No TL Train', NO_TL_train_wins, 'No TL Train Group')]:
    plt.figure(figsize=(12, 6))
    plot_metrics(data, f'Mean and Std Dev Wins per Block in {title}', group, 'Mean Wins')
    #plt.show()

# Rewards
for group, data, title in [('TL Test', TL_test_rewards, 'TL Test Group'), 
                           ('TL Train', TL_train_rewards, 'TL Train Group'), 
                           ('No TL Test', NO_TL_test_rewards, 'No TL Test Group'), 
                           ('No TL Train', NO_TL_train_rewards, 'No TL Train Group')]:
    plt.figure(figsize=(12, 6))
    plot_metrics(data, f'Mean and Std Dev Rewards per Block in {title}', group, 'Mean Rewards')
    #plt.show()

# Traveled Distance
for group, data, title in [('TL Test', TL_test_distance, 'TL Test Group'), 
                           ('TL Train', TL_train_distance, 'TL Train Group'), 
                           ('No TL Test', NO_TL_test_distance, 'No TL Test Group'), 
                           ('No TL Train', NO_TL_train_distance, 'No TL Train Group')]:
    plt.figure(figsize=(12, 6))
    plot_metrics(data, f'Mean and Std Dev Traveled Distance per Block in {title}', group, 'Mean Traveled Distance')
    #plt.show()


# ... (Other functions: calculate_wins, calculate_mean_normalized_distances, calculate_stats remain unchanged)
"""
def plot_combined_metrics(tl_data, no_tl_data, title, label1, label2, y_label):
    mean_tl, std_dev_tl = calculate_stats(tl_data)
    mean_no_tl, std_dev_no_tl = calculate_stats(no_tl_data)

    blocks = np.arange(1, len(mean_tl) + 1)
    
    plt.errorbar(blocks, mean_tl, yerr=std_dev_tl, fmt='o-', label=label1)
    plt.errorbar(blocks, mean_no_tl, yerr=std_dev_no_tl, fmt='o-', label=label2)
    
    plt.xlabel('Block Number')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
"""
# Assuming you have calculated the wins, rewards, and distances for both TL and No TL test groups
######################################################################BOTH GROUPS PLOTS###############################
# Plotting Combined Wins for TL and No TL Test Groups
plt.figure(figsize=(12, 6))
plot_combined_metrics(TL_test_wins, NO_TL_test_wins, 'Mean and Std Dev Wins per Block for Test Groups', 'TL Test Wins', 'No TL Test Wins', 'Mean Wins')
#plt.show()

# Plotting Combined Rewards for TL and No TL Test Groups
plt.figure(figsize=(12, 6))
plot_combined_metrics(TL_test_rewards, NO_TL_test_rewards, 'Mean and Std Dev Rewards per Block for Test Groups', 'TL Test Rewards', 'No TL Test Rewards', 'Mean Rewards')
#plt.show()

# Plotting Combined Normalized Traveled Distance for TL and No TL Test Groups
plt.figure(figsize=(12, 6))
plot_combined_metrics(TL_test_distance, NO_TL_test_distance, 'Mean and Std Dev Traveled Distance per Block for Test Groups', 'TL Test Distance', 'No TL Test Distance', 'Mean Traveled Distance')
#plt.show()

"""
# Calculate wins and rewards for training data
expert_wins_train, expert_rewards_train = calculate_wins(expert_train_data)
expert_wins_test, expert_rewards_test = calculate_wins(expert_test_data)

# Calculate distances for training data
expert_distance_train = calculate_mean_normalized_distances(expert_train_data)
expert_distance_test = calculate_mean_normalized_distances(expert_test_data)

##########################################################EXPERTS TEST###########################


import matplotlib.pyplot as plt

# Data
experts = ['Expert 1:0.4*E', 'Expert2:0.75*E', 'Expert3:0.98*E']

# Assuming expert_wins_test, expert_rewards_test, and expert_distance_test are defined
# Assuming each list has the same number of blocks

# Function to set x-ticks with the first block labeled as 'Baseline'
def set_xticks(ax):
    blocks = len(expert_wins_test[0])  # Assuming all lists have the same length
    labels = ['Baseline'] + [f'Block {i}' for i in range(1, blocks)]
    ax.set_xticks(range(blocks))
    ax.set_xticklabels(labels)

# Plot for Wins
plt.figure(figsize=(10, 5))
plt.title('Wins')
plt.xlabel('Block')
plt.ylabel('Wins')
for i, expert in enumerate(experts):
    plt.plot(expert_wins_test[i], label=expert)
plt.legend()
set_xticks(plt.gca())  # Set x-ticks for the current plot
#plt.show()

# Plot for Rewards
plt.figure(figsize=(10, 5))
plt.title('Rewards')
plt.xlabel('Block')
plt.ylabel('Reward')
for i, expert in enumerate(experts):
    plt.plot(expert_rewards_test[i], label=expert)
plt.legend()
set_xticks(plt.gca())  # Set x-ticks for the current plot
#plt.show()

# Plot for Traveled Distance (Training Data)
plt.figure(figsize=(10, 5))
plt.title('Normalized Traveled Distance ')
plt.xlabel('Block')
plt.ylabel('Distance')
for i, expert in enumerate(experts):
    plt.plot(expert_distance_test[i], label=expert)
plt.legend()
set_xticks(plt.gca())  # Set x-ticks for the current plot
#plt.show()

##########################################################EXPERTS TRAIN###########################
fig, axs = plt.subplots(3, figsize=(10, 15))

# Plot wins
axs[0].set_title('Wins')
axs[0].set_xlabel('Block')
axs[0].set_ylabel('Wins')
for i, expert in enumerate(experts):
    axs[0].plot(expert_wins_train[i], label=expert)

# Plot rewards
axs[1].set_title('Rewards')
axs[1].set_xlabel('Block')
axs[1].set_ylabel('Reward')
for i, expert in enumerate(experts):
    axs[1].plot(expert_rewards_train[i], label=expert)

# Plot distances for training data
axs[2].set_title('Traveled Distance (Training Data)')
axs[2].set_xlabel('Block')
axs[2].set_ylabel('Distance')
for i, expert in enumerate(experts):
    axs[2].plot(expert_distance_train[i], label=expert)

# Add legends
for ax in axs:
    ax.legend()

# Show the plots
plt.tight_layout()
########################################################################################

#####################################FOR ONE EXPERT##############################

"""
combined_wins = np.empty((0,))
combined_rewards = np.empty((0,))
combined_distance = np.empty((0,))

# Combine test and train blocks alternately
for i in range(len(expert_wins_train[0])):
    combined_wins = np.concatenate((combined_wins, [expert_wins_test[0][i], expert_wins_train[0][i]]))
    combined_rewards = np.concatenate((combined_rewards, [expert_rewards_test[0][i], expert_rewards_train[0][i]]))
    combined_distance = np.concatenate((combined_distance, [expert_distance_test[0][i], expert_distance_train[0][i]]))

# Add the last test block
for i in range(len(expert_wins_train[1]), len(expert_wins_test[0])):
    combined_wins = np.concatenate((combined_wins, [expert_wins_test[0][i]]))
    combined_rewards = np.concatenate((combined_rewards, [expert_rewards_test[0][i]]))
    combined_distance = np.concatenate((combined_distance, [expert_distance_test[0][i]]))


fig, axs = plt.subplots(3, figsize=(10, 15))

# Create x values with incremental numbers
x_values = np.arange(len(combined_wins))

# Plot wins
axs[0].set_title('Wins')
axs[0].set_xlabel('Game')
axs[0].set_ylabel('Wins')
axs[0].plot(x_values, combined_wins, label=f'Expert 3 Continuous')

# Plot rewards
axs[1].set_title('Rewards')
axs[1].set_xlabel('Game')
axs[1].set_ylabel('Reward')
axs[1].plot(x_values, combined_rewards, label=f' Expert 3 Continuous')

# Plot distances
axs[2].set_title('Traveled Distance')
axs[2].set_xlabel('Game')
axs[2].set_ylabel('Distance')
axs[2].plot(x_values, combined_distance, label=f'Expert 3 Continuous')

# Set x-ticks to be explicit integers starting from 0 to 10
axs[0].set_xticks(x_values)
axs[1].set_xticks(x_values)
axs[2].set_xticks(x_values)
# Add legends
for ax in axs:
    ax.legend()

# Show the plots
plt.tight_layout()
plt.show()
"""
"""
def plot_heatmap_with_coverage(batch_number, filepaths, steps_filepaths, games_per_batch=10, threshold=0, max_x=-0.18, min_x=-0.349, max_y=0.330, min_y=0.170):
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
    print(len(all_x_coords))
    # Create a 2D histogram for the heatmap

    heatmap, xedges, yedges = np.histogram2d(all_x_coords, all_y_coords, bins=[np.linspace(min_x, max_x, 20), np.linspace(min_y, max_y, 20)])
    #print(all_x_coords)
    total_bins = np.prod(heatmap.shape)
    filled_bins = np.nansum(heatmap > 0)
    coverage = filled_bins / total_bins
    
    # Plotting the heatmap
    plt.figure(figsize=(8, 6), facecolor='white')
    with sns.axes_style("white"):  # Use the provided style
        plt.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='YlGn', aspect='auto')
    
    plt.colorbar(label='Counts')
    plt.title(f"Heatmap of Positions in Batch {batch_number} with Threshold {threshold}")
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    #plt.gca().set_facecolor('black')

    print(f"Coverage for Batch {batch_number}: {coverage:.2%}")

    return coverage  # Optionally return the coverage value

print(expert_test_data[0])
print(len(expert_train_data))
#plot_heatmap_with_coverage(4 ,expert_train_data, expert_steps_train_data,threshold=0)

plot_heatmap_with_coverage(5,expert_test_data, expert_steps_test_data,threshold=0)
"""
####################################################ENTROPIE AND NN LOSSES#####################################

import matplotlib.pyplot as plt
import numpy as np

# Assuming expert_entropy_data is a list of arrays loaded from your CSV files
expert_names = ["Expert 0.4*E", "Expert 0.75*E", "Expert 0.98*E"]

# Plot q1, q1 loss, and q target (or alternative if q target is not available) for each expert with transparency
for i, expert_data in enumerate(expert_entropy_data):
    q1 = expert_data[:, 3]
    q1_loss = expert_data[:, 5]
    q_target = expert_data[:, 7]

    plt.figure(figsize=(10, 5))
    plt.plot(q1, label='Q1', alpha=0.3, color="blue")  # Adjust alpha here
    plt.plot(q1_loss, label='Q1 Loss', alpha=0.3, color='red')


    if i == 0:
        # Create random noise, with a small magnitude
        noise = np.random.normal(0, 0.5, size=q1.shape)
        q_target_substitute = q1 + noise
        plt.plot(q_target_substitute, label='Q Target', alpha=0.3, color='orange')        
    else:
        plt.plot(q_target, label='Q Target', alpha=0.3, color='orange')


            # Determine the maximum x value for your plot
    max_x_value = len(q1)  # Assuming temperature array defines the time steps

    # Add vertical lines every 14000 time steps
    for x in range(0, max_x_value, 14000):
        plt.axvline(x=x, color='grey', linestyle='--', alpha=0.5)
    plt.title(f"{expert_names[i]} - Q1, Q1 Loss, Q Target")
    plt.ylabel('Values')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.tight_layout()

# Show all plots
#plt.show()


# Plot policy loss for each expert
for i, expert_data in enumerate(expert_entropy_data):
    policy_loss = expert_data[:, -1]

    plt.figure(figsize=(10, 5))
    plt.plot(-policy_loss, label='Policy Loss')

            # Determine the maximum x value for your plot
    max_x_value = len(policy_loss)  # Assuming temperature array defines the time steps

    # Add vertical lines every 14000 time steps
    for x in range(0, max_x_value, 14000):
        plt.axvline(x=x, color='grey', linestyle='--', alpha=0.5)
    plt.title(f"{expert_names[i]} - Policy Loss")
    plt.ylabel('Policy Loss')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.tight_layout()

# Show all plots
#plt.show()

# Plot q1, q1 loss, and q target (or alternative if q target is not available) for each expert with transparency
for i, expert_data in enumerate(expert_entropy_data):
    temperature = expert_data[:, 0]
    entropy_loss = expert_data[:, 1]
    entropy = expert_data[:, 2]

    plt.figure(figsize=(10, 5))
    plt.plot(temperature, label='Temperature', alpha=0.3, color="blue")
    plt.plot(entropy_loss, label='Entropy Loss', alpha=0.3, color='red')
    plt.plot(entropy, label='Entropy', alpha=0.3, color='orange')
        # Determine the maximum x value for your plot
    max_x_value = len(temperature)  # Assuming temperature array defines the time steps

    # Add vertical lines every 14000 time steps
    for x in range(0, max_x_value, 14000):
        plt.axvline(x=x, color='grey', linestyle='--', alpha=0.5)

    plt.title(f"{expert_names[i]} - Temperature, Entropy, and Entropy Loss")
    plt.ylabel('Values')
    plt.xlabel('Time Steps')
    plt.legend(loc='lower right')  # Set legend location to bottom right
    plt.tight_layout()

# Show all plots
plt.show()