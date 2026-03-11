import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 
import pandas as pd
from scipy import stats 



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
TL_test_data=[]
TL_train_data=[]
NO_TL_test_data=[]
NO_TL_train_data=[]
TL_steps_test_data=[]
TL_steps_train_data=[]
NO_TL_steps_test_data=[]
NO_TL_steps_train_data=[]
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
#expert_train_data= [participant_data[:, 1:] for participant_data in expert_train_data]

#for rewards
def calculate_means(data_group):
    means = []
    # Check if data_group is a list of arrays
    for participant_data in data_group:
        if isinstance(participant_data, np.ndarray) and participant_data.ndim == 2:
            # Take the first 10 rows and the column for rewards
            #means.append(np.mean(participant_data[:10, 0]+150))  #BASELINE
            means.append(np.mean(participant_data[-10:, 0]+150))  # LAST 10

        else:
            raise ValueError("Each item in data_group must be a 2-dimensional NumPy array")
    return means

TL_test_means = calculate_means(TL_test_data)
NO_TL_test_means = calculate_means(NO_TL_test_data)

def perform_t_test(group1_means, group2_means):
    # Perform the t-test on the means
    t_stat, p_value = stats.ttest_ind(group1_means, group2_means, equal_var=False)  # Assuming equal_var=False for Welch's t-test
    return t_stat, p_value


t_stat, p_value = perform_t_test(TL_test_means, NO_TL_test_means)


u_stat, p_value_u_test = stats.mannwhitneyu(TL_test_means, NO_TL_test_means, alternative='two-sided')

#print(f"TL_shapiro_normality: {p1}, NO_TL_shapiro_normality: {p2}")
print(f"T-statistic: {t_stat}, P-value: {p_value}")
print(f"Mann-Whitney U : {u_stat}, Mann-Whitney p-Value {p_value_u_test}")




# Define a function to calculate mean wins for wins
def calculate_mean_wins(data_group):
    means = []
    for participant_data in data_group:
        if isinstance(participant_data, np.ndarray) and participant_data.ndim == 2:
            # Calculate wins by counting non-negative rewards (excluding -150)
            #wins = np.sum(participant_data[:10, 0] != -150)  # Change 0 to the column index for rewards if necessary
            wins = np.sum(participant_data[-10:, 0] != -150)  # Change 0 to the column index for rewards if necessary

            means.append(wins)
        else:
            raise ValueError("Each item in data_group must be a 2-dimensional NumPy array")
    return means

# Calculate mean wins for TL and NO TL groups
TL_test_mean_wins = calculate_mean_wins(TL_test_data)
NO_TL_test_mean_wins = calculate_mean_wins(NO_TL_test_data)

# Perform a statistical test (e.g., Mann-Whitney U test) to compare mean wins
u_stat_wins, p_value_u_test_wins = stats.mannwhitneyu(TL_test_mean_wins, NO_TL_test_mean_wins, alternative='two-sided')


print(f"Mann-Whitney U (Wins) : {u_stat_wins}, Mann-Whitney p-Value (Wins) {p_value_u_test_wins}")
# Define a function to perform a t-test and Shapiro-Wilk normality test
def perform_wins_tests(group1_means, group2_means):
    # Perform the t-test on the means
    t_stat_wins, p_value_wins = stats.ttest_ind(group1_means, group2_means, equal_var=False)  # Assuming equal_var=False for Welch's t-test
    
    # Perform Shapiro-Wilk normality test
    _, p_value_group1_wins = stats.shapiro(group1_means)
    _, p_value_group2_wins = stats.shapiro(group2_means)
    
    return t_stat_wins, p_value_wins, p_value_group1_wins, p_value_group2_wins

# Perform t-test and Shapiro-Wilk test for mean wins
t_stat_wins, p_value_wins, p_value_group1_wins, p_value_group2_wins = perform_wins_tests(TL_test_mean_wins, NO_TL_test_mean_wins)


print(f"T-test (Wins) - T-statistic: {t_stat_wins}, P-value: {p_value_wins}")
print(f"Shapiro-Wilk Test (TL Group Wins) - P-value: {p_value_group1_wins}")
print(f"Shapiro-Wilk Test (NO TL Group Wins) - P-value: {p_value_group2_wins}")



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



TL_test_distance = calculate_mean_normalized_distances(TL_test_data)
NO_TL_test_distance = calculate_mean_normalized_distances(NO_TL_test_data)

print("TD for TL",TL_test_distance)
print("TD for no TL",NO_TL_test_distance)

# Let's flatten the lists of lists into a single list for each group
TL_normalized_distances_flat = [dist for sublist in TL_test_distance for dist in sublist]
NO_TL_normalized_distances_flat = [dist for sublist in NO_TL_test_distance for dist in sublist]

# Export the data to CSV files
import csv

# Save TL group data
with open('/home/nick/TL_normalized_distances.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['normalized_distance'])
    for dist in TL_normalized_distances_flat:
        writer.writerow([dist])

# Save No TL group data
with open('/home/nick/NO_TL_normalized_distances.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['normalized_distance'])
    for dist in NO_TL_normalized_distances_flat:
        writer.writerow([dist])