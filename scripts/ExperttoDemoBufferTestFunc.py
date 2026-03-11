import numpy as np
import pickle

def ExpertoDemo(filename, save_path, selection_mode='random'):
    data = np.load(filename, allow_pickle=True, encoding='latin1')
    np.set_printoptions(threshold=np.inf)

    last_column = data[:, -1]
    total_games = np.sum(last_column)
    print("Total number of games:", total_games)

    indices = np.where(last_column)[0]
    splits = np.array_split(indices, len(indices) // 10)
    episodes = []

    num_extracted_games_per_episode = [2, 2, 2, 2, 2] # Define the number of games per block

    for split in splits:
        start_idx = split[0] - 1 if split[0] != 0 else 0
        end_idx = split[-1] + 1
        split_data = data[start_idx:end_idx]
        episodes.append(split_data)

    extracted_games = []

    for episode_idx, episode in enumerate(episodes):
        column_3_values = episode[:, 2]
        winning_games_indices = np.where((column_3_values == 10) & (episode[:, -1] == True))[0]

        winning_games = []
        for game_idx in winning_games_indices:
            game_start_idx = game_idx - 1
            while game_start_idx >= 0 and not episode[game_start_idx, -1]:
                game_start_idx -= 1
            game_start_idx = max(0, game_start_idx)
            winning_games.append(episode[game_start_idx+1:game_idx + 1])

        print("Number of WINNING games in episode:", len(winning_games))

        if selection_mode == 'largest':
            # Sort winning games by their length in descending order
            winning_games.sort(key=lambda x: x.shape[0], reverse=True)
        elif selection_mode == 'random':
            # Shuffle the winning games
            np.random.shuffle(winning_games)
        else:
            raise ValueError("Invalid selection mode. Choose 'largest' or 'random'.")

        num_extracted_games = min(num_extracted_games_per_episode[episode_idx], len(winning_games))
        extracted_games.extend(winning_games[:num_extracted_games])

    for game in extracted_games:
        print("Game shape:", game.shape)

    print("Number of extracted games total:", len(extracted_games))

    # Save the demogames to an array and to a specific path
    demo_games = np.vstack(extracted_games)

    with open(save_path, 'wb') as f:
        pickle.dump(demo_games, f, protocol=2)

    print("shape", demo_games.shape)

save_path = '/home/ttsitos/catkin_ws/src/hrc_study_tsitosetal/buffers/demo_buffer/demo_data_relu_keepingfromthe1stbatchtoo.npy'
filename = '/home/ttsitos/catkin_ws/src/hrc_study_tsitosetal/buffers/expert_buffer/buffer_data_relu_04entropy.npy'
ExpertoDemo(filename, save_path, selection_mode='largest') # Change 'largest' to 'random' to select randomly
