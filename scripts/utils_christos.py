import os
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import rospy
from sac_discrete_agent import DiscreteSACAgent


class RosParameterManager:
    """Centralized ROS parameter management."""
    
    @staticmethod
    def get_sac_params() -> dict:
        """Get SAC-related parameters."""
        return {
            'buffer_max_size': rospy.get_param("rl_control/SAC/buffer_max_size", 10000),
            'update_interval': rospy.get_param("rl_control/Experiment/learn_every_n_episodes", 10),
            'reward_scale': rospy.get_param("rl_control/Experiment/reward_scale", 2),
            'n_actions': rospy.get_param("rl_control/Experiment/number_of_agent_actions", 3),
        }
    
    @staticmethod
    def get_experiment_params() -> dict:
        """Get experiment-related parameters."""
        return {
            'participant': rospy.get_param('rl_control/Game/participant_name', 'thanasis'),
            'total_updates': rospy.get_param('rl_control/Experiment/total_update_cycles', 1000),
            'action_duration': int(rospy.get_param('rl_control/Experiment/action_duration', 100) * 1000),
            'scheduling': rospy.get_param('rl_control/Experiment/scheduling', 'uniform'),
            'full_path': rospy.get_param("rl_control/Game/full_path", "/home/ttsitos/catkin_ws/src/transfer_learning_SAC/"),
        }
    
    @staticmethod
    def get_transfer_learning_params() -> dict:
        """Get transfer learning related parameters."""
        return {
            'transfer_learning': rospy.get_param("rl_control/Game/load_model_transfer_learning", False),
            'lfd_participant': rospy.get_param("rl_control/Game/lfd_participant_gameplay", False),
            'lfd_expert': rospy.get_param("rl_control/Game/lfd_expert_gameplay", False),
            'initialized_agent': rospy.get_param("/rl_control/Game/initialized_agent", False),
            'initialized_agent_dir': rospy.get_param("/rl_control/Game/initialized_agent_dir", ""),
            'load_model_training': rospy.get_param("rl_control/Game/load_model_training", False),
            'load_model_training_dir': rospy.get_param("rl_control/Game/load_model_training_dir", ""),
        }


class DirectoryManager:
    """Handles directory creation and path management."""
    
    @staticmethod
    def create_unique_directory(base_path: str) -> str:
        """Create a unique directory by appending numbers if needed."""
        path = Path(base_path)
        counter = 1
        
        while path.with_name(f"{path.name}_{counter}").exists():
            counter += 1
            
        unique_path = path.with_name(f"{path.name}_{counter}")
        return str(unique_path)
    
    @staticmethod
    def create_directories(base_dir: str) -> Tuple[str, str, str]:
        """Create necessary subdirectories."""
        base_path = Path(base_dir)
        data_dir = base_path / 'data'
        plot_dir = base_path / 'plots'
        
        # Create directories
        base_path.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        return str(base_path), str(data_dir), str(plot_dir)


class ExperimentNaming:
    """Handles experiment directory naming logic."""
    
    @staticmethod
    def generate_experiment_name(params: dict) -> str:
        """Generate experiment directory name based on parameters."""
        updates_k = int(params['total_updates'] / 1000)
        
        # Determine transfer learning suffix
        if params['transfer_learning']:
            suffix = "PPR_TL"
        elif params['lfd_participant'] or params['lfd_expert']:
            suffix = "LfD_TL"
        else:
            suffix = "no_TL"
        
        return (f"{updates_k}K_every{params['update_interval']}_{params['scheduling']}_"
                f"{params['action_duration']}ms_{params['participant']}_{suffix}")


def get_sac_agent(observation_space: int, chkpt_dir: str = "") -> DiscreteSACAgent:
    """
    Create and configure a SAC agent.
    
    Args:
        observation_space: Size of the observation space
        chkpt_dir: Checkpoint directory path
        
    Returns:
        Configured DiscreteSACAgent
    """
    rospy.init_node('rl_control')
    
    sac_params = RosParameterManager.get_sac_params()
    exp_params = RosParameterManager.get_experiment_params()
    tl_params = RosParameterManager.get_transfer_learning_params()
    
    # Determine checkpoint directory
    if not chkpt_dir or tl_params['initialized_agent']:
        all_params = {**exp_params, **tl_params}
        experiment_name = ExperimentNaming.generate_experiment_name(all_params)
        
        model_base_path = Path(exp_params['full_path']) / 'rl_models' / experiment_name
        chkpt_dir = DirectoryManager.create_unique_directory(str(model_base_path))
    
    return DiscreteSACAgent(
        input_dims=observation_space,
        n_actions=sac_params['n_actions'],
        chkpt_dir=chkpt_dir,
        buffer_max_size=sac_params['buffer_max_size'],
        update_interval=sac_params['update_interval'],
        reward_scale=sac_params['reward_scale']
    )


def get_save_dir(load_model_for_training: bool = False) -> Tuple[str, str, str]:
    sac_params = RosParameterManager.get_sac_params()
    exp_params = RosParameterManager.get_experiment_params()
    tl_params = RosParameterManager.get_transfer_learning_params()

    base_path = Path(exp_params['full_path']) / 'games_info'
    
    if load_model_for_training:
        # Use existing directory
        save_dir = tl_params['load_model_training_dir']
        data_dir = Path(save_dir) / 'data'
        plot_dir = Path(save_dir) / 'plots'
        return str(save_dir), str(data_dir), str(plot_dir)
    
    # Create new directory
    all_params = {**sac_params, **exp_params, **tl_params}
    experiment_name = ExperimentNaming.generate_experiment_name(all_params)
    experiment_path = base_path / experiment_name
    
    unique_path = DirectoryManager.create_unique_directory(str(experiment_path))
    return DirectoryManager.create_directories(unique_path)


class DataSaver:
    """Handles data saving operations."""
    
    @staticmethod
    def save_training_data(game: Any, data_dir: str) -> None:
        """Save main training data."""
        filepath = Path(data_dir) / 'data.csv'
        
        with open(filepath, 'ab') as f:
            headers = [['expert'], ['Rewards'], ['Episodes Duration in Seconds'], 
                      ['Travelled Distance'], ['Episodes Duration in Timesteps']]
            np.savetxt(f, list(zip(*headers)), delimiter=',', fmt='%s')
            
            data = [
                len(game.reward_history) * [game.expert_action_flag],
                game.reward_history,
                game.episode_duration,
                game.travelled_distance,
                game.number_of_timesteps
            ]
            np.savetxt(f, list(zip(*data)), delimiter=',')
    
    @staticmethod
    def save_test_data(game: Any, data_dir: str) -> None:
        """Save test data."""
        filepath = Path(data_dir) / 'test_data.csv'
        
        with open(filepath, 'ab') as f:
            headers = [['Rewards'], ['Episodes Duration in Seconds'], 
                      ['Travelled Distance'], ['Episodes Duration in Timesteps']]
            np.savetxt(f, list(zip(*headers)), delimiter=',', fmt='%s')
            
            data = [
                game.test_reward_history,
                game.test_episode_duration,
                game.test_travelled_distance,
                game.test_number_of_timesteps
            ]
            np.savetxt(f, list(zip(*data)), delimiter=',')
    
    @staticmethod
    def save_rl_data(game: Any, data_dir: str) -> None:
        """Save RL state information."""
        filepath = Path(data_dir) / 'rl_data.csv'
        
        with open(filepath, 'ab') as f:
            np.savetxt(f, [game.state_info[0]], delimiter=',', fmt='%s')
            np.savetxt(f, game.state_info[1:], delimiter=',')
    
    @staticmethod
    def save_test_rl_data(game: Any, data_dir: str) -> None:
        """Save test RL state information."""
        filepath = Path(data_dir) / 'rl_test_data.csv'
        
        with open(filepath, 'ab') as f:
            np.savetxt(f, [game.test_state_info[0]], delimiter=',', fmt='%s')
            np.savetxt(f, game.test_state_info[1:], delimiter=',')
    
    @staticmethod
    def save_entropy_data(game: Any, data_dir: str) -> None:
        """Save entropy data."""
        filepath = Path(data_dir) / 'entropy.csv'
        
        with open(filepath, 'ab') as f:
            np.savetxt(f, [game.temp[0]], delimiter=',', fmt='%s')
            np.savetxt(f, game.temp[1:], delimiter=',')


def save_data(game: Any, data_dir: str) -> None:
    """
    Save all game data.
    
    Args:
        game: Game object containing data to save
        data_dir: Directory to save data files
    """
    DataSaver.save_training_data(game, data_dir)
    DataSaver.save_test_data(game, data_dir)
    DataSaver.save_rl_data(game, data_dir)
    DataSaver.save_test_rl_data(game, data_dir)
    DataSaver.save_entropy_data(game, data_dir)


class PlotGenerator:
    """Handles plot generation and saving."""
    
    @staticmethod
    def create_plot(data: List[float], title: str, xlabel: str, ylabel: str, 
                   filename: str, plot_dir: str) -> None:
        """Create and save a single plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(np.arange(1, len(data) + 1), data)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        
        filepath = Path(plot_dir) / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    @staticmethod
    def generate_training_plots(game: Any, plot_dir: str) -> None:
        """Generate training plots."""
        plots_config = [
            (game.reward_history, 'Rewards over episodes', 'Episodes(N)', 'Rewards', 'rewards.png'),
            (game.episode_duration, 'Episodes duration', 'Episodes(N)', 'Duration(sec)', 'time_duration.png'),
            (game.travelled_distance, 'Travelled Distance', 'Episodes(N)', 'Travelled(m)', 'travelled_distance.png'),
            (game.number_of_timesteps, 'Number of Timesteps', 'Episodes(N)', 'Timesteps(M)', 'number_of_timesteps.png'),
        ]
        
        for data, title, xlabel, ylabel, filename in plots_config:
            PlotGenerator.create_plot(data, title, xlabel, ylabel, filename, plot_dir)
    
    @staticmethod
    def generate_test_plots(game: Any, plot_dir: str) -> None:
        """Generate test plots."""
        plots_config = [
            (game.test_reward_history, 'Test Rewards over episodes', 'Episodes(N)', 'Rewards', 'test_rewards.png'),
            (game.test_episode_duration, 'Test Episodes duration', 'Episodes(N)', 'Duration(sec)', 'test_time_duration.png'),
            (game.test_travelled_distance, 'Test Travelled Distance', 'Episodes(N)', 'Travelled(m)', 'test_travelled_distance.png'),
            (game.test_number_of_timesteps, 'Test Number of Timesteps', 'Episodes(N)', 'Timesteps(M)', 'test_number_of_timesteps.png'),
        ]
        
        for data, title, xlabel, ylabel, filename in plots_config:
            PlotGenerator.create_plot(data, title, xlabel, ylabel, filename, plot_dir)


def plot_statistics(game: Any, plot_dir: str) -> None:
    """
    Generate and save all statistics plots.
    
    Args:
        game: Game object containing data to plot
        plot_dir: Directory to save plot files
    """
    PlotGenerator.generate_training_plots(game, plot_dir)
    PlotGenerator.generate_test_plots(game, plot_dir)