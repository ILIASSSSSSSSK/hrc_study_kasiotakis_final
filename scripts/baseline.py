#!/usr/bin/env python3.6
import rospy
from std_msgs.msg import Bool, Float64MultiArray
from geometry_msgs.msg import Twist
from cartesian_state_msgs.msg import PoseTwist
from human_robot_collaborative_learning.srv import Reset
import math
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import threading
import random
import os
import time

class GameController:
    def __init__(self):
        rospy.init_node('baseline_game_controller_python_node', anonymous=True)

        # Hardcoded parameters for audio file paths (from rl_params_christos.yaml)
        full_path_from_yaml = "/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/"
        audio_dir = os.path.join(full_path_from_yaml, "audio_files")

        # Audio file names
        start_audio_file_names = ['beep-07a.mp3', 'beep-09.mp3']
        win_audio_file_name = 'street-fighter-ii-you-win-perfect.mp3'
        lose_audio_file_name = 'gaming-sound-effect-hd.mp3'

        # Load audio segments
        self.start_audio = [AudioSegment.from_file(os.path.join(audio_dir, file)) for file in start_audio_file_names]
        self.win_audio = AudioSegment.from_file(os.path.join(audio_dir, win_audio_file_name))
        self.lose_audio = AudioSegment.from_file(os.path.join(audio_dir, lose_audio_file_name))
        
        self.current_ee_state = None
        self.game_counter = 0 # Tracks the current game number
        self.total_games = 8 # Total number of games to play
        self.game_active = False # Flag to indicate if a game is currently in progress
        self.game_finished_event = threading.Event() # Event to signal game end
        self.robot_ready_event = threading.Event()

        # ROS Subscribers
        self.experiment_finished_pub = rospy.Publisher('experiment_finished', Bool, queue_size=1)
        self.game_control_lock_pub = rospy.Publisher('game_control_lock', Bool, queue_size=1)
        self.ur3_state_sub = rospy.Subscriber('ur3_cartesian_velocity_controller/ee_state', PoseTwist, self.ee_state_callback)
        self.game_status_sub = rospy.Subscriber('game_win_status', Bool, self.game_status_callback)
        self.game_stats_sub = rospy.Subscriber('game_score_stats', Float64MultiArray, self.game_stats_callback)
        self.robot_ready_sub = rospy.Subscriber('robot_ready', Bool, self.robot_ready_callback)

        rospy.sleep(1) # Give some time for publishers/subscribers to set up

    def robot_ready_callback(self, msg):
        if msg.data:
            self.robot_ready_event.set()

    
    def ee_state_callback(self, msg):
        """Callback for the robot's end-effector state."""
        self.current_ee_state = msg

    def game_status_callback(self, msg):
        """Callback for game win/lose status published by the C++ node."""
        win_game = msg.data
        rospy.loginfo("Game ended. Win: %s", win_game)
        
        if win_game:
            threading.Thread(target=play, args=(self.win_audio,)).start()
        else:
            threading.Thread(target=play, args=(self.lose_audio,)).start()
        
        self.game_active = False # Mark game as inactive
        self.game_finished_event.set() # Signal that the game has finished


    def game_stats_callback(self, msg):
        """Callback for game statistics (time duration, travelled distance) published by the C++ node."""
        time_duration = msg.data[0] if len(msg.data) > 0 else 0.0
        travelled_distance = msg.data[1] if len(msg.data) > 1 else 0.0
        rospy.loginfo("Game stats: Duration: %.2f, Travelled Distance: %.2f", time_duration, travelled_distance)

    def robot_ready(self, position_idx):
        state = self.current_ee_state
        if state is None:
            return False

    def reset_game(self):
        self.robot_ready_event.clear()
        self.game_counter += 1
        position_to_request = self.game_counter % 2

        lock_msg = Bool()
        lock_msg.data = True
        self.game_control_lock_pub.publish(lock_msg)
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                rospy.wait_for_service('reset', timeout=5.0)
                reset_service = rospy.ServiceProxy('reset', Reset)
                rospy.loginfo('Resetting game %d/%d with position: %d', self.game_counter, self.total_games, position_to_request)
                reset_service(position=position_to_request)
                rospy.loginfo('Game reset successful.')
                break
            except (rospy.ServiceException, rospy.ROSException) as e:
                rospy.logwarn("Service call failed (attempt %d/%d): %s", attempt+1, max_retries, e)
                if attempt == max_retries - 1:
                    rospy.logerr("Reset service call failed after %d attempts. Shutting down.", max_retries)
                    rospy.signal_shutdown("Reset service call failed. Shutting down.")
                else:
                    rospy.sleep(1.0)
        else:
            return

        rospy.loginfo("Waiting for robot to reach start_position...")
        if not self.robot_ready_event.wait(timeout=10.0):
            rospy.loginfo("timeout waiting for robot to reach start position")
        
        if self.start_audio:
            start_time = rospy.Time.now().to_sec()
            count = 0
            while rospy.Time.now().to_sec() - start_time <= 4:
                play(self.start_audio[0]) if count < 3 else play(self.start_audio[1])
                count += 1
                time.sleep(0.5)
        
        self.game_active = True # Mark game as active
        self.game_finished_event.clear() # Clear the event for the new game

        # Unlock control after reset
        lock_msg.data = False
        self.game_control_lock_pub.publish(lock_msg)
                
    def run(self):
        initial_lock_msg = Bool()
        initial_lock_msg.data = True
        self.game_control_lock_pub.publish(initial_lock_msg)

        rospy.loginfo("Waiting for robot state...")
        while self.current_ee_state is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Robot state received. Starting game loop.")

        # Loop for the specified number of games
        while not rospy.is_shutdown() and self.game_counter < self.total_games:
            if not self.game_active:
                self.reset_game()
                # Wait for the current game to finish (signaled by game_status_callback)
                self.game_finished_event.wait() 
                # Add a small delay between games
                rospy.sleep(0.3) # 2 seconds pause between games

        rospy.loginfo("All %d games completed. Shutting down baseline_game_controller_python_node.", self.total_games)
        end_msg = Bool()
        end_msg.data = True
        self.experiment_finished_pub.publish(end_msg)
        rospy.sleep(0.5) # Give some time for the message to be sent

        rospy.signal_shutdown("All games completed.")

if __name__ == "__main__":
    game_controller = GameController()
    game_controller.run()