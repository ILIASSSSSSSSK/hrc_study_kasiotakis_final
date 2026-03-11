#!/usr/bin/env python3
import rospy
import threading
import subprocess
from pydub import AudioSegment
from pydub.playback import play
from std_msgs.msg import Bool

class RL_Control:
    def __init__(self):
        # My parameters - change them to be read from config
        self.baseline_episodes = 8
        self.max_timesteps = 10

        # Game parameters       
        self.goal = rospy.get_param('rl_control/Game/goal', [0, 0])
        self.goal_dis = rospy.get_param('rl_control/Game/goal_distance', 2)
        self.goal_vel = rospy.get_param('rl_control/Game/goal_velocity', 2)
        self.action_duration = rospy.get_param('rl_control/Experiment/action_duration', 0.1)
        audio_dir = os.path.join(rospy.get_param('rl_control/Game/full_path'), 'audio_files')
        start_audio_files = rospy.get_param('rl_control/Game/start_audio', ['', ''])
        self.start_audio = [AudioSegment.from_mp3(os.path.join(audio_dir, file)) for file in start_audio_files]
        win_audio_file = rospy.get_param('rl_control/Game/win_audio', '')
        lose_audio_file = rospy.get_param('rl_control/Game/lose_audio', '')
        self.win_audio = AudioSegment.from_mp3(os.path.join(audio_dir, win_audio_file))
        self.lose_audio = AudioSegment.from_mp3(os.path.join(audio_dir, lose_audio_file))
        
        
        self.human_action_sub = rospy.Subscriber('cmd_vel', Twist, self.human_callback)
        self.agent_action_pub = rospy.Publisher('agent_action_topic', Float64, queue_size=10)
        self.train_pub = rospy.Publisher('train_topic', Bool, queue_size=10)
        self.score_pub = rospy.Publisher('score_topic', Score, queue_size=10)
        self.t_win = threading.Thread(target=play, args=(self.win_audio,))
        self.t_lose = threading.Thread(target=play, args=(self.lose_audio,))
        rospy.sleep(1)

    def baseline_loop(self):
        for i in range(1, self.baseline_episodes+1):
            self.reset()
            self.run(i)

    def reset(self):
        rospy.wait_for_service('reset')
        self.episode_reward = 0
        try:
            reset_game = rospy.ServiceProxy('reset', Reset)
            rospy.loginfo('Resetting the game')
            reset_game()
            rospy.loginfo('Game reset. Start episode')
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s"%e)

    def run(self):
        rospy.loginfo('Episode: {}'.format(i_episode))
        start_time = rospy.Time.now().to_sec()
        count = 0
        while rospy.Time.now().to_sec() - start_time <= 4:
            play(self.start_audio[0]) if count < 3 else play(self.start_audio[1])
            count += 1
            rospy.sleep(0.5)

        #self.start_clock()

        while not rospy.is_shutdown():
            self.timestep += 1
            
            if self.timestep == self.max_timesteps:
                self.timeout = True
            
            if rospy.get_time() - tmp_time > self.action_duration:
                tmp_time = rospy.get_time()
            
            rospy.sleep(self.action_duration)
            
            self.done = self.check_if_game_ended(block_number)

            if self.human_action is None:
                self.human_action = 0
            
            if self.timestep == 1:
                self.start_time = rospy.get_time()
        
            if self.done:
                self.end_time = rospy.get_time()
                break

    def check_if_game_ended(self, block_number):
        pos_x_ee=0
        pos_y_ee=0
        vel_x_ee=0
        vel_y_ee=0
       
        vel_x_ee=self.ur3_state.twist.linear.x
        vel_y_ee=self.ur3_state.twist.linear.y


        pos_x_ee=self.ur3_state.pose.position.x
        pos_y_ee=self.ur3_state.pose.position.y
        
        if (distance.euclidean([pos_x_ee, pos_y_ee], self.goal) <= self.goal_dis and 
            distance.euclidean([vel_x_ee, vel_y_ee], [0, 0]) <= self.goal_vel) or self.timeout:
        
            if self.timeout:
                rospy.loginfo('Episode ended with timeout')
                t_lose = threading.Thread(target=play, args=(self.lose_audio,))
                t_lose.start()
            else:
                rospy.loginfo('Episode ended with goal reached')
                t_win = threading.Thread(target=play, args=(self.win_audio,))
                t_win.start()
            return True
        return False


if __name__ == "__main__":
    game = RL_Control()
    game.basline_loop()
    #save_data(game, data_dir)
    #plot_statistics(game, plot_dir)
    
    rospy.loginfo("Human Baseline Ended!!!")
    rospy.loginfo("Total experiment duration: {} secs".format(end_experiment_time - start_experiment_time))