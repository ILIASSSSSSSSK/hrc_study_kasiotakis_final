// baseline.cpp
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Bool.h> // For win/lose status
#include <std_msgs/Float64MultiArray.h> // For game stats (duration, travelled distance)
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Accel.h>
#include <cartesian_state_msgs/PoseTwist.h>
#include <human_robot_collaborative_learning/Reset.h> // Include the service header

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <memory>
#include <random> 

class AccCommand{
public:
	ros::NodeHandle n;
	ros::Publisher pub, game_status_pub, game_stats_pub, robot_ready_pub; // Added standard publishers
	ros::ServiceServer service; 
	ros::Subscriber human_sub, state_sub, experiment_finished_sub, game_control_lock_sub; 
	float tmp_time, min_vel, max_vel, min_y, max_y, velocity_tolerance, position_tolerance;
    double timestart, start_time, end_time, pause_duration, start_pause, end_pause, max_game_duration;
	bool cmd_y, reset, stop_y, pause, play, timeout, game_locked, reset_requested; 
	int count_y, reset_count, num_of_games; 
	float last_time, travelled_distance;
	int start_pos_toggle;
	std::vector<float> start_pos;
	std::vector<float> start_pos_0;
	std::vector<float> start_pos_1;
	boost::shared_ptr<cartesian_state_msgs::PoseTwist const> state;
	cartesian_state_msgs::PoseTwist prev_state;
	geometry_msgs::Twist::Ptr cmd_vel;	
	geometry_msgs::Twist::Ptr zero_vel;	
	geometry_msgs::Accel::Ptr cmd_acc;
	human_robot_collaborative_learning::Reset::Request last_reset_req;
    std::string participant_file;
    std::ofstream duration_file, stats_file;

	AccCommand();
	bool reset_game(human_robot_collaborative_learning::Reset::Request& req, human_robot_collaborative_learning::Reset::Response& res);
	void process_reset();
	void ee_state_callback(const cartesian_state_msgs::PoseTwist::ConstPtr &ee_state);
	void human_callback(const geometry_msgs::Twist::ConstPtr &msg);
	void experiment_finished_callback(const std_msgs::Bool::ConstPtr &msg);
	void game_control_lock_callback(const std_msgs::Bool::ConstPtr &msg);
	void run();
};

float distance(const cartesian_state_msgs::PoseTwist::ConstPtr& state, const std::vector<float>& pos){
	return sqrt(pow(state->pose.position.x-pos[0], 2) + pow(state->pose.position.y-pos[1], 2) + pow(state->pose.position.z-pos[2], 2));
}

AccCommand::AccCommand(){
	this->pub = this->n.advertise<geometry_msgs::Twist>("ur3_cartesian_velocity_controller/command_cart_vel", 100);
	this->game_status_pub = this->n.advertise<std_msgs::Bool>("game_win_status", 100); // Initialize game_status_pub
    this->game_stats_pub = this->n.advertise<std_msgs::Float64MultiArray>("game_score_stats", 100); // Initialize game_stats_pub
    this->robot_ready_pub = this ->n.advertise<std_msgs::Bool>("robot_ready", 100);
	this->tmp_time = 0;
	this->start_pos_toggle = 0;
	this->state = boost::make_shared<cartesian_state_msgs::PoseTwist>();
	this->cmd_vel = boost::make_shared<geometry_msgs::Twist>();
	this->zero_vel = boost::make_shared<geometry_msgs::Twist>();
	this->cmd_acc = boost::make_shared<geometry_msgs::Accel>();
	
	this->service = n.advertiseService("reset", &AccCommand::reset_game, this);

	n.param("robot_movement_generation/min_vel", this->min_vel, 0.0f);
	n.param("robot_movement_generation/max_vel", this->max_vel, 0.0f);
	n.param("robot_movement_generation/min_y", this->min_y, 0.0f);
	n.param("robot_movement_generation/max_y", this->max_y, 0.0f);
    n.param("robot_movement_generation/participant_file", this->participant_file, std::string("/home/ttsitos/catkin_ws/src/human_robot_collaborative_learning/games_info/baseline/thanasis"));
	n.param("robot_movement_generation/start_position_0", this->start_pos_0, std::vector<float>(0));
	n.param("robot_movement_generation/start_position_1", this->start_pos_1, std::vector<float>(0));
	n.param("robot_movement_generation/position_tolerance", this->position_tolerance, 0.0f);
	n.param("robot_movement_generation/velocity_tolerance", this->velocity_tolerance, 0.0f);
	n.param("robot_movement_generation/num_of_games", this->num_of_games, 0);
	
	this->cmd_y = false; 
	this->reset = true; 
	this->stop_y = false;
	this->count_y = 0;
	this->game_locked = true;

    this->duration_file.open(this->participant_file + ".txt");
	std::string headers = "time_duration,travelled_distance\n";
	this->duration_file << headers;
	this->stats_file.open(this->participant_file + "_stats.txt");
	headers = "human_action,ee_pos_y,ee_vel_y,cmd_acc_human,time\n";
	this->stats_file << headers;
	this->reset_count = 0;
	this->pause = false;
	this->play = true;
	this->pause_duration = 0;
	this->travelled_distance = 0;
	this->max_game_duration = 10.0;
	this->timestart = ros::Time::now().toSec(); 
}

// Reset service callback (now non-blocking)
bool AccCommand::reset_game(human_robot_collaborative_learning::Reset::Request& req, human_robot_collaborative_learning::Reset::Response& res){
    this->reset_requested = true;
    this->last_reset_req = req;
    res.success = true;
    return true;
}

// Actual reset logic, called from main loop if reset_requested is true
void AccCommand::process_reset() {
    this->reset = true;
    this->cmd_y = false;
    this->count_y = 0;
    this->stop_y = false;
    this->travelled_distance = 0;
    this->stats_file << "====================\n";

    int chosen_pos_idx = last_reset_req.position % 2;
    this->start_pos = (chosen_pos_idx == 0) ? this->start_pos_0 : this->start_pos_1;

    // Stop robot before moving to start position
    this->cmd_vel->linear.x = 0;
    this->cmd_vel->linear.y = 0;
    this->cmd_vel->linear.z = 0;
    this->pub.publish(*this->cmd_vel);
    ros::Duration(1.0).sleep();

    // Move robot to the chosen start position
    while (distance(this->state, this->start_pos) >= 0.002 && ros::ok()){
        this->cmd_vel->linear.x = 0.5*(this->start_pos[0] - this->state->pose.position.x);
        this->cmd_vel->linear.y = 0.5*(this->start_pos[1] - this->state->pose.position.y);
        this->cmd_vel->linear.z = 0.5*(this->start_pos[2] - this->state->pose.position.z);
        this->pub.publish(*this->cmd_vel);
        ros::Duration(0.008).sleep();
        ros::spinOnce();
    }
    // Stop robot once at start position
    this->cmd_vel->linear.x = 0;
    this->cmd_vel->linear.y = 0;
    this->cmd_vel->linear.z = 0;
    this->pub.publish(*this->cmd_vel);

    ROS_INFO("State reset. Game Start...");
    this->reset = false;
    this->reset_count++;
    this->start_time = ros::Time::now().toSec();
    this->pause_duration = 0;
    this->prev_state.pose.position.y = this->state->pose.position.y;
    this->prev_state.pose.position.x = this->state->pose.position.x;
    this->timeout = false;
    this->reset_requested = false;

    std_msgs::Bool ready_msg;
    ready_msg.data = true;
    robot_ready_pub.publish(ready_msg);
}


void AccCommand::ee_state_callback(const cartesian_state_msgs::PoseTwist::ConstPtr &ee_state){
	this->state = ee_state;
	if (not this->pause){
		if (not this->reset){ 
			// Check for game timeout
			if (ros::Time::now().toSec() - this->start_time >= this->max_game_duration)
				this->timeout = true;

            this->cmd_vel->linear.x = 0; // Assuming X and Z axes are not controlled by human
            this->cmd_vel->linear.z = 0; 
            
			if (this->cmd_y){
				// Calculate travelled distance in Y
				if (this->prev_state.pose.position.y != 0.0f) 
					this->travelled_distance += abs(this->state->pose.position.y - this->prev_state.pose.position.y);
				this->prev_state = *this->state; 
				this->count_y ++;
				
				// Update Y velocity based on acceleration command
				this->cmd_vel->linear.y += 0.008*this->cmd_acc->linear.y;
				// Clamp Y velocity within min/max limits
				if (this->cmd_vel->linear.y < this->min_vel)
					this->cmd_vel->linear.y = this->min_vel;
				else if (this->cmd_vel->linear.y > this->max_vel)
					this->cmd_vel->linear.y = this->max_vel;
				
                float current_y = this->state->pose.position.y;
                float predicted_y = current_y + 0.01 * this->cmd_vel->linear.y; 

				// Logic to stop Y movement if it's going out of bounds
				if (this->stop_y){
					if (current_y < (this->min_y + this->max_y)/float(2))
						if (this->cmd_acc->linear.y <= 0) // If accelerating in opposite direction, stop
							this->cmd_vel->linear.y = 0;
						else // If accelerating towards center, allow movement
							this->stop_y = false;
					else // current_y > center
						if (this->cmd_acc->linear.y >= 0) // If accelerating in opposite direction, stop
							this->cmd_vel->linear.y = 0;
						else // If accelerating towards center, allow movement
							this->stop_y = false;
					this->count_y = 0; // Reset counter after stopping
				}
				// If movement is active and predicted position is out of bounds, stop and set stop_y flag
				else if (this->count_y > 50 && (predicted_y > this->max_y || predicted_y < this->min_y)){
					this->cmd_vel->linear.y = 0;
					this->stop_y = true;
				}
			}
            
            // Check for game end condition (goal reached or timeout)
			if ((abs(this->state->pose.position.y - 0.256) < this->position_tolerance && abs(this->state->twist.linear.y) < this->velocity_tolerance) || this->timeout){
				bool win_condition = (abs(this->state->pose.position.y - 0.256) < this->position_tolerance && abs(this->state->twist.linear.y) < this->velocity_tolerance);
				ROS_WARN(win_condition ? "YOU WIN" : "YOU LOSE");
				this->reset = true; // Set reset flag to true to prepare for next game
				this->end_time = ros::Time::now().toSec();
				
				this->cmd_vel->linear.x = 0;
				this->cmd_vel->linear.y = 0;
				this->cmd_vel->linear.z = 0;
				this->pub.publish(*this->cmd_vel);


				// Log game duration and travelled distance to file
				std::ostringstream time_duration_oss, trav_dis_oss;
				double duration = this->end_time - this->start_time - this->pause_duration;
				time_duration_oss << duration;
				trav_dis_oss << this->travelled_distance;
				this->duration_file << time_duration_oss.str() << "," << trav_dis_oss.str() << "\n";

                // Publish game status (win/lose) using std_msgs::Bool
                std_msgs::Bool game_status_msg;
                game_status_msg.data = win_condition;
                this->game_status_pub.publish(game_status_msg);

                // Publish game statistics (duration, travelled distance) using std_msgs::Float64MultiArray
                std_msgs::Float64MultiArray game_stats_msg;
                game_stats_msg.data.push_back(duration);
                game_stats_msg.data.push_back(this->travelled_distance);
                this->game_stats_pub.publish(game_stats_msg);
			}
		}
		this->pub.publish(*this->cmd_vel); // Publish velocity command    
	}
	else{
		this->pub.publish(*this->zero_vel); // Publish zero velocity if paused
	}
}

void AccCommand::human_callback(const geometry_msgs::Twist::ConstPtr &msg){
	// Pause/Unpause mechanism
	if (msg->linear.x == 0.5f && msg->angular.z == 1.0f){ 
		this->cmd_vel->linear.y = 0; // Stop Y movement on pause
		this->pause = not this->pause;
		this->start_pause = ros::Time::now().toSec();
	}
	else{
		if (not this->reset){ // Only process human input if game is not in reset state
			if (this->pause){ // If coming out of pause, account for pause duration
				this->end_pause = ros::Time::now().toSec();
				this->pause_duration += this->end_pause - this->start_pause;
			}
			this->pause = false;
			// Convert human input (linear.x) to acceleration command for Y-axis
			this->cmd_acc->linear.y = msg->linear.x / float(5); 
			this->cmd_y = true; // Enable Y-axis control
			
			// Log human action and robot state
			std::ostringstream human_action, ee_pos_y, ee_vel_y, cmd_acc_human, time_val; 
			human_action << msg->linear.x; // Log the raw human input
			ee_pos_y << this->state->pose.position.y;
			ee_vel_y << this->state->twist.linear.y;
			cmd_acc_human << this->cmd_acc->linear.y; // Log the actual acceleration command
			time_val << ros::Time::now().toSec() - this->start_time - this->pause_duration;
			this->stats_file << human_action.str() << "," << ee_pos_y.str() << "," << ee_vel_y.str() << "," << cmd_acc_human.str() << "," << time_val.str() << "\n";
		}
	}
}

void AccCommand::game_control_lock_callback(const std_msgs::Bool::ConstPtr &msg){
	this->game_locked = msg->data;
	ROS_INFO("Game control locked status received: %s", this->game_locked ? "LOCKED" : "UNLOCKED");
}

void AccCommand::run(){
	// Subscribe to human command and robot state
	this->human_sub = this->n.subscribe("cmd_vel", 100, &AccCommand::human_callback, this);
	this->state_sub = this->n.subscribe("ur3_cartesian_velocity_controller/ee_state", 100, &AccCommand::ee_state_callback, this);
	this->experiment_finished_sub = this->n.subscribe("experiment_finished", 100, &AccCommand::experiment_finished_callback, this);
	this->game_control_lock_sub = this->n.subscribe("game_control_lock", 1, &AccCommand::game_control_lock_callback, this);

    ros::Rate loop_rate(100); // Loop at 100 Hz
	while (ros::ok() && this->play){ // Keep running as long as ROS is okay and play flag is true
		ros::spinOnce(); // Process all pending callbacks
        if (this->reset_requested) {
            this->process_reset();
        }
        loop_rate.sleep(); // Maintain loop frequency
    }
	// Close log files when program finishes
	this->duration_file.close();
	this->stats_file.close();
}

void AccCommand::experiment_finished_callback(const std_msgs::Bool::ConstPtr &msg){
    if (msg->data == true){
        ROS_INFO("Experiment finished signal received. Freezing robot and shutting down.");
        this->cmd_vel->linear.x = 0;
        this->cmd_vel->linear.y = 0;
        this->cmd_vel->linear.z = 0;
        this->pub.publish(*this->cmd_vel); // Publish zero velocity immediately
        this->play = false; // Stop the main loop in run()
        ros::shutdown(); // Also shut down the ROS node
    }
}	

int main(int argc, char** argv){
    ros::init(argc, argv, "robot_motion_generator");
    AccCommand robot_control; 
    robot_control.run();
    return 0;
}