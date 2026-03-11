#include "human_robot_collaborative_learning/robot_control_sign.h"
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <iostream>
#include <fstream>
#include <ros/ros.h>
double target_z=0.2226;

AccCommand::AccCommand(){
	this->spinner = boost::make_shared<ros::AsyncSpinner>(3);
	bool simulation;
	n.param("/rl_control/Game/gazebo_simulation", simulation, false);
	if(not simulation)
	{	
		this->pub = this->n.advertise<geometry_msgs::Twist>("ur3_cartesian_velocity_controller/command_cart_vel", 100);
	}
	else
	{
	    this->pub = this->n.advertise<geometry_msgs::Twist>("ur3_cartesian_velocity_controller_sim/command_cart_vel", 100);
	}
	this->tmp_time = 0;
	this->state = boost::make_shared<cartesian_state_msgs::PoseTwist>();
	this->cmd_vel = boost::make_shared<geometry_msgs::Twist>();
	this->zero_vel = boost::make_shared<geometry_msgs::Twist>();
	this->cmd_acc = boost::make_shared<geometry_msgs::Accel>();
	this->service = n.advertiseService("reset", &AccCommand::reset_game, this);
	this->cmd_x = false, this->cmd_y = false, this->reset = false, this->train = false;
	this->stop_x = false, this->stop_y = false;
	this->count_x = 0, this->count_y = 0;
	this->last_time = 0.0f;
	n.param("robot_movement_generation/min_vel", this->min_vel, 0.0f);
	n.param("robot_movement_generation/max_vel", this->max_vel, 0.0f);
	n.param("robot_movement_generation/min_x", this->min_x, 0.0f);
	n.param("robot_movement_generation/max_x", this->max_x, 0.0f);
	n.param("robot_movement_generation/min_y", this->min_y, 0.0f);
	n.param("robot_movement_generation/max_y", this->max_y, 0.0f);
	n.param("robot_movement_generation/start_position_0", this->start_position_0, std::vector<float>(0.0f));
	n.param("robot_movement_generation/start_position_1", this->start_position_1, std::vector<float>(0.0f));
	n.param("robot_movement_generation/start_position_2", this->start_position_2, std::vector<float>(0.0f));
	n.param("robot_movement_generation/start_position_3", this->start_position_3, std::vector<float>(0.0f));
	n.param("robot_movement_generation/start_position_4", this->start_position_4, std::vector<float>(0.0f));
	n.param("robot_movement_generation/start_position_5", this->start_position_5, std::vector<float>(0.0f));
	n.param("robot_movement_generation/start_position_6", this->start_position_6, std::vector<float>(0.0f));
	n.param("robot_movement_generation/start_position_7", this->start_position_7, std::vector<float>(0.0f));
	this->start_positions.push_back(this->start_position_0);
	this->start_positions.push_back(this->start_position_1);
	this->start_positions.push_back(this->start_position_2);
	this->start_positions.push_back(this->start_position_3);
	this->start_positions.push_back(this->start_position_4);
	this->start_positions.push_back(this->start_position_5);
	this->start_positions.push_back(this->start_position_6);
	this->start_positions.push_back(this->start_position_7);
	this->spinner->start();
}


bool AccCommand::reset_game(human_robot_collaborative_learning::Reset::Request& req, human_robot_collaborative_learning::Reset::Response& res){
	int r_pos = req.position;
	this->reset = true;
	this->cmd_x = false;
	this->cmd_y = false;
    srand(time(NULL));
	std::vector<float> start_pos = this->start_positions[r_pos];
	this->cmd_vel->linear.x = 0;
	this->cmd_vel->linear.y = 0;
	this->cmd_vel->linear.z = 0;
	this->pub.publish(*this->cmd_vel);
	ros::Duration(5).sleep();
	while (distance(this->state, start_pos) >= 0.0005){
		this->cmd_vel->linear.x = start_pos[0] - this->state->pose.position.x;
		this->cmd_vel->linear.y = start_pos[1] - this->state->pose.position.y;
		this->cmd_vel->linear.z = start_pos[2] - this->state->pose.position.z;
		this->pub.publish(*this->cmd_vel);
		ros::Duration(0.008).sleep();
	}
	this->cmd_vel->linear.x = 0;
	this->cmd_vel->linear.y = 0;
	this->cmd_vel->linear.z = 0;
	this->pub.publish(*this->cmd_vel);
	ROS_INFO("Successfully reset the game");
    this->reset = false;
	return true;
}


void AccCommand::ee_state_callback(const cartesian_state_msgs::PoseTwist::ConstPtr &ee_state){
	this->state = ee_state;
	if (not this->reset){
		if (not this->train){
			if (this->cmd_x){
				this->count_x ++;
				this->cmd_vel->linear.x += 0.008*this->cmd_acc->linear.x;
				if (this->cmd_vel->linear.x < this->min_vel)
					this->cmd_vel->linear.x = this->min_vel;
				else if (this->cmd_vel->linear.x > this->max_vel)
					this->cmd_vel->linear.x = this->max_vel;
				if (this->stop_x){
					if (this->state->pose.position.x < (this->min_x + this->max_x)/float(2))
						if (this->cmd_acc->linear.x <= 0)
							this->cmd_vel->linear.x = 0;
						else
							this->stop_x = false;
					else
						if (this->cmd_acc->linear.x >= 0)
							this->cmd_vel->linear.x = 0;
						else
							this->stop_x = false;
					this->count_x = 0;
				}
				else if (this->count_x > 50 and (this->state->pose.position.x + 0.01*this->cmd_vel->linear.x > this->max_x or this->state->pose.position.x + 0.01*this->cmd_vel->linear.x < this->min_x)){
					this->cmd_vel->linear.x = 0;
					this->stop_x = true;
				}
			}
			
			bool simulation;
			n.param("/rl_control/Game/gazebo_simulation", simulation, false);

			if (simulation and (this->state->pose.position.z < target_z)) 
            {
            this->cmd_vel->linear.z = 0.01;  // Move up
            } 
            else if (simulation and (this->state->pose.position.z > target_z)) {
            this->cmd_vel->linear.z = -0.01; // Move down
            } else 
            { 
            this->cmd_vel->linear.z = 0; // Stop moving when at target
            }
            
			if (this->cmd_y){
				this->count_y ++;
				this->cmd_vel->linear.y += 0.008*this->cmd_acc->linear.y;
				//std::cout<<(this->cmd_acc->linear.y);                
				if (this->cmd_vel->linear.y < this->min_vel)
					this->cmd_vel->linear.y = this->min_vel;
				else if (this->cmd_vel->linear.y > this->max_vel)
					this->cmd_vel->linear.y = this->max_vel;
				if (this->stop_y){
					if (this->state->pose.position.y < (this->min_y + this->max_y)/float(2))
						if (this->cmd_acc->linear.y <= 0)
							this->cmd_vel->linear.y = 0;
						else
							this->stop_y = false;
					else
						if (this->cmd_acc->linear.y >= 0)
							this->cmd_vel->linear.y = 0;
						else
							this->stop_y = false;
					this->count_y = 0;
				}
				else if (this->count_y > 50 and (this->state->pose.position.y + 0.01*this->cmd_vel->linear.y > this->max_y or this->state->pose.position.y + 0.01*this->cmd_vel->linear.y < this->min_y)){
					this->cmd_vel->linear.y = 0;
					this->stop_y = true;
				}
			}
			//this->cmd_vel->linear.z = 0;
			
			this->pub.publish(*this->cmd_vel);
		}
		else{
			this->pub.publish(*this->zero_vel);
		}
	}
}
void AccCommand::agent_callback(const std_msgs::Float64::Ptr &msg){
	if (msg->data == 2.0f){
		msg->data = -1.0f;
	}
	this->cmd_acc->linear.x = msg->data / float(5);
	this->cmd_x = true;
}

void AccCommand::human_callback(const geometry_msgs::Twist::ConstPtr &msg){
	this->last_time = ros::Time::now().toSec();
	if (not this->train and not this->reset){
		this->cmd_acc->linear.y = msg->linear.x / float(5);
		this->cmd_y = true;
	}
}

void AccCommand::train_callback(const std_msgs::Bool::ConstPtr &msg){
	this->train = msg->data;
}

void AccCommand::run(){
	this->human_sub = this->n.subscribe("cmd_vel", 100, &AccCommand::human_callback, this);
	this->agent_sub = this->n.subscribe("agent_action_topic", 100, &AccCommand::agent_callback, this);
	bool simulation;
	n.param("/rl_control/Game/gazebo_simulation", simulation, false);
	if(not simulation)
	{	
	this->state_sub = this->n.subscribe("ur3_cartesian_velocity_controller/ee_state", 100, &AccCommand::ee_state_callback, this);
	}
	else
	{	
	this->state_sub = this->n.subscribe("ur3_cartesian_velocity_controller_sim/ee_state", 100, &AccCommand::ee_state_callback, this);
	}
	this->train_sub = this->n.subscribe("train_topic", 100, &AccCommand::train_callback, this);
	ros::waitForShutdown();
}

// ILIAS VERSION
// bool AccCommand::reset_game(human_robot_collaborative_learning::Reset::Request& req, human_robot_collaborative_learning::Reset::Response& res){
// 	this->reset = true;
// 	this->cmd_x = false;
// 	this->cmd_y = false;
//     srand(time(NULL));
//     //My code
//     /*
//     int i;
    
//     std::ifstream infile("/home/kassiotakis/Desktop/catkin_ws5/src/hrc_study_tsitosetal/src/position.txt");

//     // Check if the file was opened successfully
//     if (infile.is_open()) {
//         // Read the integer from the file
//         infile >> i;

//         // Check if reading was successful
//         if (infile) {
//             std::cout << "The integer read from the file is: " << i << std::endl;
//         } else {
//             std::cerr << "Error reading integer from the file." << std::endl;
//         }

//         // Close the file
//         infile.close();
//     } else {
//         std::cerr << "Error opening the file." << std::endl;
//     }
    
//     //
    
//     //std::cout<<"Give me an initial position to start";
//     //std::cin>>i;
//     */
//     ros::NodeHandle nh;
//     bool random_position;
//     if (nh.getParam("rl_control/Game/random_position", random_position))
//     {
//         if(random_position)
//         {
//          int i=rand() % start_positions.size();
//          std::vector<float> start_pos = this->start_positions[i];
//          std::cout<<"I am going to position: "<<i<<std::endl;
//          std::cout<<"I am going to position with coordinates: "<<std::endl;
// 		 std::cout<<"x: "<<start_pos[0]<<std::endl;
// 		 std::cout<<"y: "<<start_pos[1]<<std::endl;
// 		 std::cout<<"z: "<<start_pos[2]<<std::endl;
//          this->cmd_vel->linear.x = 0;
//          this->cmd_vel->linear.y = 0;
//          this->cmd_vel->linear.z = 0;
//          this->pub.publish(*this->cmd_vel);
//          ros::Duration(5).sleep();
//          while (distance(this->state, start_pos) >= 0.0005){
// 		 this->cmd_vel->linear.x = start_pos[0] - this->state->pose.position.x;
// 		 this->cmd_vel->linear.y = start_pos[1] - this->state->pose.position.y;
// 		 this->cmd_vel->linear.z = start_pos[2] - this->state->pose.position.z;
// 		 this->pub.publish(*this->cmd_vel);
// 		 ros::Duration(0.008).sleep();
//          }
//          this->cmd_vel->linear.x = 0;
//          this->cmd_vel->linear.y = 0;
//          this->cmd_vel->linear.z = 0;
//          this->pub.publish(*this->cmd_vel);
//          ROS_INFO("Successfully reset the game");
//          this->reset = false;
//          return true;
            
//         }
//         else
//         {
//          int i;
//          if (nh.getParam("rl_control/Game/init_cyclic_pos", i))
//           {
//          std::vector<float> start_pos = this->start_positions[i];
//          std::cout<<"I am going to position: "<<i<<std::endl;
//          std::cout<<"I am going to position with coordinates: "<<std::endl;
// 		 std::cout<<"x: "<<start_pos[0]<<std::endl;
// 		 std::cout<<"y: "<<start_pos[1]<<std::endl;
// 		 std::cout<<"z: "<<start_pos[2]<<std::endl;
//          this->cmd_vel->linear.x = 0;
//          this->cmd_vel->linear.y = 0;
//          this->cmd_vel->linear.z = 0;
//          this->pub.publish(*this->cmd_vel);
//          ros::Duration(5).sleep();
//          while (distance(this->state, start_pos) >= 0.0005){
// 		 this->cmd_vel->linear.x = start_pos[0] - this->state->pose.position.x;
// 		 this->cmd_vel->linear.y = start_pos[1] - this->state->pose.position.y;
// 		 this->cmd_vel->linear.z = start_pos[2] - this->state->pose.position.z;
// 		 this->pub.publish(*this->cmd_vel);
// 		 ros::Duration(0.008).sleep();
//          }
//          this->cmd_vel->linear.x = 0;
//          this->cmd_vel->linear.y = 0;
//          this->cmd_vel->linear.z = 0;
//          this->pub.publish(*this->cmd_vel);
//          ROS_INFO("Successfully reset the game");
//          i=i+1;
//          if(i>3)
//          {
//              i=0;
//         }
//          nh.setParam("/rl_control/Game/init_cyclic_pos", i);
//          this->reset = false;
//          return true;  
//           }
//         }
//     }

// }

