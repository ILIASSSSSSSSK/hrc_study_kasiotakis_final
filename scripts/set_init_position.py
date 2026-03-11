#!/usr/bin/env python3.6
import time
import rospy

from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelConfiguration
from controller_manager_msgs.srv import SwitchController
from std_msgs.msg import Float64MultiArray

def set_model_configuration_client(model_name, model_param_name, joint_names, joint_positions, gazebo_namespace):
    rospy.wait_for_service(gazebo_namespace+'/set_model_configuration')
    time.sleep(2)
    try:
        set_model_configuration = rospy.ServiceProxy(gazebo_namespace+'/set_model_configuration', SetModelConfiguration)
        resp = set_model_configuration(model_name, model_param_name, joint_names, joint_positions)
        unpause = rospy.ServiceProxy(gazebo_namespace+"/unpause_physics", Empty)
        time.sleep(2)
        unpause()
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
            # Wait for the service to be available
    
    

    
rospy.init_node("set_init_position_node")
set_model_configuration_client("robot", "robot_description", ["elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], [-1.5949791113482874, -1.9096925894366663, -0.22891742387880498, -2.7291744391070765, -1.9404695669757288, 0.4364197850227356], "gazebo")

#[-1.8, -2.2, -0.3, -2.2, -1.5, 0.5]

 
#forget set_model_configuration
#use rosservice  call /controller_manager/switch_controller "{start_controllers: ['joint_group_position_controller'], stop_controllers: [ur3_cartesian_velocity_controller_sim], strictness: 2}". express it in python 
#with that way you enable joint_group_position_controller which controls the positions of all the joint!. ur3_cartesian_velocity_controller_sim controller is disabled because it is antigonising the joint_group_position_controller
#remove reset() to reinitialise the game: our controller does it for you!
#mission: find a way to express in python rostopic pub -1 /joint_group_position_controller/command std_msgs/Float64MultiArray "data: [-0.7467053572284144, -2.423556152974264, 0.6513956228839319, -3.0507639090167444, -2.3643069903003138, 0.4621802568435669]"
#be carefull: inside the data the joints are with that order [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]
#show the configurations are the following:
"""
["elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
position0_config=[-1.3772237936602991, -2.057300869618551, -0.9691837469684046, -2.749833885823385, -2.679509703313009, 0.5080214142799377]
position1_config=[-0.7467053572284144, -2.423556152974264, -0.6513956228839319, -3.0507639090167444, -2.3643069903003138, 0.4621802568435669]
position2_config=[-2.149219814931051, -1.411879841481344, -0.48472386995424444, -2.6642029921161097, -2.1951726118670862, 0.45479273796081543]
position3_config=[-1.5949791113482874, -1.9096925894366663, -0.22891742387880498, -2.7291744391070765, -1.9404695669757288, 0.4364197850227356]

i want them in this style now:

position0_config=[ -0.9691837469684046, -2.057300869618551, -1.3772237936602991, -2.749833885823385, -2.679509703313009, 0.5080214142799377]
position1_config=[ -0.6513956228839319, -2.423556152974264,-0.7467053572284144, -3.0507639090167444, -2.3643069903003138, 0.4621802568435669]
position2_config=[-0.48472386995424444, -1.411879841481344, -2.149219814931051, -2.6642029921161097, -2.1951726118670862, 0.45479273796081543]
position3_config=[-0.22891742387880498, -1.9096925894366663, -1.5949791113482874,-2.7291744391070765, -1.9404695669757288, 0.4364197850227356]
"""



