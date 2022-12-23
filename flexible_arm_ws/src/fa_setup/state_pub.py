#!/usr/bin/env python

import numpy as np
from math import pi, sin

import rospy
from sensor_msgs.msg import JointState


def publisher():
    # declares where my node is going to publish and which message type
    # queue_size limits amount of queue messages in any subscribes is not receinving
    # it fast enough
    my_publisher = rospy.Publisher('/joint_states', JointState, queue_size = 10)
    # tells rospy the name of my node
    rospy.init_node('my_node', anonymous=False)

    # define rate
    rate = rospy.Rate(100)

    # joint names
    active_joints = ['active_joint'+str(i) for i in range(1, 4)]
    passive_joints_link2 = ['passive_joint' + str(i) for i in range(1, 11)]
    passive_joints_link3 = ['passive_joint3_' + str(i) for i in range(1, 11)]
    joint_names = (active_joints[:2] + passive_joints_link2 + 
                   [active_joints[2]] + passive_joints_link3)

    # joint positions
    try:
        abs_path = ("/home/shamil/Desktop/phd/code/flexible_arm/"
                    "flexible_arm_ws/src/fa_setup/js_traj.csv")
        q_ref = np.loadtxt(abs_path, delimiter=',')
        no_samples = q_ref.shape[0]
    except IOError:
        print("Path to reference is wrong!")

    # counter that goes through samples
    k = 0

    # form a message
    msg = JointState()
    msg.name = joint_names
    msg.position = []
    msg.velocity = []
    msg.effort = []

    while not rospy.is_shutdown():
        msg.header.stamp = rospy.Time.now()
        # msg.position[0] = sin(rospy.get_rostime().to_sec())
        msg.position = q_ref[k,:]
        my_publisher.publish(msg)
        if k+1 < no_samples:
            k += 1
        else:
            k = 0
        rate.sleep()


if __name__== '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass