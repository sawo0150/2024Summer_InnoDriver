#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray

class GoalStatePublisherNode:
    def __init__(self):
        self.sub_goal = rospy.Subscriber('calculated_goal_state', Float64MultiArray, self.goal_callback)
        self.pub_goal = rospy.Publisher('goal_state', Float64MultiArray, queue_size=2)
        self.latest_msg = None
        self.publish_rate = 10  # 10 Hz

    def goal_callback(self, msg):
        self.latest_msg = msg

    def publish_loop(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            if self.latest_msg is not None:
                # rospy.loginfo(f"Publishing message: {self.latest_msg.data}")
                self.pub_goal.publish(self.latest_msg)
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('goal_state_publisher_node', anonymous=True)
    node = GoalStatePublisherNode()
    node.publish_loop()
