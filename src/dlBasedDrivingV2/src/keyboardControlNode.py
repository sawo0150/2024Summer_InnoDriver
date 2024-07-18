#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool
import pygame

class KeyboardControlNode:
    def __init__(self):
        rospy.init_node('keyboard_control_node', anonymous=True)
        
        # Publisher
        self.pub = rospy.Publisher('control_signal', Bool, queue_size=2)
        
        # Initialize pygame for keyboard control
        pygame.init()
        self.screen = pygame.display.set_mode((100, 100))  # Small window for capturing events
        self.clock = pygame.time.Clock()
        
        # State variable
        self.running = False

    def run(self):
        try:
            while not rospy.is_shutdown():
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.running = not self.running
                            self.pub.publish(self.running)
                            rospy.loginfo(f"Driving state changed: {'Running' if self.running else 'Stopped'}")
                self.clock.tick(5)  # Limit the loop to 10 frames per second
        except rospy.ROSInterruptException:
            pass
        finally:
            pygame.quit()

if __name__ == '__main__':
    try:
        keyboard_control_node = KeyboardControlNode()
        keyboard_control_node.run()
    except rospy.ROSInterruptException:
        pass
