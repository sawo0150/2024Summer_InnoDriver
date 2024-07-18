#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray

# 조이스틱 버튼 및 축 상태를 나타내는 딕셔너리
button_mapping = {
    314: ['BTN_Select', 0],
    315: ['BTN_Start', 0],
    316: ['BTN_Mode', 0],
    311: ['BTN_R1', 0],
    313: ['BTN_R2', 0],
    310: ['BTN_L1', 0],
    312: ['BTN_L2', 0],
    304: ['BTN_A', 0],
    305: ['BTN_B', 0],
    307: ['BTN_X', 0],
    308: ['BTN_Y', 0]
}

joystick_mapping = {
    2: ['ABS_RX', 128],
    5: ['ABS_RY', 128],
    0: ['ABS_LX', 128],
    1: ['ABS_LY', 128],
    16: ['ABS_LUX', 128],
    17: ['ABS_LUY', 128],
    9: ['ABS_R2', 128],
    10: ['ABS_L2', 128]
}
maxAngle = 14 
maxR_pulse = 255
maxL_pulse = 240

def joy_callback(data):
    global button_mapping, joystick_mapping

    # Joy 메시지로부터 버튼 상태 업데이트
    for i in range(len(data.buttons)):
        button_mapping[list(button_mapping.keys())[i]][1] = data.buttons[i]

    # Joy 메시지로부터 축 상태 업데이트
    for i in range(len(data.axes)):
        joystick_mapping[list(joystick_mapping.keys())[i]][1] = int(data.axes[i] * 256.0)

def joystick_controller():
    # ROS 노드 초기화
    rospy.init_node('joystick_controller', anonymous=True)

    # Joy 메시지를 구독하고 콜백 함수 설정
    rospy.Subscriber('joy', Joy, joy_callback)
    # goal_state 토픽을 게시하는 Publisher 생성
    pub = rospy.Publisher('goal_state', Float64MultiArray, queue_size=10)

    rate = rospy.Rate(10)  # 10Hz

    while not rospy.is_shutdown():
        # goal_state 메시지 생성 및 데이터 채우기
        msg = Float64MultiArray()
        msg.data = []

        steerAngle = (128-joystick_mapping[2][1])/128 
        msg.data.append(steerAngle)

        power = (1-joystick_mapping[1][1] / 255)
        msg.data.append(power)
        
        # goal_state 토픽에 메시지 publish
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        print("Starting joystick_controller node...")
        joystick_controller()
    except rospy.ROSInterruptException:
        pass
