#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Joy
import threading
import evdev
from evdev import InputDevice, categorize, ecodes

# 연결된 장치 확인
devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
for device in devices:
    print(device.path, device.name, device.phys)
    if 'input0' in device.phys:
        print("'input0' 문자열이 파일에 포함되어 있습니다.")
        device_path = device.path

# PS3 컨트롤러가 연결된 경로 지정
# device_path = '/dev/input/event2'  # PS3 컨트롤러가 연결된 장치 경로로 변경해야 합니다.
gamepad = InputDevice(device_path)


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

def contorller_inform_upadate(event):
    # print("notified", event.code, event.value)
    if event.type == ecodes.EV_KEY:
        btn_list = button_mapping[event.code]
        btn_list[1] = event.value
        button_mapping[event.code] = btn_list
        
    elif event.type == ecodes.EV_ABS:
        Joy_list = joystick_mapping[event.code]
        value = event.value if (event.code != 17 or event.code != 16) else 128*(1+event.value)
        Joy_list[1] = value
        joystick_mapping[event.code] = Joy_list
        
def joystick_publisher():
    # ROS 노드 초기화
    rospy.init_node('joystick_publisher', anonymous=True)
    pub = rospy.Publisher('joy', Joy, queue_size=10)
    rate = rospy.Rate(10)  # 10Hz

    while not rospy.is_shutdown():
        joy_msg = Joy()
        joy_msg.header.stamp = rospy.Time.now()

        # 버튼 상태를 Joy 메시지에 추가
        buttons = [0] * (len(button_mapping.keys()))
        i=0
        for key, value in button_mapping.items():
            buttons[i] = value[1]
            i+=1

        # 축 상태를 Joy 메시지에 추가
        axes = [0.0] * (len(joystick_mapping.keys()))
        i=0
        for key, value in joystick_mapping.items():
            axes[i] = value[1] / 256.0  # 값을 0.0과 1.0 사이로 정규화
            i+=1

        joy_msg.buttons = buttons
        joy_msg.axes = axes

        # 메시지를 publish
        pub.publish(joy_msg)


        # print(joystick_mapping)
        # print(button_mapping)
        rate.sleep()

def read_gamepad():
    for event in gamepad.read_loop():
        contorller_inform_upadate(event)
        print(f'{event.code} moved to {event.value}')


gamepad_thread = threading.Thread(target=read_gamepad)
gamepad_thread.start()

if __name__ == '__main__':
    try:
        print("joystick publish ")
        joystick_publisher()
    except rospy.ROSInterruptException:
        pass
