#!/usr/bin/env python3

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

# 버튼과 조이스틱 매핑 정보
buttons = {
    0: 'Select', 1: 'L3', 2: 'R3', 3: 'Start', 4: 'D-pad Up', 5: 'D-pad Right',
    6: 'D-pad Down', 7: 'D-pad Left', 8: 'L2', 9: 'R2', 10: 'L1', 11: 'R1',
    12: 'Triangle', 13: 'Circle', 14: 'Cross', 15: 'Square', 16: 'PS'
}

joysticks = {
    0: 'Left Joystick X', 1: 'Left Joystick Y', 2: 'Right Joystick X', 5: 'Right Joystick Y'
}

# 입력 읽기 및 처리
for event in gamepad.read_loop():
    if event.type == ecodes.EV_KEY:
        button_name = buttons.get(event.code, 'Unknown')
        state = 'pressed' if event.value else 'released'
        print(f'Button {button_name} {event.code} {state}')
    elif event.type == ecodes.EV_ABS:
        joystick_name = joysticks.get(event.code, 'Unknown')
        print(f'Joystick {joystick_name} {event.code} moved to {event.value}')