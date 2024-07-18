#include <ros.h>
#include <std_msgs/Float64MultiArray.h>

// ROS 노드 핸들러 생성
ros::NodeHandle nh;

// 토픽으로 보낼 메시지 생성
std_msgs::Float64MultiArray current_state_msg;

// publish할 토픽 설정
ros::Publisher pub("current_state", &current_state_msg);

// 아날로그 핀 설정
const int sensorPin = A4;

void setup() {
  // ROS 노드 초기화
  nh.initNode();

  // 토픽을 ROS 네트워크에 advertise
  nh.advertise(pub);

  // 메시지 배열 크기 설정
  current_state_msg.data_length = 1;
  current_state_msg.data = (float*)malloc(sizeof(float) * current_state_msg.data_length);
}

void loop() {
  // 가변저항 값 읽기
  int sensorValue = analogRead(sensorPin);

  // 읽은 값을 0.0 ~ 1023.0 범위의 float 타입으로 변환
  current_state_msg.data[0] = (float)sensorValue;

  // 토픽 publish
  pub.publish(&current_state_msg);

  // ROS 네트워크에 메시지 전송
  nh.spinOnce();

  // 일정 시간 대기 (여기서는 100ms)
  delay(100);
}
