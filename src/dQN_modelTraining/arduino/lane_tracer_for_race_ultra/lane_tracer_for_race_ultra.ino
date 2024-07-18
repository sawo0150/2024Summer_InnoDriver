
#include <Car_Library.h>
#include <ros.h>
#include <std_msgs/Int16.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float64MultiArray.h>

ros::NodeHandle nh;

// 모터 핀 설정
int motor_left_1 = 9;
int motor_left_2 = 8;
int motor_left_PWM = 10;
int motor_right_1 = 11;
int motor_right_2 = 13;
int motor_right_PWM = 12;
int motor_front_1 = 4;
int motor_front_2 = 5;
int motor_front_PWM = 6;
int analogPin = A4;

// PID 제어 변수
float Kp = 0.2;
float Ki = 0.05;
float Kd = 0.1;
int last_error = 0;
int integral = 0;

// 목표 조향각 및 모터 펄스 값 (subscribe 정보 공유용)
float goal_steer = 0;
float goal_power = 0;
int goal_pulse_left = 0;
int goal_pulse_right = 0;

// 목표 조향각 및 모터 펄스 값 (subscribe 정보 공유용)
int current_steer = 0;
float current_power = 0;

int current_stateArray[2];

// 모터 출력 불균형 custom?
int power_L_MAX = 255;
int power_R_MAX = 220;
float angle_MAX = 23.0;
// left power 170
// right power 230
float val;
float val_max;
float val_min;
float val_mid;

int power = 220;
int TR_front = 11;
int EC_front = 10;
int TR_right = 13;
int EC_right = 12;
int once = 1;

//초음파 센서 setting
const int numSensors = 5;  // 사용할 초음파 센서의 개수
const int trigPins[numSensors] = {24, 26, 28, 30, 32};  // 트리거 핀 배열
const int echoPins[numSensors] = {25, 27, 29, 31, 33};  // 에코 핀 배열
float distances[numSensors];  // 거리 값을 저장할 배열
int ultraIndex = 0;

void Motor_Forward(int Speed, int IN1, int IN2, int PWM) {
     digitalWrite(IN1, HIGH);
     digitalWrite(IN2, LOW);
     analogWrite(PWM, Speed);
}

void Motor_Backward(int Speed, int IN1, int IN2, int PWM) {
     digitalWrite(IN1, LOW);
     digitalWrite(IN2, HIGH);
     analogWrite(PWM, Speed);
}

void Motor_Brake(int IN1, int IN2){
     digitalWrite(IN1, LOW);
     digitalWrite(IN2, LOW);
}

void forward(int power_left, int power_right){
  motor_forward(motor_right_1, motor_right_2, motor_right_PWM, power_right);
  motor_forward(motor_left_1, motor_left_2, motor_left_PWM, power_left);
}
void backward(int power_left, int power_right){
  motor_backward(motor_right_1, motor_right_2,motor_right_PWM, power_right);
  motor_backward(motor_left_1, motor_left_2,motor_left_PWM, power_left);
}
void left_tilt(int power_tilt){
  motor_forward(motor_front_1, motor_front_2, motor_front_PWM, power_tilt);
}
void right_tilt(int power_tilt){
  motor_backward(motor_front_1, motor_front_2, motor_front_PWM, power_tilt);
}
void stop_tilt(){
  motor_hold(motor_front_1, motor_front_2);
}
void pause(){
  motor_hold(motor_right_1,motor_right_2);
  motor_hold(motor_left_1, motor_left_2);
}
int myRound(float number) {
    return (number >= 0) ? (int)(number + 0.5) : (int)(number - 0.5);
}

void change_angle(float angle){
  // int current_angle = (potentiometer_Read(analogPin) - val_mid);
  // int error = angle - current_angle;
  // integral += error;
  // int derivative = error - last_error;
  // int output = Kp * error + Ki * integral + Kd * derivative;
  // last_error = error;

  // if (output > 0) {
  //   motor_backward(motor_front_1, motor_front_2,motor_front_PWM, min(-output, 255));
  // } else if (output < 0) {
  //   motor_forward(motor_front_1, motor_front_2,motor_front_PWM, min(output, 255));
  // } else {
  //   motor_hold(motor_front_1, motor_front_2);
  //   integral = 0;
  // }


  if((potentiometer_Read(analogPin) - val_mid) >= (angle+2)){
    right_tilt(power);
  }else if((potentiometer_Read(analogPin) - val_mid) <= (angle-2)){
    left_tilt(power);
  }else{
    stop_tilt();
  }

  // if (angle > 1.0){
  //   right_tilt();
  // } else if (angle < -1.0){
  //   left_tilt();
  // } else{
  //   stop_tilt();
  // }

}

void motor(const std_msgs::Float64MultiArray& msg) {
  goal_steer = msg.data[0]*angle_MAX;
  goal_power = msg.data[1];
  goal_pulse_left = myRound(power_L_MAX *goal_power);
  goal_pulse_right = myRound(power_R_MAX * goal_power);

}

void read_ultrasonics(int index)
{
  distances[index] = ultrasonic_distance(trigPins[index], echoPins[index]);
}
// ROS 구독자
ros::Subscriber<std_msgs::Float64MultiArray> sub("goal_state", motor);
// ROS 발행자
std_msgs::Float64MultiArray current_state_msg;
ros::Publisher pub1("current_state", &current_state_msg);
std_msgs::Float64MultiArray distance_msg;
ros::Publisher pub2("ultraDistance", &distance_msg);

void setup() {

  Serial.begin(57600);

  pinMode(motor_left_1,OUTPUT);
  pinMode(motor_left_2,OUTPUT);
  pinMode(motor_left_PWM,OUTPUT);
  pinMode(motor_right_1,OUTPUT);
  pinMode(motor_right_2,OUTPUT);
  pinMode(motor_right_PWM,OUTPUT);
  pinMode(motor_front_1,OUTPUT);
  pinMode(motor_front_2,OUTPUT);
  pinMode(motor_front_PWM,OUTPUT);

  pinMode(analogPin,INPUT);
  // 트리거 및 에코 핀을 출력/입력 모드로 설정
  for (int i = 0; i < numSensors; i++) {
      pinMode(trigPins[i], OUTPUT);
      pinMode(echoPins[i], INPUT);
  }
  
  left_tilt(200);
  delay(4000);
  stop_tilt();
  val_max= potentiometer_Read(analogPin);
  

  // Motor_Forward(255, motor_right_1, motor_right_2, motor_right_PWM);  // Forward, PWM setting 0-255
  // Motor_Forward(255, motor_left_1, motor_left_2, motor_left_PWM);  
  
  right_tilt(200);
  delay(4000);
  stop_tilt();
  Motor_Brake(motor_right_1, motor_right_2);  // 인수 추가
  Motor_Brake(motor_left_1, motor_left_2);  // 인수 추가
  val_min= potentiometer_Read(analogPin);
  
  val_mid = (val_max + val_min)/2.0;
  
  nh.initNode();
  nh.subscribe(sub);
  nh.advertise(pub1);
  nh.advertise(pub2);

  current_state_msg.data_length = 2;
  current_state_msg.data = (float*)malloc(sizeof(float) * current_state_msg.data_length);
  // 메시지 데이터 배열 크기 설정
  distance_msg.data_length = numSensors;
  distance_msg.data = (float*)malloc(sizeof(float) * numSensors);
}
unsigned long previousMillis = 0; // 마지막으로 초음파 센서를 읽은 시간
const long interval = 300; // 1초 간격

void loop() {
  unsigned long currentMillis = millis();

  // 1초 간격으로 초음파 센서 값 읽기
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;
    // 초음파 센서 값 읽기
    read_ultrasonics(ultraIndex);
    distance_msg.data[ultraIndex] = distances[ultraIndex];
    ultraIndex+=1;
    if (ultraIndex>4){
      ultraIndex = 0;
      pub2.publish(&distance_msg);
    }
  }
  // 현재 조향각을 읽어서 퍼블리시
  current_state_msg.data[0] = (potentiometer_Read(analogPin) - val_mid)/float(angle_MAX);  
  // current_state_msg.data[0] = potentiometer_Read(analogPin)- val_mid;
  current_state_msg.data[1] = current_power;
  // 메시지 퍼블리시
  pub1.publish(&current_state_msg);

  // change_angle(goal_steer);

  // delay를 사용하여 PID 제어 주기를 조절할 수 있습니다.
  // 필요에 따라 PID 제어의 빈도를 더 높이거나 낮출 수 있습니다.

  for(int i=0; i<5; i++){
    change_angle(goal_steer);
    delay(5); // 예시로 0.01초마다 PID 제어를 수행합니다.
  }

  Motor_Forward(goal_pulse_right, motor_right_1, motor_right_2, motor_right_PWM);  // Forward, PWM setting 0-255
  Motor_Forward(goal_pulse_left, motor_left_1, motor_left_2, motor_left_PWM);  
  
  // forward(goal_pulse_left, goal_pulse_right);
  current_power = goal_power;
  
  nh.spinOnce();
}
