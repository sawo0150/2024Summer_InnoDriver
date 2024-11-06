//#include <Car_Library.h>
#include <ros.h>
// #include <std_msgs/Int16.h>
#include <std_msgs/Int32.h>
// #include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Bool.h>

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
int echo = 2;
int trig = 3;
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
int power_R_MAX = 255;
float angle_MAX = 14.0;
// left power 170
// right power 230
float val;
float val_max;
float val_min;
float val_mid;

int power = 150;
//int TR_front = 11;
//int EC_front = 10;
//int TR_right = 13;
//int EC_right = 12;
//int once = 1;
float distance_ac;
//초음파 센서 setting
//const int numSensors = 3;  // 사용할 초음파 센서의 개수
//const int trigPins[numSensors] = {2, 4, 6};  // 트리거 핀 배열
//const int echoPins[numSensors] = {3, 5, 7};  // 에코 핀 배열
//float distances[numSensors];  // 거리 값을 저장할 배열
int i = 0;

bool obstacle_detected = false;
int obstacle_detected_num = 0;

bool isFirstParking = true;
bool control_signal = false;

void obstacle_callback(const std_msgs::Int32& msg) {
  obstacle_detected_num = msg.data;
}

void control_signal_callback(const std_msgs::Bool& msg) {
  control_signal = msg.data;
}

ros::Subscriber<std_msgs::Int32> sub_obs("obstacle_detected", obstacle_callback);
ros::Subscriber<std_msgs::Bool> sub_cont("control_signal", control_signal_callback);

float potentiometer_Read(int pin)
{
    int value;
    // 정확한 각도 계산 : 270도 = 1024
    value = float(analogRead(pin)) / 1024.0 *270.0;

    return value;
}


float ultrasonic_distance(int trigPin, int echoPin)
{
    float cycletime;
    float distance;

    digitalWrite(trigPin, LOW);
    digitalWrite(echoPin, LOW);
    delayMicroseconds(2);
  
    digitalWrite(trigPin, HIGH);
    delay(10);
    digitalWrite(trigPin, LOW);
  
    cycletime = pulseIn(echoPin, HIGH); 
  
    distance = ((340 * cycletime) / 10000) / 2;  
    Serial.print("Distance:");
    Serial.print(distance);
    Serial.println("cm");

    return distance;
}


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
  Motor_Forward(power_right, motor_right_1, motor_right_2, motor_right_PWM);
  Motor_Forward(power_left, motor_left_1, motor_left_2, motor_left_PWM);
}
void backward(int power_left, int power_right){
  Motor_Backward(power_right, motor_right_1, motor_right_2, motor_right_PWM);
  Motor_Backward(power_left, motor_left_1, motor_left_2, motor_left_PWM);
}
void left_tilt(int power_tilt){
  Motor_Forward(power_tilt, motor_front_1, motor_front_2, motor_front_PWM);
}
void right_tilt(int power_tilt){
  Motor_Backward(power_tilt, motor_front_1, motor_front_2, motor_front_PWM);
}
void stop_tilt(){
  Motor_Brake(motor_front_1, motor_front_2);
}
void pause(){
  Motor_Brake(motor_right_1,motor_right_2);
  Motor_Brake(motor_left_1, motor_left_2);
}
int myRound(float number) {
    return (number >= 0) ? (int)(number + 0.5) : (int)(number - 0.5);
}

void parking(){
  for(int i=0; i<165; i++){
    change_angle(0);
    delay(10); // 예시로 0.01초마다 PID 제어를 수행합니다.
  }
  pause();
  for(int i=0; i<100; i++){
    change_angle(23);
    delay(10); // 예시로 0.01초마다 PID 제어를 수행합니다.
  }
  delay(1500);
  forward(40, 40);
  delay(7000);
  pause();
  for(int i=0; i<110; i++){
    change_angle(-23);
    delay(10); // 예시로 0.01초마다 PID 제어를 수행합니다.
  }
  delay(2000);
  backward(40, 40);
  delay(1900);
  for(int i=0; i<980; i++){
    change_angle(0);
    delay(10); // 예시로 0.01초마다 PID 제어를 수행합니다.
  }
  pause();
  delay(6000);
  i += 1;
  forward(40, 40);
  for(int i=0; i<310; i++){
    change_angle(0);
    delay(10); // 예시로 0.01초마다 PID 제어를 수행합니다.
  }
  for(int i=0; i<900; i++){
    change_angle(-23);
    delay(10); // 예시로 0.01초마다 PID 제어를 수행합니다.
  }
  for(int i=0; i<250; i++){
    change_angle(0);
    delay(10); // 예시로 0.01초마다 PID 제어를 수행합니다.
  }
  pause();
  delay(2000);
  forward(40, 40);
  for(int i=0; i<1600; i++){
    change_angle(0);
    delay(10); // 예시로 0.01초마다 PID 제어를 수행합니다.
  }
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


  if((potentiometer_Read(analogPin) - val_mid) >= (angle + 1.5 )){
    right_tilt(power);
  }else if((potentiometer_Read(analogPin) - val_mid) <= (angle - 1.5)){
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

//void motor(const std_msgs::Float64MultiArray& msg) {
  //goal_steer = msg.data[0]*angle_MAX;
  //goal_power = msg.data[1];
  //goal_pulse_left = myRound(power_L_MAX *goal_power);
  //goal_pulse_right = myRound(power_R_MAX * goal_power);

//}

//void read_multiple_ultrasonics()
//{
  //  for (int i = 0; i < numSensors; i++) {
    //    distances[i] = ultrasonic_distance(trigPins[i], echoPins[i]);
    //}
//}
// ROS 구독자
//ros::Subscriber<std_msgs::Float64MultiArray> sub("goal_state", motor);
// ROS 발행자
//std_msgs::Float64MultiArray current_state_msg;
//ros::Publisher pub1("current_state", &current_state_msg);
//std_msgs::Float64MultiArray distance_msg;
//ros::Publisher pub2("ultraDistance", &distance_msg);

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

  pinMode(trig, OUTPUT);
  pinMode(echo, INPUT);

  // 트리거 및 에코 핀을 출력/입력 모드로 설정
  //for (int i = 0; i < numSensors; i++) {
      //pinMode(trigPins[i], OUTPUT);
      //pinMode(echoPins[i], INPUT);
  //}
  
  left_tilt(250);
  delay(4000);
  stop_tilt();
  val_max= potentiometer_Read(analogPin);
  

  // Motor_Forward(255, motor_right_1, motor_right_2, motor_right_PWM);  // Forward, PWM setting 0-255
  // Motor_Forward(255, motor_left_1, motor_left_2, motor_left_PWM);  
  
  right_tilt(250);
  delay(4000);
  stop_tilt();
  Motor_Brake(motor_right_1, motor_right_2);  // 인수 추가
  Motor_Brake(motor_left_1, motor_left_2);  // 인수 추가
  val_min= potentiometer_Read(analogPin);
  
  val_mid = (val_max + val_min)/2.0;
  
 
  nh.initNode();
  nh.subscribe(sub_obs);
  nh.subscribe(sub_cont);
  // nh.advertise(pub1);
  //nh.advertise(pub2);
  
  //current_state_msg.data_length = 2;
  //current_state_msg.data = (float*)malloc(sizeof(float) * current_state_msg.data_length);
  // 메시지 데이터 배열 크기 설정
  //distance_msg.data_length = numSensors;
  //distance_msg.data = (float*)malloc(sizeof(float) * numSensors);
}

void loop() {

  // 초음파 센서 값 읽기
  //read_multiple_ultrasonics();
  // 거리 값을 메시지에 복사
  //for (int i = 0; i < numSensors; i++) {
    //  distance_msg.data[i] = distances[i];
  //}
  // 현재 조향각을 읽어서 퍼블리시
  // current_state_msg.data[0] = (potentiometer_Read(analogPin) - val_mid)/float(angle_MAX);  
  //current_state_msg.data[0] = potentiometer_Read(analogPin)- val_mid;
  //current_state_msg.data[1] = current_power;
  // 메시지 퍼블리시
  //pub2.publish(&distance_msg);
  //pub1.publish(&current_state_msg);

  // change_angle(goal_steer);

  // delay를 사용하여 PID 제어 주기를 조절할 수 있습니다.
  // 필요에 따라 PID 제어의 빈도를 더 높이거나 낮출 수 있습니다.
  // distance_ac = ultrasonic_distance(trig, echo);
  if (control_signal == true){
    if (isFirstParking == false){

      Motor_Forward(0, motor_right_1, motor_right_2, motor_right_PWM);  // Forward, PWM setting 0-255
      Motor_Forward(0, motor_left_1, motor_left_2, motor_left_PWM);  
    
    }
    if (obstacle_detected_num> 8 & isFirstParking){
      isFirstParking = false;
      parking();  
    }

    for(int i=0; i<10; i++){
      change_angle(0);
      delay(10); // 예시로 0.01초마다 PID 제어를 수행합니다.
    }

    Motor_Forward(40, motor_right_1, motor_right_2, motor_right_PWM);  // Forward, PWM setting 0-255
    Motor_Forward(50, motor_left_1, motor_left_2, motor_left_PWM);  
    
    // forward(goal_pulse_left, goal_pulse_right);
    //current_power = goal_power;
  } else{

    Motor_Forward(0, motor_right_1, motor_right_2, motor_right_PWM);  // Forward, PWM setting 0-255
    Motor_Forward(0, motor_left_1, motor_left_2, motor_left_PWM);  
    
  }
  nh.spinOnce();
}
