/*
    Function_Library.h - Library for Future Car Arduino function
    Created by SKKU Automation LAB, November 12, 2022
    + 2024.07.05 Park Sang Won (SNU ME 24) - Innodriver 수정
*/
#ifndef Car_Lib_h
#define Car_Lib_h

float ultrasonic_distance(int trigPin, int echoPin);
float potentiometer_Read(int pin);

void motor_forward(int IN1, int IN2, int PWM, int speed);
void motor_backward(int IN1, int IN2, int PWM, int speed);
void motor_hold(int IN1, int IN2);


#endif