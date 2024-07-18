#include "Arduino.h"
#include "Car_Library.h"

float ultrasonic_distance(int trigPin, int echoPin)
{
    long distance, duration;

    digitalWrite(trigPin, LOW);
    digitalWrite(echoPin, LOW);
    delayMicroseconds(2);

    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    duration = pulseIn(echoPin, HIGH);
    distance = ((float)(340 * duration) / 1000) / 2;

    return distance;
}

float potentiometer_Read(int pin)
{
    int value;
    // 정확한 각도 계산 : 270도 = 1024
    value = float(analogRead(pin)) / 1024.0 *270.0;

    return value;
}

void motor_forward(int IN1, int IN2, int PWM, int speed)
{
    // analogWrite(IN1, speed);
    // analogWrite(IN2, LOW);
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    analogWrite(PWM, speed);
}

void motor_backward(int IN1, int IN2, int PWM, int speed)
{
    // analogWrite(IN1, LOW);
    // analogWrite(IN2, speed);
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    analogWrite(PWM, speed);
}

void motor_hold(int IN1, int IN2)
{
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
}