#ifndef SERVO_MAP_H
#define SERVO_MAP_H

// Digit to servo angle mapping
// Adjust these angles based on your physical button layout
const int DIGIT_ANGLES[10] = {
  10,  // 0
  20,  // 1
  30,  // 2
  40,  // 3
  50,  // 4
  60,  // 5
  70,  // 6
  80,  // 7
  90,  // 8
  100  // 9
};

// Servo configuration
const int SERVO_PIN = 13;
const int SERVO_DELAY = 500;  // milliseconds
const int SERVO_REST_POSITION = 0;

// Button press timing
const int PRESS_DURATION = 300;   // milliseconds to hold button
const int RELEASE_DURATION = 200; // milliseconds between presses

#endif
