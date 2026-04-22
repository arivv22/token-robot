# token-robot
just robot automaticaly input your electricity token lol


This project have purpose easily your life to input token electricity repeatly 
sometime you wish someone do that for you
especially for your short from input your token


# Structure 
Actually this project have 3 main
1. Backend-go using for download your image give to OCR Service
2. OCR Service decode your image to input token number give to ESP32
3. ESP32 use for push number in your electricity

# Installation
Python try to install using conda
```pip install -r requirements.txt```
Golang 
```go mod init token-robot ```
```go get github.com/gin-gonic/gin```
ESP32 (Arduino + Servo) 
Library 
- install Servo.h