# Token Robot

🤖 An automated system for inputting electricity tokens using computer vision and robotics.

## Overview

Token Robot is a comprehensive solution that automates the process of entering electricity tokens by:
- Capturing token images
- Extracting token numbers using OCR
- Automatically inputting the token via a robotic arm

This project eliminates the repetitive task of manually entering electricity tokens, making your life easier and more convenient.

## Architecture

The system consists of three main components:

### 1. Backend (Go)
- **Purpose**: Downloads and processes images, coordinates between services
- **Technology**: Gin web framework
- **Responsibilities**: 
  - Image handling and storage
  - Communication with OCR service
  - API endpoints for system control

### 2. OCR Service (Python)
- **Purpose**: Extracts token numbers from images using computer vision
- **Technology**: Flask, OpenCV, Tesseract OCR
- **Responsibilities**:
  - Image preprocessing
  - Text extraction and recognition
  - Token number validation
  - Communication with ESP32

### 3. Hardware Controller (ESP32)
- **Purpose**: Physical token input mechanism
- **Technology**: Arduino IDE, Servo motors
- **Responsibilities**:
  - Receives token numbers from OCR service
  - Controls servo motors for button pressing
  - Physical interaction with electricity meter

## Prerequisites

- Python 3.8+
- Go 1.19+
- Arduino IDE
- ESP32 development board
- Servo motor
- Camera module (for image capture)

## Installation

### Backend (Go)
```bash
cd backend-go
go mod init token-robot
go get github.com/gin-gonic/gin
go run main.go
```

### OCR Service (Python)
```bash
cd ocr-service
pip install -r requirements.txt
python app.py
```

**Python Dependencies:**
- numpy==1.26.4
- opencv-python==4.6.0.66
- pytesseract==0.3.10
- flask
- pillow

### ESP32 (Arduino)
1. Open `esp32/main.ino` in Arduino IDE
2. Install required libraries:
   - `Servo.h` (built-in)
   - ESP32 board support package
3. Upload the sketch to your ESP32

## Hardware Setup

### Required Components
- ESP32 development board
- SG90 or similar servo motor
- Jumper wires
- Power supply for ESP32
- Camera module (optional, can use existing camera)

### Wiring
- Connect servo signal pin to ESP32 GPIO pin (as defined in code)
- Connect servo power (5V) and ground
- Ensure stable power supply for smooth servo operation

## Usage

1. **Start all services** in order:
   - Backend Go service
   - OCR Python service
   - ESP32 hardware controller

2. **Capture token image** using camera or upload existing image

3. **Process token** through the system:
   - Backend receives image
   - OCR service extracts token number
   - ESP32 automatically inputs the token

## Configuration

### OCR Settings
- Adjust image preprocessing parameters in `ocr-service/app.py`
- Configure Tesseract language settings if needed

### Hardware Settings
- Modify servo pin assignments in `esp32/main.ino`
- Adjust servo timing and angles for your specific electricity meter

## Troubleshooting

### Common Issues
- **OCR Accuracy**: Ensure good lighting and clear image focus
- **Servo Movement**: Check power supply and wiring connections
- **Communication**: Verify all services are running on correct ports

### Debug Mode
Enable debug logging in individual components for detailed troubleshooting.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
- Check the troubleshooting section
- Review component-specific documentation
- Open an issue on GitHub