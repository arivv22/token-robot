# Token Robot ESP32

## Installation Requirements

### Arduino IDE Setup
1. Install Arduino IDE 2.0+
2. Add ESP32 Board Manager:
   - File → Preferences → Additional Board Manager URLs:
   - `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Tools → Board → Boards Manager → Search "ESP32" → Install

### Required Libraries
Install these libraries via Library Manager:
1. **ArduinoJson** by Benoit Blanchon
2. **Servo** (built-in)
3. **WiFi** (built-in)
4. **WebServer** (built-in)

### Installation Steps
1. Open Arduino IDE
2. Select Board: Tools → Board → ESP32 → ESP32 Dev Module
3. Install ArduinoJson:
   - Tools → Manage Libraries
   - Search "ArduinoJson"
   - Install "ArduinoJson" by Benoit Blanchon
4. Upload this code to ESP32

### Configuration
1. Update WiFi credentials in `main.ino`:
   ```cpp
   const char* ssid = "YOUR_WIFI";
   const char* password = "YOUR_PASSWORD";
   ```
2. Adjust servo angles in `servo_map.h` based on your button layout:
   ```cpp
   const int DIGIT_ANGLES[10] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
   ```

### Usage
1. Power on ESP32
2. Connect to WiFi
3. Check Serial Monitor for IP address
4. Send POST request to `http://[ESP32_IP]/input-token` with JSON:
   ```json
   {"token": "12345678901234567890"}
   ```

### API Endpoints
- `GET /` - API documentation
- `POST /input-token` - Process token input
