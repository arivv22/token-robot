#include <WiFi.h>
#include <WebServer.h>
#include <Servo.h>
#include <ArduinoJson.h>
#include "servo_map.h"

const char* ssid = "YOUR_WIFI";
const char* password = "YOUR_PASSWORD";

WebServer server(80);
Servo servo;

void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  servo.attach(SERVO_PIN);
  servo.write(SERVO_REST_POSITION);
  delay(1000);

  server.on("/input-token", HTTP_POST, handleToken);
  server.on("/", HTTP_GET, handleRoot);
  
  server.begin();
  Serial.println("Server started");
}

void loop() {
  server.handleClient();
}

void pressButton(int digit) {
  if (digit < 0 || digit > 9) {
    Serial.println("Invalid digit: " + String(digit));
    return;
  }
  
  int angle = DIGIT_ANGLES[digit];
  Serial.println("Pressing digit " + String(digit) + " at angle " + String(angle));
  
  servo.write(angle);
  delay(PRESS_DURATION);
  servo.write(SERVO_REST_POSITION);
  delay(RELEASE_DURATION);
}

void handleRoot() {
  server.send(200, "text/html", "<h1>Token Robot ESP32</h1><p>POST to /input-token with JSON: {\"token\": \"12345678901234567890\"}</p>");
}

void handleToken() {
  String body = server.arg("plain");
  Serial.println("Received: " + body);

  // Parse JSON
  DynamicJsonDocument doc(200);
  DeserializationError error = deserializeJson(doc, body);
  
  if (error) {
    Serial.println("JSON parse failed");
    server.send(400, "application/json", "{\"status\":\"error\",\"message\":\"Invalid JSON\"}");
    return;
  }

  String token = doc["token"];
  
  if (token.length() != 20) {
    Serial.println("Invalid token length: " + String(token.length()));
    server.send(400, "application/json", "{\"status\":\"error\",\"message\":\"Token must be 20 digits\"}");
    return;
  }

  Serial.println("Processing token: " + token);
  
  // Process each digit
  for (int i = 0; i < token.length(); i++) {
    char c = token[i];
    if (isdigit(c)) {
      int digit = c - '0';
      pressButton(digit);
    } else {
      Serial.println("Non-digit character: " + String(c));
    }
  }

  Serial.println("Token processing completed");
  server.send(200, "application/json", "{\"status\":\"success\",\"token\":\"" + token + "\"}");
}