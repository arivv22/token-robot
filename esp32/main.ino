#include <WiFi.h>
#include <WebServer.h>
#include <Servo.h>

const char* ssid = "YOUR_WIFI";
const char* password = "YOUR_PASSWORD";

WebServer server(80);

Servo servo;

void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }

  servo.attach(13); // pin servo

  server.on("/input-token", HTTP_POST, handleToken);
  server.begin();
}

void loop() {
  server.handleClient();
}

void pressButton(int angle) {
  servo.write(angle);
  delay(500);
  servo.write(0);
  delay(500);
}

void handleToken() {
  String body = server.arg("plain");

  Serial.println("Received: " + body);

  // dummy parsing (simple)
  for (int i = 0; i < body.length(); i++) {
    char c = body[i];
    if (isdigit(c)) {
      pressButton(90); // sementara semua tekan sama
    }
  }

  server.send(200, "application/json", "{\"status\":\"ok\"}");
}