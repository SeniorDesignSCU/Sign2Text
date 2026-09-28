// Signs to Speech — glove firmware v0 (Week 1–2 deliverable)
//
// Streams 5 flex sensors + 6-axis IMU at 50 Hz over USB serial as CSV, so the
// Python logger (glove/tools/log_serial.py) can record training data.
// BLE streaming and on-device inference (Edge Impulse) come in Weeks 5–6.
//
// Wiring per finger: 3.3V -> flex sensor -> A<n> node -> 10k -> GND
//   A0 thumb, A1 index, A2 middle, A3 ring, A4 pinky
//
// CSV row: t_ms, f0..f4 (raw ADC 0-4095), ax, ay, az (g), gx, gy, gz (deg/s)

#include <Arduino.h>
#include <Arduino_BMI270_BMM150.h>

static const int FLEX_PINS[5] = {A0, A1, A2, A3, A4};
static const uint32_t PERIOD_US = 20000;  // 50 Hz

uint32_t next_us = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}
  analogReadResolution(12);
  if (!IMU.begin()) {
    Serial.println("ERR IMU init failed");
    while (true) { delay(1000); }
  }
  Serial.println("t_ms,f0,f1,f2,f3,f4,ax,ay,az,gx,gy,gz");
  next_us = micros();
}

void loop() {
  uint32_t now = micros();
  if ((int32_t)(now - next_us) < 0) return;
  next_us += PERIOD_US;

  float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0;
  if (IMU.accelerationAvailable()) IMU.readAcceleration(ax, ay, az);
  if (IMU.gyroscopeAvailable())    IMU.readGyroscope(gx, gy, gz);

  Serial.print(millis());
  for (int i = 0; i < 5; i++) { Serial.print(','); Serial.print(analogRead(FLEX_PINS[i])); }
  Serial.print(','); Serial.print(ax, 3); Serial.print(','); Serial.print(ay, 3); Serial.print(','); Serial.print(az, 3);
  Serial.print(','); Serial.print(gx, 2); Serial.print(','); Serial.print(gy, 2); Serial.print(','); Serial.print(gz, 2);
  Serial.println();
}
