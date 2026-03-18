/*
  LiquidCrystal Library - Hello World

 Demonstrates the use a 16x2 LCD display.  The LiquidCrystal
 library works with all LCD displays that are compatible with the
 Hitachi HD44780 driver. There are many of them out there, and you
 can usually tell them by the 16-pin interface.

 This sketch prints "Hello World!" to the LCD
 and shows the time.

  The circuit:
 * LCD RS pin to digital pin 12
 * LCD Enable pin to digital pin 11
 * LCD D4 pin to digital pin 5
 * LCD D5 pin to digital pin 4
 * LCD D6 pin to digital pin 3
 * LCD D7 pin to digital pin 2
 * LCD R/W pin to ground
 * LCD VSS pin to ground
 * LCD VCC pin to 5V
 * 10K resistor:
 * ends to +5V and ground
 * wiper to LCD VO pin (pin 3)

 Library originally added 18 Apr 2008
 by David A. Mellis
 library modified 5 Jul 2009
 by Limor Fried (http://www.ladyada.net)
 example added 9 Jul 2009
 by Tom Igoe
 modified 22 Nov 2010
 by Tom Igoe
 modified 7 Nov 2016
 by Arturo Guadalupi

 This example code is in the public domain.

 https://docs.arduino.cc/learn/electronics/lcd-displays

*/

// include the library code:
#include <LiquidCrystal.h>

// initialize the library by associating any needed LCD interface pin
// with the arduino pin number it is connected to
const int rs = 12, en = 11, d4 = 5, d5 = 4, d6 = 3, d7 = 2;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

static String pendingLine;
static uint8_t nextRow = 0;
static unsigned long firstLineAtMs = 0;
static constexpr unsigned long SECOND_LINE_TIMEOUT_MS = 750;

static void lcdPrintRow(uint8_t row, const String &text) {
  lcd.setCursor(0, row);
  lcd.print(text);

  // Clear any leftover characters from a previous longer line.
  const int remaining = 16 - text.length();
  for (int i = 0; i < remaining; i++) lcd.print(' ');
}

void setup() {
  Serial.begin(115200);

  // Set up the LCD's number of columns and rows:
  lcd.begin(16, 2);
  lcd.clear();
  lcdPrintRow(0, "Waiting UART...");
  lcdPrintRow(1, "");
}

void loop() {
  // If we received only one line, auto-finish the message after a short delay.
  if (nextRow == 1 && (millis() - firstLineAtMs) > SECOND_LINE_TIMEOUT_MS) {
    nextRow = 0;
  }

  while (Serial.available() > 0) {
    char c = static_cast<char>(Serial.read());
    if (c == '\r') continue;

    if (c == '\n') {
      String line = pendingLine;
      pendingLine = "";

      if (line.length() > 16) line.remove(16);

      if (nextRow == 0) {
        lcdPrintRow(0, line);
        lcdPrintRow(1, "");
        nextRow = 1;
        firstLineAtMs = millis();
      } else {
        lcdPrintRow(1, line);
        nextRow = 0;
      }
    } else {
      if (pendingLine.length() < 16) {
        pendingLine += c;
      }
    }
  }
}