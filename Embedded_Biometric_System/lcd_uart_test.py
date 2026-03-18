#!/usr/bin/env python3

import time

try:
    import serial
except ImportError as e:
    raise ImportError("pyserial is required. Install it with: pip install pyserial") from e


class LCDUARTDisplay:
    LCD_WIDTH = 16

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        timeout: float = 1.0,
        delay_after_open: float = 2.0,
        clear_delay: float = 0.05,
    ):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.delay_after_open = delay_after_open
        self.clear_delay = clear_delay

        self.ser = serial.Serial(self.port, baudrate=self.baud, timeout=self.timeout)
        time.sleep(self.delay_after_open)

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @classmethod
    def _trim_line(cls, text: str) -> str:
        return text[:cls.LCD_WIDTH]

    @classmethod
    def _split_message(cls, message: str) -> tuple[str, str | None]:
        lines = message.splitlines()

        if len(lines) == 0:
            return "", None
        if len(lines) == 1:
            return cls._trim_line(lines[0]), None

        return cls._trim_line(lines[0]), cls._trim_line(lines[1])

    def _write_lines(self, line1: str, line2: str | None = None) -> None:
        payload = self._trim_line(line1) + "\n"
        if line2 is not None:
            payload += self._trim_line(line2) + "\n"

        self.ser.write(payload.encode("utf-8"))
        self.ser.flush()

    def clear(self) -> None:
        self._write_lines("", "")
        time.sleep(self.clear_delay)

    def send_message(self, message: str) -> None:
        line1, line2 = self._split_message(message)
        self.clear()
        self._write_lines(line1, line2)