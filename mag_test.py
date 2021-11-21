import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

relay_pin =18
GPIO.setup(relay_pin, GPIO.OUT)

init_button_pin = 24
exit_button_pin = 25
GPIO.setup(init_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(exit_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

while(True):
    GPIO.output(relay_pin, 0)
