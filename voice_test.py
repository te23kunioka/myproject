import pyttsx3
import time

while True:

    engine = pyttsx3.init()

    engine.say("はい")

    engine.runAndWait()

    print("しゃべった")

    time.sleep(3)