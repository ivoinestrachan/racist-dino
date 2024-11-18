import cv2
import pyttsx3

engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()


camera = cv2.VideoCapture(0)
speak("Hi, I am the racist dinosaur")

while True:
    ret, frame = camera.read()
    if not ret:
        print("failed to grab frame.")
        break

    cv2.imshow("camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        speak("goodbye cruel world")
        break

camera.release()
cv2.destroyAllWindows()
