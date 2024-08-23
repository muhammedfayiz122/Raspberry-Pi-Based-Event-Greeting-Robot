import RPi.GPIO as GPIO
import cv2
import numpy as np
from keras.models import model_from_json
import pyttsx3
import time
import random
import pygame
import threading
global motorThread
a=1
# Set the GPIO pin number for the servo motor
servo_pin = 3

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

last_motorWorkingTime = time.time()-7
thread_status = False

last_audioplayingingTime = time.time() - 7

# Create a PWM object
pwm = GPIO.PWM(servo_pin, 50)  # 50 Hz (20 ms PWM period)


def motor():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(servo_pin, GPIO.OUT)
    pwm = GPIO.PWM(servo_pin, 50) 
    global thread_status 
    angle1 = 120
    angle2 = 270
    # Function to rotate the servo to a specific angle
    def rotate_arms(angle):
        duty_cycle = angle / 18 + 2  # Map angle to duty cycle (2 to 12)
        pwm.start(duty_cycle)
        
    def servo(angle1,angle2):
        a=angle1
        limit = (angle1+angle2)/2
        for i in range( angle1, angle2, 3 ):
       # print(i)
            if i<limit :
                a = a + 3
                rotate_arms(a)
                flag = 1

            elif flag==1 :
                time.sleep(5)
                flag=2
            # Rotate the servo to 90 degrees
            if i>limit :
                a = a - 3
                rotate_arms(a)
            time.sleep(0.07)

            print("motor is running @ ")   
            print(a)
        
      #  time.sleep(20)
    servo(angle1,angle2)
    thread_status = False
    print("motor completed")
    pwm.stop()
    GPIO.cleanup()


#thread for motor 
motorThread = threading.Thread(target=motor)
    
emotion_dict = {0: "Whats with that face", 1: "Disgusted", 2: "Ohhh got confused", 3: "Glad to see your happy face", 4: "Hai... Welcome to Agnitus", 5: "Hope agnitus can bring smile on your face", 6: "WOOW surprised"}

image_dict = {
    "Whats with that face": "images/ANGRY.png",
    "Disgusted": "images/DEAD.png",
    "Ohhh got confused": "images/CONFUSED.png",
    "Glad to see your happy face": "images/HAPPY.png",
    "Hai... Welcome to Agnitus": "images/NEUTRAL.png",
    "Hope agnitus can bring smile on your face": "images/SAD.png",
    "WOOW surprised": "images/SURPRISED.png"
}



# Load the text-to-speech engine
engine = pyttsx3.init()

# Load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into the model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

if a==1:
    
    # Start the webcam feed
    cap = cv2.VideoCapture(0)

    # Initialize variables for the last emotion time and last spoken emotion
    last_spoken_emotion = None

    default_img_path = "images/NEUTRAL.png"  # Replace with the actual path to your default image
    default_img = cv2.imread(default_img_path)
    default_img = cv2.resize(default_img, (1100, 540))
    cv2.imshow("Emotion Image", default_img)
    cv2.setWindowProperty("Emotion Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    

    while True:
        
            
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640,480))
        if not ret:
            break

        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
        # Detect faces available on the camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        if len(num_faces) > 0:
            largest_face = max(num_faces, key=lambda f: f[2] * f[3])
            (x, y, w, h) = largest_face
            
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            #emotion detection is started
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            emotion_label = emotion_dict[maxindex]
            cv2.putText(frame, emotion_label, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            if emotion_label != last_spoken_emotion:

                #motor thread starting
                motor_now = time.time()
                if not motorThread.is_alive() and motor_now - last_motorWorkingTime >= 13 :
                    last_motorWorkingTime = motor_now
                    motorThread = threading.Thread(target=motor)
                    motorThread.start()
                    print("Motor thread started")
                else:
                    print("Motor thread is already running")

                
                audio_now = time.time()
                if audio_now - last_audioplayingingTime >= 5 :
                    last_spoken_emotion = emotion_label
                    engine.say(emotion_label)
                             # Speak the detected emotion
                

                image_path = image_dict.get(emotion_label, "path_to_default_image")
                image = cv2.imread(image_path)
                
                # Adjust the emotion image size here
                width = 1100# Desired width of the emotion image frame
                height = 540  # Desired height of the emotion image frame
                resized_image = cv2.resize(image, (width, height))
                
                
                cv2.setWindowProperty("Emotion Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("Emotion Image", resized_image)
        
        else:
            
            cv2.setWindowProperty("Emotion Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Emotion Image", default_img)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        engine.runAndWait()

pygame.mixer.music.stop()
GPIO.cleanup()
cap.release()
cv2.destroyAllWindows()


