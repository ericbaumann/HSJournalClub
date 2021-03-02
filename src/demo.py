import cv2
import os
import tensorflow as tf
import numpy as np
from gtts import gTTS  # Import Google Text to Speech

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Language for speech
language = 'en'

# labels
CATEGORIES = ["A", "B", "C", "D", "del", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "nothing", "O", "P", "Q",
              "R", "S", "space", "T", "U",
              "V", "W", "X", "Y", "Z"]
IMG_SIZE = 28
FONT = cv2.FONT_HERSHEY_SIMPLEX


def predict(image_data):
    predictions = model.predict(image_data)
    final = CATEGORIES[int(np.argmax(predictions[0]))]
    return final


def speak_letter(letter):
    # Create the text to be spoken
    prediction_text = letter

    # Create a speech object from text to be spoken
    speech_object = gTTS(text=prediction_text, lang=language, slow=False)

    # Save the speech object in a file called 'prediction.mp3'
    speech_object.save("prediction.mp3")

    # Playing the speech using mpg321
    os.system("afplay prediction.mp3")


def process_frame(frame):
    # crop to region of interest
    cropped_frame = frame[70:350, 70:350]

    # resize image to match model's expected sizing
    resized_cropped_frame = cv2.resize(cropped_frame, (IMG_SIZE, IMG_SIZE))

    # grayscale the image
    gray = cv2.cvtColor(resized_cropped_frame, cv2.COLOR_BGR2GRAY)

    # scale the image
    scaled_gray = gray / 255.0

    cv2.imshow("scaled gray", scaled_gray)

    return scaled_gray.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# Get a live stream from the webcam
cap = cv2.VideoCapture(0)

# Load the model
model = tf.keras.models.load_model("./models/Mar-01-2021-asl_model")

# Global variable to keep track of time
time_counter = 0

# Flag to check if 'c' is pressed
captureFlag = False

# Toggle real time processing
realTime = True

# var to track current word
current_word = ""

# Infinite loop
while True:

    # Display live feed until ESC key is pressed
    # Press ESC to exit
    keypress = cv2.waitKey(1)

    # Read a single frame from the live feed
    ret, frame = cap.read()

    # Set a region of interest
    cv2.rectangle(frame, (70, 70), (350, 350), (0, 255, 0), 2)

    # Show the live stream
    cv2.imshow("Live Stream", frame)

    prepared_frame = process_frame(frame)

    # To get time intervals
    if time_counter % 30 == 0 and realTime:

        letter = predict(prepared_frame)
        print("Letter: ", letter.upper())
        print("Current word: ", current_word)

        if letter.upper() != 'NOTHING' and letter.upper() != 'SPACE' and letter.upper() != 'DEL':
            current_word += letter.upper()
            speak_letter(letter)

        # Say the letter out loud
        elif letter.upper() == 'SPACE':
            if len(current_word) > 0:
                speak_letter(current_word)
            current_word = ""

        elif letter.upper() == 'DEL':
            if len(current_word) > 0:
                current_word = current_word[:-1]

        elif letter.upper() == 'NOTHING':
            pass

        else:
            print("UNEXPECTED INPUT: ", letter.upper())

    # 'C' is pressed
    if keypress == ord('c'):
        captureFlag = True
        realTime = False

    # 'R' is pressed
    if keypress == ord('r'):
        realTime = True

    if captureFlag:
        captureFlag = False

        # Show the image considered for classification

        # Get the letter and the score
        letter = predict(prepared_frame)
        print("Letter: ", letter.upper())
        print("Current word: ", current_word)

        if letter.upper() != 'NOTHING' and letter.upper() != 'SPACE' and letter.upper() != 'DEL':
            current_word += letter.upper()
            speak_letter(letter)

        # Say the letter out loud
        elif letter.upper() == 'SPACE':
            if len(current_word) > 0:
                speak_letter(current_word)
            current_word = ""

        elif letter.upper() == 'DEL':
            if len(current_word) > 0:
                current_word = current_word[:-1]

        elif letter.upper() == 'NOTHING':
            pass

        else:
            print("UNEXPECTED INPUT: ", letter.upper())

    # If ESC is pressed
    if keypress == 27:
        exit(0)

    # Update time
    time_counter = time_counter + 1

    # show status of word
    cv2.putText(frame, current_word, (200, 100), FONT, 1, (0, 0, 0), 2, cv2.LINE_AA)

# Stop using camera
cap.release()

# Destroy windows created by OpenCV
cv2.destroyAllWindows()