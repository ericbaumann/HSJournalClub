import cv2  # Import OpenCV for image processing
import sys  # Import for time
import os  # Import for reading files
import threading  # Import for separate thread for image classification
import numpy as np  # Import for converting vectors
from gtts import gTTS  # Import Google Text to Speech

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf  # Import tensorflow for Inception Net's backend

# Language in which you want to convert
language = 'en'

# Get a live stream from the webcam
live_stream = cv2.VideoCapture(1)

# Word for which letters are currently being signed
current_word = ""

# Load training labels
CATEGORIES = ["A", "B", "C", "D", "del", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "nothing", "O", "P", "Q", "R", "S", "space", "T", "U",
              "V", "W", "X", "Y", "Z"]
IMG_SIZE = 28

# Load trained model's graph
# with tf.io.gfile.GFile("./asl_model/saved_model.pb", 'rb') as f:
#     # Define a tensorflow graph
#     graph_def = tf.compat.v1.GraphDef()
#
#     # Read and import line by line from the trained model's graph
#     graph_def.ParseFromString(f.read())
#     _ = tf.import_graph_def(graph_def, name='')


def prepare(filepath):
    new_array = cv2.resize(filepath, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def predict(model, image_data):
    # Focus on Region of Interest (Image within the bounding box)
    resized_image = image_data[70:350, 70:350]

    # Resize to 200 x 200
    resized_image = cv2.resize(resized_image, (200, 200))

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Just for Debugging
    cv2.imshow("grayscale + resized Image", gray)

    print(prepare(gray))

    predictions = model.predict(prepare(gray))

    print(predictions)

    final = (CATEGORIES[int(np.argmax(predictions[0]))])

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


with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    model = tf.keras.models.load_model("./asl_model")

    # tf.compat.v1.saved_model.loader.load(
    #     sess, tags=[tf.compat.v1.saved_model.tag_constants.SERVING], export_dir="./asl_model/")

    # Feed the image_data as input to the graph and get first prediction
    # softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

    # Global variable to keep track of time
    time_counter = 0

    # Flag to check if 'c' is pressed
    captureFlag = False

    # Toggle real time processing
    realTime = True

    # Infinite loop
    while True:

        # Display live feed until ESC key is pressed
        # Press ESC to exit
        keypress = cv2.waitKey(1)

        # Flip the image laterally
        # img = cv2.flip(img, 1)

        # TESTING:
        # threading.Timer(5.0, printit).start()

        # Read a single frame from the live feed
        img = live_stream.read()[1]

        # Set a region of interest
        cv2.rectangle(img, (70, 70), (350, 350), (0, 255, 0), 2)

        # Show the live stream
        cv2.imshow("Live Stream", img)

        # To get time intervals
        if time_counter % 45 == 0 and realTime:

            letter = predict(model, img)
            # cv2.imshow("Resized Image", img)
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
            letter = predict(img)
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

# Stop using camera
live_stream.release()

# Destroy windows created by OpenCV
cv2.destroyAllWindows()