from pythonosc import dispatcher
from pythonosc import osc_server
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import joblib
import pygame
import time


datapoints = 0
REQUIRED_DATAPOINTS = 236
tp9_min_max = None
fp1_min_max = None
fp2_min_max = None
tp10_min_max = None
rf_model = None
knn_model = None
tp_9_values = []
tp_10_values = []


def play_mp3(file_path):
    # Load the MP3 file
    pygame.mixer.music.load(file_path)
    # Start playing the MP3 file
    pygame.mixer.music.play()
    print("Playing... Press Ctrl+C to stop.")

    # Wait until the music stops playing
    while pygame.mixer.music.get_busy():
        time.sleep(1)


def initialize():
    global tp9_min_max, fp1_min_max, fp2_min_max, tp10_min_max, rf_model, knn_model
    # tp9 import pkl
    with open("muse_tp9_scaler.pkl", "rb") as f:
        print("scaler tp9 loaded")
        tp9_min_max = joblib.load(f)
    # af7 import pkl
    with open("muse_tp10_scaler.pkl", "rb") as f:
        print("scaler tp10 loaded")
        tp10_min_max = joblib.load(f)
    # import rf_model
    with open("rf_model_muse_binary.joblib", "rb") as f:
        rf_model = joblib.load(f)
        print("Model loaded")
    with open("knn_model_binary.pkl", "rb") as f:
        knn_model = joblib.load(f)
        print("Model loaded")


# Define a handler function to process incoming OSC messages
def message_handler(address, *args):
    global tp9_min_max, fp2_min_max, fp1_min_max, tp10_min_max, rf_model, datapoints, tp_9_values, tp_10_values
    # print(f"Received message at {address} with arguments: {args}")
    tp9 = args[0]  # eeg0
    af7 = args[1]  # eeg1
    af8 = args[2]  # eeg2
    tp10 = args[3]  # eeg3
    # print(tp9, tp10)
    if (
        np.isnan(tp9)
        or np.isnan(tp10)
        or np.isinf(tp9)
        or np.isinf(tp10)
        or tp9 == 0
        or tp10 == 0
    ):

        print("NaN value detected. Skipping")
        # reset the datapoints
        tp_9_values = []
        tp_10_values = []
        datapoints = 0
        return
    # print(f"TP9: {tp9}, TP10: {tp10}")
    # we can only use tp9 and tp10
    # print(tp9_min_max, tp10_min_max)
    tp_9_values.append(tp9)
    tp_10_values.append(tp10)

    datapoints += 1
    # print(datapoints)
    try:
        if datapoints == REQUIRED_DATAPOINTS:
            scaled_tp9 = tp9_min_max.fit_transform(np.array(tp_9_values).reshape(-1, 1))
            scaled_tp10 = tp10_min_max.fit_transform(
                np.array(tp_10_values).reshape(-1, 1)
            )
            tp9_np = np.array(scaled_tp9)
            tp10_np = np.array(scaled_tp10)
            X = np.concatenate((tp9_np, tp10_np), axis=0).reshape(1, -1)
            print(X.shape)
            # pred = knn_model.predict(X)
            pred = rf_model.predict(X)
            # print(pred)
            proba = rf_model.predict_proba(X)
            # proba = knn_model.predict_proba(X)

            if pred == 0:
                print(f"No Digit")
            else:
                print("Thinking of digit: ", proba[0])
                play_mp3(f"thinking_of_digit.mp3")

            datapoints = 0
            tp_9_values = []
            tp_10_values = []
    except Exception as e:
        datapoints = 0
        tp_9_values = []
        tp_10_values = []
        print(e)

    # the order should be tp9, tp10, fp1, fp2

    # print(f"TP9: {tp9}, AF7: {af7}, AF8: {af8}, TP10: {tp10}")


def hsi_handler(address: str, *args):
    global hsi, hsi_string

    hsi = args
    if (args[0] + args[1] + args[2] + args[3]) == 4:
        hsi_string_new = "Muse Fit Good + Will start predicting"
    else:
        hsi_string_new = "Muse Fit Bad on: "
        if args[0] != 1:
            hsi_string_new += "Left Ear. "
        if args[1] != 1:
            hsi_string_new += "Left Forehead. "
        if args[2] != 1:
            hsi_string_new += "Right Forehead. "
        if args[3] != 1:
            hsi_string_new += "Right Ear."
        hsi_string_new = "Muse Fit Bad..."
    if hsi_string != hsi_string_new:
        hsi_string = hsi_string_new
        print(hsi_string)


ip = "0.0.0.0"  # Localhost, or use your machine's IP for networked listening
port = 5000  # Port to listen on

# Set up a dispatcher to map OSC addresses to handler functions
initialize()
pygame.mixer.init()
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/muse/eeg", message_handler)
dispatcher.map("/muse/elements/horseshoe", hsi_handler)
# You can map additional addresses if needed
# dispatcher.map("/another_address", another_handler_function)

# Set up the OSC server
server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)

print(f"Listening for OSC messages on {ip}:{port}")

# Start the server
server.serve_forever()  # Keeps the server running and listening for messages
