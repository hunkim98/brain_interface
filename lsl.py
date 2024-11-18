"""Example program to show how to read a multi-channel time series from LSL."""

from pylsl import StreamInlet, resolve_stream
from model.emotiv_digit import EEGNet
import torch
import joblib
import pygame
import numpy as np
import time


model = EEGNet()
rf_model = None
PATH = "model/model_32_14.pth"
all_muse_scalers = None
input = []

channel_values = {
    "AF3": [],
    "F7": [],
    "F3": [],
    "FC5": [],
    "T7": [],
    "P7": [],
    "O1": [],
    "O2": [],
    "P8": [],
    "T8": [],
    "FC6": [],
    "F4": [],
    "F8": [],
    "AF4": [],
}


def predict():
    model = EEGNet()
    model.load_state_dict(torch.load(PATH, weights_only=True))
    # model.predict()


def initialize():
    global model, rf_model, all_muse_scalers
    print("initializing ")
    with open("rf_model_emotiv_1890.pkl", "rb") as f:
        rf_model = joblib.load(f)
        print("RF Model loaded")
    with open("all_muse_channel_scalers.pkl", "rb") as f:
        all_muse_scalers = joblib.load(f)
        print("All muse scalers loaded")
    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()


def play_number_digit(number):
    if number == 0:
        play_mp3("./mp3/thinking0.mp3")
    elif number == 1:
        play_mp3("./mp3/thinking1.mp3")
    elif number == 2:
        play_mp3("./mp3/thinking2.mp3")
    elif number == 3:
        play_mp3("./mp3/thinking3.mp3")
    elif number == 4:
        play_mp3("./mp3/thinking4.mp3")
    elif number == 5:
        play_mp3("./mp3/thinking5.mp3")
    elif number == 6:
        play_mp3("./mp3/thinking6.mp3")
    elif number == 7:
        play_mp3("./mp3/thinking7.mp3")
    elif number == 8:
        play_mp3("./mp3/thinking8.mp3")
    elif number == 9:
        play_mp3("./mp3/thinking9.mp3")


def interpret(sample):
    global input, channel_values
    af3 = sample[3]
    f7 = sample[4]
    f3 = sample[5]
    fc5 = sample[6]
    t7 = sample[7]
    p7 = sample[8]
    o1 = sample[9]
    o2 = sample[10]
    p8 = sample[11]
    t8 = sample[12]
    fc6 = sample[13]
    f4 = sample[14]
    f8 = sample[15]
    af4 = sample[16]
    try:
        # Order of the channels
        # 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
        channels = [
            "AF3",
            "F7",
            "F3",
            "FC5",
            "T7",
            "P7",
            "O1",
            "O2",
            "P8",
            "T8",
            "FC6",
            "F4",
            "F8",
            "AF4",
        ]
        input_row = [af3, f7, f3, fc5, t7, p7, o1, o2, p8, t8, fc6, f4, f8, af4]

        channel_values["AF3"].append(af3)
        channel_values["F7"].append(f7)
        channel_values["F3"].append(f3)
        channel_values["FC5"].append(fc5)
        channel_values["T7"].append(t7)
        channel_values["P7"].append(p7)
        channel_values["O1"].append(o1)
        channel_values["O2"].append(o2)
        channel_values["P8"].append(p8)
        channel_values["T8"].append(t8)
        channel_values["FC6"].append(fc6)
        channel_values["F4"].append(f4)
        channel_values["F8"].append(f8)
        channel_values["AF4"].append(af4)

        input.append(input_row)

        # if len(input) == 32:
        #     input_tensor = torch.tensor(input)
        #     input_tensor = input_tensor.unsqueeze(0)
        #     output = model(input_tensor)
        #     probs = torch.nn.functional.softmax(output, dim=1)
        #     pred = torch.argmax(probs, dim=1)
        #     print(pred)
        #     input.clear()
        if len(input) == 135:  # rf model
            # need to change
            scaled = {}
            con_input = []
            for key, value in channel_values.items():
                con_input.extend(value)
            con_input = np.array(con_input).reshape(1, -1)
            print(con_input.shape)
            pred = rf_model.predict(con_input)
            proba = rf_model.predict_proba(con_input)
            max_proba = np.max(proba)
            if max_proba > 0.3:
                if np.argmax(proba) == 10:
                    return
                play_number_digit(np.argmax(proba))
                print("Strongly predicted ", np.argmax(proba))
            else:
                print("Not strongly predicted")
            channel_values = {
                "AF3": [],
                "F7": [],
                "F3": [],
                "FC5": [],
                "T7": [],
                "P7": [],
                "O1": [],
                "O2": [],
                "P8": [],
                "T8": [],
                "FC6": [],
                "F4": [],
                "F8": [],
                "AF4": [],
            }
            input.clear()

    except Exception as e:
        print(e, "hi")
        # print("Error")
        input.clear()
        channel_values = {
            "AF3": [],
            "F7": [],
            "F3": [],
            "FC5": [],
            "T7": [],
            "P7": [],
            "O1": [],
            "O2": [],
            "P8": [],
            "T8": [],
            "FC6": [],
            "F4": [],
            "F8": [],
            "AF4": [],
        }


def play_mp3(file_path):
    # Load the MP3 file
    pygame.mixer.music.load(file_path)
    # Start playing the MP3 file
    pygame.mixer.music.play()
    print("Playing... Press Ctrl+C to stop.")

    # Wait until the music stops playing
    while pygame.mixer.music.get_busy():
        time.sleep(1)


def main():
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    pygame.mixer.init()
    try:
        initialize()
    except:
        print("Model not loaded")

    streams = resolve_stream("type", "EEG")

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    info = inlet.info()
    print(f"\nThe manufacturer is: {info.desc().child_value('manufacturer')}")
    print("The channel labels are listed below:")
    ch = info.desc().child("channels").child("channel")
    labels = []
    for _ in range(info.channel_count()):
        labels.append(ch.child_value("label"))
        ch = ch.next_sibling()
    print(f"  {', '.join(labels)}")

    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        sample, timestamp = inlet.pull_sample()
        interpret(sample)
        # print(timestamp, sample)


# the order are
# [
#     "COUNTER",
#     "INTERPOLATED",
#     "AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4",
#     "RAW_CQ",
#     "MARKER_HARDWARE",
#     "MARKERS"
# ]

if __name__ == "__main__":
    main()
