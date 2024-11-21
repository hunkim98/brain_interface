"""Example program to show how to read a multi-channel time series from LSL."""

from pylsl import StreamInlet, resolve_stream
from model.emotiv_digit import EEGNet
import torch
import joblib
import pygame
import numpy as np
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, lfilter, stft
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(EEGCNN, self).__init__()
        # Group 1 (G1): Time-frequency convolution
        self.conv1 = nn.Conv2d(
            input_channels, 32, kernel_size=(3, 3), stride=1, padding=(1, 1)
        )  # G1 Conv
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for G1
        self.relu1 = nn.ReLU()

        # Group 2 (G2): Spatial convolution
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=(3, 3), stride=1, padding=(1, 1)
        )  # G2 Conv
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization for G2
        self.relu2 = nn.ReLU()

        # MaxPool layer
        self.pool = nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2)
        )  # Reduces both height and width

        # Fully connected layers
        self.flattened_size = self._get_flattened_size(input_channels, 129, 42)
        self.fc1 = nn.Linear(self.flattened_size, 128)  # Fully connected layer
        self.dropout = nn.Dropout(0.1)  # Dropout for regularization
        self.fc2 = nn.Linear(128, num_classes)  # Output layer for classification
        self.softmax = nn.Softmax(dim=1)

    def _get_flattened_size(self, input_channels, height, width):
        with torch.no_grad():
            x = torch.zeros(1, input_channels, height, width)
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.pool(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        # Group 1 (G1): Time-frequency convolution
        x = self.relu1(self.bn1(self.conv1(x)))

        # MaxPool
        x = self.pool(x)

        # Group 2 (G2): Spatial convolution
        x = self.relu2(self.bn2(self.conv2(x)))

        # MaxPool
        x = self.pool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No softmax here; handled by CrossEntropyLoss
        x = self.softmax(x)
        return x


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    # y = filtfilt(b, a, data)
    return y


def compute_stft_for_segment(segment, fs, n_fft=256, hop_length=128):
    """
    Compute STFT for a multi-channel EEG segment.
    Args:
        segment: 2D array of shape (256, 14), where rows are time points and columns are channels.
        fs: Sampling frequency in Hz.
        n_fft: Number of FFT points.
        hop_length: Number of overlapping samples between windows.
    Returns:
        stft_results: 3D array of shape (n_channels, n_freq_bins, n_time_frames)
                      containing the magnitude spectrograms for each channel.
    """
    n_channels = segment.shape[1]
    stft_results = pd.DataFrame()

    for ch in range(n_channels):
        # Compute STFT for each channel
        _, _, Zxx = stft(segment[:, ch], fs, nperseg=n_fft, noverlap=hop_length)

        magnitude = np.abs(Zxx)
        magnitude_db = 20 * np.log10(
            magnitude + 1e-6
        )  # Add small value to avoid log(0)

        # merge with previous dataframes
        stft_results = pd.concat([stft_results, pd.DataFrame(magnitude_db)], axis=1)

    return np.array(stft_results)  # Shape: (n_channels, n_freq_bins, n_time_frames)


rf_model = None
cnn_model = None
min_stft = None
max_stft = None
input = []
low_cut = 0.5
high_cut = 30.0
fs = 128

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

segment_size = 256


def initialize():
    global rf_model, min_stft, max_stft, cnn_model
    print("initializing ")
    with open("rf_model_custom.joblib", "rb") as f:
        rf_model = joblib.load(f)
        print("RF Model loaded")
    with open("min_stft.npy", "rb") as f:
        min_stft = np.load(f)
        print("Min STFT loaded")
    with open("max_stft.npy", "rb") as f:
        max_stft = np.load(f)
        print("Max STFT loaded")
    with open("model_cnn.pth", "rb") as f:
        cnn_model = EEGCNN(1, 10)
        cnn_model.load_state_dict(torch.load(f))
        print("CNN Model loaded")


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
    global input, channel_values, cnn_model
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

        if len(input) == segment_size:  # rf model
            # need to change
            data = bandpass_filter(input, low_cut, high_cut, fs)
            extracted = compute_stft_for_segment(data, fs)
            extracted = extracted.reshape(1, 1, extracted.shape[0], extracted.shape[1])
            # chec if any is bigger than min_stft raise error
            # we raise error since we do not want to predict
            # if extracted.min() < min_stft:
            #     raise ValueError("Min value is less than min_stft")
            # if extracted.max() > max_stft:
            #     raise ValueError("Max value is more than max_stft")

            min_stft = np.min(extracted)
            max_stft = np.max(extracted)

            # after it passes, normalize the data
            extracted = (extracted - min_stft) / (max_stft - min_stft)
            proba = (
                cnn_model(torch.tensor(extracted, dtype=torch.float32)).detach().numpy()
            )
            # get probability
            # do a softmax
            #
            softmax = nn.Softmax(dim=1)
            proba = softmax(torch.tensor(proba)).detach().numpy().flatten()

            print(proba)
            max_proba = np.max(proba)
            if max_proba > 0.15:
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
