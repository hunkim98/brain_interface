"""Example program to show how to read a multi-channel time series from LSL."""

from pylsl import StreamInlet, resolve_stream
from model.emotiv_digit import EEGNet
import torch

model = EEGNet()
PATH = "model/model_32_14.pth"
input = []


def predict():
    model = EEGNet()
    model.load_state_dict(torch.load(PATH, weights_only=True))
    # model.predict()


def initialize():
    global model
    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()


def interpret(sample):
    af3 = sample[2]
    f7 = sample[3]
    f3 = sample[4]
    fc5 = sample[5]
    t7 = sample[6]
    p7 = sample[7]
    o1 = sample[8]
    o2 = sample[9]
    p8 = sample[10]
    t8 = sample[11]
    fc6 = sample[12]
    f4 = sample[13]
    f8 = sample[14]
    af4 = sample[15]
    try:
        input_row = [af3, f7, f3, fc5, t7, p7, o1, o2, p8, t8, fc6, f4, f8, af4]
        input.append(input_row)
        if len(input) == 32:
            input_tensor = torch.tensor(input)
            input_tensor = input_tensor.unsqueeze(0)
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1)
            print(pred)
            input.clear()
    except:
        print("Error")
        input.clear()


def main():
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    try:
        initialize()
        print("Model loaded")
    except:
        print("Model not loaded")

    streams = resolve_stream("type", "EEG")

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

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
