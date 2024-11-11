from pythonosc import dispatcher
from pythonosc import osc_server
from scipy.signal import butter, lfilter
import numpy as np
from collections import deque

# Sampling frequency (adjust to match your EEG device)
fs = 220  # For example, 256 Hz

# Define filter parameters for different EEG bands
bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}


# Prepare deques to store the recent samples for each channel
buffer_size = 236  # Adjust this based on desired window length
tp9_buffer = deque(maxlen=buffer_size)
af7_buffer = deque(maxlen=buffer_size)
af8_buffer = deque(maxlen=buffer_size)
tp10_buffer = deque(maxlen=buffer_size)


# Define a handler function to process incoming OSC messages
def message_handler(address, *args):
    # Append incoming data to each buffer
    # tp9_buffer.append(args[0])
    # af7_buffer.append(args[1])
    # af8_buffer.append(args[2])
    # tp10_buffer.append(args[3])
    print(args)

    # Ensure buffers are full before calculating band power
    if len(tp9_buffer) == buffer_size:
        # Calculate power for each frequency band and each channel

        # Print or log the calculated band powers
        # print(
        #     f"Alpha Power - TP9: {tp9_alpha:.2f}, AF7: {af7_alpha:.2f}, AF8: {af8_alpha:.2f}, TP10: {tp10_alpha:.2f}"
        # )
        # print all bands
        # print(
        #     f"Delta Power - TP9: {compute_band_power(tp9_buffer, 'delta'):.2f}, AF7: {compute_band_power(af7_buffer, 'delta'):.2f}, AF8: {compute_band_power(af8_buffer, 'delta'):.2f}, TP10: {compute_band_power(tp10_buffer, 'delta'):.2f}"
        # )
        print(
            f"Theta Power - TP9: {compute_band_power(tp9_buffer, 'theta'):.2f}, AF7: {compute_band_power(af7_buffer, 'theta'):.2f}, AF8: {compute_band_power(af8_buffer, 'theta'):.2f}, TP10: {compute_band_power(tp10_buffer, 'theta'):.2f}"
        )

        # Repeat similarly for other bands if needed, e.g., theta, beta, etc.


# Set up the dispatcher and map OSC addresses to the handler function
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/muse/elements/delta_absolute", message_handler)

# Set up and start the OSC server
ip = "127.0.0.1"  # Use your machine's IP for networked listening if needed
port = 5000  # Port to listen on

ip = "0.0.0.0"
port = 5000
server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)

print(f"Listening for OSC messages on {ip}:{port}")
server.serve_forever()  # Keeps the server running and listening for messages
