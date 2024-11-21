from pythonosc import dispatcher
from pythonosc import osc_server


# Define a handler function to process incoming OSC messages
def message_handler(address, *args):
    # print(f"Received message at {address} with arguments: {args}")
    tp9 = args[0]  # eeg0
    af7 = args[1]  # eeg1
    af8 = args[2]  # eeg2
    tp10 = args[3]  # eeg3
    print(f"TP9: {tp9}, AF7: {af7}, AF8: {af8}, TP10: {tp10}")


# Set up a dispatcher to map OSC addresses to handler functions
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/eeg", message_handler)  # Listens for messages sent to "/test" address
# You can map additional addresses if needed
# dispatcher.map("/another_address", another_handler_function)

# Set up the OSC server
ip = "127.0.0.1"  # Localhost, or use your machine's IP for networked listening
port = 5000  # Port to listen on
server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)

print(f"Listening for OSC messages on {ip}:{port}")

# Start the server
server.serve_forever()  # Keeps the server running and listening for messages
