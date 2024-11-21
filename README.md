# EEG with Muse 2 and Epoc+ Emotiv

Before using this, please use a python virtual environment. You can create a virtual environment by running the following command in the terminal:

```bash
python3 -m venv .venv
```

Then, activate the virtual environment by running the following command in the terminal:

```bash
source .venv/bin/activate
```

Then install the required packages by running the following command in the terminal:

```bash
pip install -r requirements.txt
```


1. Muse 2
   
You can use muse 2 with either 1) Muse Monitor or 2) Muse Lab. The Muse Monitor is a paid app, but it is more user-friendly and has more features. Muse Lab is free, but it is less user-friendly and has fewer features.

1.1 Mind Monitor

Install through the App Store or Google Play Store. You will need to purchase the app to use it. The app is user-friendly and has many features. You can record EEG data, view EEG data in real-time, and export EEG data to a CSV file.

After installing the app, you need to turn on data stream. Open the stream port at 5000. Remember that you should connect Muse 2 with your phone to enable streaming. After streaming is turned on use either `./muse2/mindmonitor_stream.py` or `./muse2/mindmonitor_stream_focus.py` to stream in real time. The file name with the word `focus` will use the knn model to classify the data in real time.

To activate it run the following command in the terminal:

```bash
cd muse2
python mindmonitor_stream.py
```



1.2 Muse Lab

Download the Muse Lab from the Muse website. The software is free, but it is less user-friendly and has fewer features. You can record EEG data and view EEG data in real-time, but you cannot export EEG data to a CSV file. 

You also need to download the muse mobile app from the app store or google play store. You will need to connect the muse mobile app with the muse lab to stream the data. To start streaming you should connect the muse 2 device to the mobile muse app. 

You need to set the port of streaming with mobile muse app. Set the port to 5001 in the settings. Then check your Muse lab to see if the stream works well.

To connect the muse lab with the streaming, check OSC in the Muse lab. <img src="./figures/Screenshot 2024-11-20 at 11.11.59 PM.png">
After successful connection you will be able to see the data in the Muse lab. 

2. Emotiv Epoc+

You must have a valid subscription license to use the Emotiv Epoc+ device. You can use the Emotiv Epoc+ with the Emotiv Pro software. The Emotiv Pro software is user-friendly and has many features. You can record EEG data, view EEG data in real-time, and export EEG data to a CSV file.

However, 



