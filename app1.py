from flask import Flask, jsonify, render_template, request
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytube import YouTube
import json
import io
import base64
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')



app = Flask(__name__, static_folder='static')

SEQUENCE_LENGTH = 3
DIM = (224, 224)
num_classes = 15

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, ConvLSTM3D,AveragePooling3D, MaxPooling3D,Input
from tensorflow.keras.layers import Bidirectional, ConvLSTM2D,AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Flatten, TimeDistributed, ZeroPadding3D,Dropout,BatchNormalization, LSTM
from PIL import Image
from tensorflow.keras.optimizers import Adam,Adagrad,Adadelta,SGD
from tensorflow.keras.applications import InceptionV3, DenseNet121, VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2,l1,l1_l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical,plot_model

def cnn_lstm_model():
    base_architecture = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_architecture = Model(inputs=base_architecture.input, outputs=base_architecture.get_layer('mixed7').output)

    for layer in base_architecture.layers:
        layer.trainable = False

    model = Sequential()
    model.add(TimeDistributed(base_architecture, input_shape=(SEQUENCE_LENGTH, 224, 224, 3)))
    
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(LSTM(units=32, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    return model

final_model= cnn_lstm_model()
final_model.load_weights("C://Users//asus//Downloads//SportsRS2//SportsRS//SportsRS2//models//IV3_LSTM4_wt.h5")

from keras.models import load_model
from Katna.video import Video
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os


# Define the class labels
class_labels = ['VolleyballSpiking',
 'Biking',
 'HorseRiding',
 'Fencing',
 'Archery',
 'Boxing',
 'TableTennisShot',
 'GolfSwing',
 'SoccerPenalty',
 'TennisSwing',
 'Basketball',
 'Cricket',
 'Rafting',
 'Diving',
 'JavelinThrow']

# NEW FRAME EXTRACTION
def frames_extraction(video_path):
    frames_list = []
    
    # Initialize Video module
    vd = Video()
    
    # Number of images to be returned
    no_of_frames_to_returned = SEQUENCE_LENGTH
    location = 'saved_frames'
    # Initialize diskwriter to save data at desired location
    diskwriter = KeyFrameDiskWriter(location=location)
    
    # Extract keyframes and process data with diskwriter
    vd.extract_video_keyframes(
        no_of_frames=no_of_frames_to_returned, file_path=video_path,
        writer=diskwriter
    )
    frames = []

    # List all files in the folder
    files = os.listdir(location)

    # Filter JPEG files
    jpeg_files = [file for file in files if file.lower().endswith('.jpeg')]

    # Read JPEG files and append to frames list
    for file in jpeg_files:
        file_path = os.path.join(location, file)
        frame = cv2.imread(file_path)  # Read image using OpenCV
        if frame is not None:
            frames.append(frame)


    # Perform necessary preprocessing
    for frame in frames:
        resized_frame = cv2.resize(frame, DIM)
        grayscale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)
    
    return frames_list




def download_youtube_frames(youtube_url):
    yt = YouTube(youtube_url)
    stream = yt.streams.get_highest_resolution()
  
    video_filename = f"video.mp4"
    
    # Construct full path for saving
    video_path = os.path.join("", video_filename)
    
    # Download the video with the generated filename
    # stream.download(output_path="", filename=video_filename)
    stream.download(output_path="C:/Users/asus/Downloads/SportsRS2/SportsRS/SportsRS2/static", filename=video_filename)

    return video_path



def delete_files():
    if os.path.exists("static/video.mp4"):
        # first close the file then delete it
        # os.close("static/video.mp4")
        os.remove("static/video.mp4")
        print(os.path.exists("static/video.mp4"))
    if os.path.exists("video.mp4"):
        # os.close("video.mp4")
        os.remove("video.mp4")
        print(os.path.exists("video.mp4"))


        
def predict_single_action(input_video_file_path, SEQUENCE_LENGTH):
    frames = frames_extraction(input_video_file_path)
    plt.figure(figsize=(15, 3))
    for i in range(len(frames)):
        plt.subplot(1, len(frames), i + 1)
        plt.imshow(frames[i])
        plt.axis('off')
    # Encode the plot as a base64-encoded string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_data = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    frames = np.asarray(frames)
    frames = frames.reshape(-1, SEQUENCE_LENGTH, DIM[0], DIM[1], 3)
    
    prediction = final_model.predict(frames)
    predicted_label = np.argmax(prediction, axis=1)
    
    three_confidence = np.sort(prediction[0])[-3:]
    three_classes = [class_labels[i] for i in np.argsort(prediction[0])[-3:]]
    
    return class_labels[predicted_label[0]], plot_data, three_confidence, three_classes





@app.route('/')
def index():
    page_refreshed = 'max-age=0' in request.headers.get('Cache-Control', '')
    
    if page_refreshed:
        print("The page was refreshed")
        delete_files()
      
   

    return render_template('index.html', page_refreshed=page_refreshed)



from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    video_file = ""
    video_path = ""
    if request.form['youtube_url']:
        youtube_url = request.form['youtube_url']
        youtube_url = "'" + youtube_url + "'"
        video_path = download_youtube_frames(youtube_url)
        video_path = "static/video.mp4"
    elif request.files['video_file']:
        video_file = request.files['video_file']
        if video_file and allowed_file(video_file.filename):
            filename = secure_filename(video_file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(video_path)
            # rename the file to video.mp4 if there exists a file with the same name in the static folder then delete it
            if os.path.exists("static/video.mp4"):
                os.remove("static/video.mp4")
            os.rename(video_path, "static/video.mp4")
            video_path = "static/video.mp4"

    predicted_class, plot_data = predict_single_action(video_path, SEQUENCE_LENGTH)
    confidence1 = three_confidence[2]
    confidence2 = three_confidence[1]
    confidence3 = three_confidence[0]
    # converting confidence to string and percentage format for json serialization
    confidence1 = str(round(confidence1 * 100, 2)) + "%"
    confidence2 = str(round(confidence2 * 100, 2)) + "%"
    confidence3 = str(round(confidence3 * 100, 2)) + "%"
    
    class1 = three_classes[2]
    class2 = three_classes[1]
    class3 = three_classes[0]
    print(predicted_class)

    return json.dumps({'predicted_class': predicted_class, 'plot_data': plot_data, 'video_path':video_path, 'confidence1': confidence1, 'confidence2': confidence2, 'confidence3': confidence3, 'class1': class1, 'class2': class2, 'class3': class3})



if __name__ == '__main__':
    app.run(debug=True)
