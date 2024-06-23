# Sports Action Recognition Web App

This web application allows users to upload a video or provide a YouTube URL to recognize sports actions using a CNN-LSTM model. The model predicts the type of sport action from a set of predefined classes.

## Features

- Upload a video file or provide a YouTube URL for action recognition.
- Extracts key frames from the video.
- Uses a pre-trained CNN-LSTM model to predict the action.
- Displays the predicted action along with the top 3 possible actions and their confidence scores.
- Shows the extracted frames used for prediction.

## UI of WebApp
![image](https://github.com/vedant185raut/Sports-Recognition-in-Videos-Using-Deep-Learning/assets/105361526/22330a5d-119e-4522-aa64-5676377cea25)
 ### Upload video or Input URL of youtube video.
 ![image](https://github.com/vedant185raut/Sports-Recognition-in-Videos-Using-Deep-Learning/assets/105361526/a73e399e-409e-4efb-8205-f978727702bc)
 ### Predicted Result
![image](https://github.com/vedant185raut/Sports-Recognition-in-Videos-Using-Deep-Learning/assets/105361526/e7b450d8-945b-4058-b4bd-859d8a56f30c)
![image](https://github.com/vedant185raut/Sports-Recognition-in-Videos-Using-Deep-Learning/assets/105361526/7f6336b4-213b-4ec2-b45a-5ab737f2d1da)

## Technology Stack

- Python
- Flask
- TensorFlow
- PyTube
- OpenCV
- Katna

## Model

The model used in this application is a CNN-LSTM model. The CNN part of the model uses the InceptionV3 architecture pre-trained on ImageNet. The LSTM layers are used to capture the temporal features from the video frames.

## Prerequisites

- Python 3.x
- Pip (Python package installer)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/sports-action-recognition.git
   cd sports-action-recognition
   
2. Create a virtual environment and activate it:
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`

3. Install the required packages:
   pip install -r requirements.txt
   
5. Download the pre-trained model weights and place it in the static/models directory. You can download the model weights from this link.
   Link:

## Usage

1. Run the Flask application:
   python app.py
2. Open your web browser and go to http://127.0.0.1:5000/.

3. Upload a video file or provide a YouTube URL for action recognition.

4. Click on the "Predict" button to see the results.

## Project Structure

├── app.py # Main Flask application script
├── requirements.txt # File listing Python dependencies
├── static # Static files directory
│ ├── models # Directory for model weights
│ │ └── IV3_LSTM4_wt.h5 # Pre-trained model weights file
│ └── video.mp4 # Placeholder for uploaded video
├── templates # HTML templates directory
│ └── index.html # HTML template for web interface
└── saved_frames # Directory to store extracted frames

## Functions

### `cnn_lstm_model()`

Defines the architecture of the CNN-LSTM model used for sports action recognition. The CNN part utilizes the InceptionV3 architecture pre-trained on ImageNet for feature extraction, while LSTM layers capture temporal dependencies from video frames.

### `frames_extraction(video_path)`

Extracts key frames from the provided video file using OpenCV. Frames are resized to (224, 224) pixels and normalized before being returned as a list.

### `download_youtube_frames(youtube_url)`

Downloads a video from YouTube using the provided URL. It uses the PyTube library to fetch the highest resolution video stream and saves it locally.

### `delete_files()`

Deletes temporary files and directories created during the application's operation. This includes the uploaded video files and extracted frames.

### `predict_single_action(input_video_file_path, SEQUENCE_LENGTH)`

Processes the input video file to extract frames, preprocess them, and feed them into the pre-trained CNN-LSTM model (`final_model`). Returns the predicted action label, a base64-encoded plot of frames, confidence scores for the top 3 predicted classes, and their corresponding class names.

## API Endpoints

### `/`

- **Method:** `GET`
- **Description:** Renders the main page for uploading a video or providing a YouTube URL.

### `/predict`

- **Method:** `POST`
- **Description:** Handles video uploads or YouTube URL submissions, performs action recognition, and returns the prediction results.

## Acknowledgements

- The InceptionV3 model used in this project is pre-trained on the ImageNet dataset.
- Special thanks to the developers of PyTube and Katna for their excellent libraries.

