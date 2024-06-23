# Sports Action Recognition Web App

This web application allows users to upload a video or provide a YouTube URL to recognize sports actions using a CNN-LSTM model. The model predicts the type of sport action from a set of predefined classes.

## Features

- Upload a video file or provide a YouTube URL for action recognition.
- Extracts key frames from the video.
- Uses a pre-trained CNN-LSTM model to predict the action.
- Displays the predicted action along with the top 3 possible actions and their confidence scores.
- Shows the extracted frames used for prediction.

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

.
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── static
│   ├── models
│   │   └── IV3_LSTM4_wt.h5   # Pre-trained model weights
│   └── video.mp4             # Placeholder for the uploaded video
├── templates
│   └── index.html            # HTML template for the web interface
└── saved_frames              # Directory to store extracted frames

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The InceptionV3 model used in this project is pre-trained on the ImageNet dataset.
- Special thanks to the developers of PyTube and Katna for their excellent libraries.
