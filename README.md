YOLOv3 Object Detection

Dataset and Pre-trained Weights
This project uses the YOLOv3 model with pre-trained weights on the COCO dataset.

COCO Dataset Classes: coco.names

YOLOv3 Weights: yolov3.weights

YOLOv3 Configuration: yolov3.cfg

🔗 Download the Required Files
You can download the necessary files from the official YOLO website or the darknet repository:

For Linux/macOS (bash):

# Download YOLOv3 weights
wget https://pjreddie.com/media/files/yolov3.weights

# Download YOLOv3 configuration
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg

# Download COCO class names
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

For Windows (Command Prompt):

REM Download YOLOv3 weights
curl -o yolov3.weights https://pjreddie.com/media/files/yolov3.weights

REM Download YOLOv3 configuration
curl -o yolov3.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg

REM Download COCO class names
curl -o coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

Dependencies:
Ensure you have the following libraries installed:

OpenCV (cv2): For image processing and video handling

NumPy: For efficient matrix operations

Installation:
Use the following command to install the required dependencies:

pip install opencv-python numpy

Key Features:
Real-time object detection

Supports both webcam and video file input

Draws bounding boxes with class labels and confidence scores

Configurable confidence and non-maximum suppression thresholds

Detects 80 different object categories from the COCO dataset

Usage Instructions
1. Running the Detection Script
To start real-time detection from your webcam, run the script:

python yolov3_detection.py

2. Using a Video File
To use a specific video file, modify the video_source parameter in the process_video() function:

video_source = "path/to/your/video.mp4"

3. Exiting the Script
Press q to quit the detection.

Performance Considerations:
For real-time performance, a dedicated GPU is recommended.

Accuracy and speed depend on your hardware specifications.

You can fine-tune the detection thresholds:

Confidence Threshold: Reduces false positives by filtering weak detections.

Non-Maximum Suppression (NMS) Threshold: Removes overlapping boxes for cleaner results.

Example Output:

The script displays the detected objects in real-time with labeled bounding boxes. Here’s an example:

Detected Objects: person, car, bicycle, etc.

Bounding Boxes: Highlight detected objects with their confidence scores.

FPS: Displays frames per second for performance monitoring.

References
YOLOv3 GitHub
OpenCV Documentation
