import cv2
import numpy as np

class YOLOv3ObjectDetector:
    def __init__(self, weights_path, config_path, classes_path):

        self.net = cv2.dnn.readNet(weights_path, config_path)
        
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        self.output_layers = self.get_output_layers()
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
    
    def get_output_layers(self):

        layer_names = self.net.getLayerNames()
        return [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def detect_objects(self, frame):
        
        height, width = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        self.net.setInput(blob)
        
        outs = self.net.forward(self.output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        for i in indexes:
            if len(indexes) > 0:
                i = i[0] if isinstance(i, list) else i
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add label and confidence
                text = f'{label}: {confidence:.2f}'
                cv2.putText(frame, text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def process_video(self, video_source=0):
        
        cap = cv2.VideoCapture(video_source)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            frame_with_detections = self.detect_objects(frame)
            
            # Display
            cv2.imshow('YOLOv3 Object Detection', frame_with_detections)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Paths to YOLO files (you need to download these)
    WEIGHTS_PATH = 'yolov3.weights'
    CONFIG_PATH = 'yolov3.cfg'
    CLASSES_PATH = 'coco.names'
    
    # Initialize detector
    detector = YOLOv3ObjectDetector(WEIGHTS_PATH, CONFIG_PATH, CLASSES_PATH)
    
    # Start real-time detection (0 for webcam, or provide video file path)
    detector.process_video()

if __name__ == '__main__':
    main()