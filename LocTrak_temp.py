import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt

# Initialize YOLO (You'll need to download the YOLOv4 weights and cfg file)
net = cv2.dnn.readNet("..\Downloads\yolov4.weights", "..\Downloads\yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize MediaPipe Pose components
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize data storage
left_knee_y_values = []
right_knee_y_values = []
catcher_left_hand_coords = []

# For video input
cap = cv2.VideoCapture('..\Downloads\Video.mp4')
pitcher_pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
catcher_pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


frame_num = 0

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        print('Unable to open video')
        break

    # YOLO person detection
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to show on the object (class id, confidence, bounding box coordinates)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
    person_boxes = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(class_ids[i])

                
            # Initialize a list to store 'person' bounding boxes
            

            # Inside your loop where YOLO detections are processed
            if (label == '0') & (h > 250) & (w > 100):  # Assuming '0' is the class ID for 'person'
                person_boxes.append([x, y, w, h])

    # Sort person_boxes based on the x-coordinate (leftmost)
    person_boxes.sort(key=lambda coords: coords[0])
    
    # Now, person_boxes[0] should be the leftmost person (likely the pitcher)
    pitcher_box = person_boxes[0]

    # Crop the image for the pitcher
    pitcher_cropped_image = image[pitcher_box[1]:pitcher_box[1]+pitcher_box[3], pitcher_box[0]:pitcher_box[0]+pitcher_box[2]]

    # Process with MediaPipe
    pitcher_results = pitcher_pose.process(cv2.cvtColor(pitcher_cropped_image, cv2.COLOR_BGR2RGB))
    #mp_drawing.draw_landmarks(pitcher_cropped_image, pitcher_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    if pitcher_results.pose_landmarks:
        # Get the knee landmarks
        left_knee_landmark = pitcher_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee_landmark = pitcher_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
                        
        # Append the y-values to our lists
        left_knee_y_values.append(left_knee_landmark.y)
        right_knee_y_values.append(right_knee_landmark.y)
        
        
        
    # Sort person_boxes based on the height (shortest)
    person_boxes.sort(key=lambda coords: coords[3])
    
    # Now, person_boxes[0] should be the shortest person (likely the catcher)
    if person_boxes[0][3] < 300:
        catcher_box = person_boxes[0]
        

    # Crop the image for the catcher
    catcher_cropped_image = image[catcher_box[1]:catcher_box[1]+catcher_box[3], catcher_box[0]:catcher_box[0]+catcher_box[2]]

    # Process with MediaPipe
    catcher_results = catcher_pose.process(cv2.cvtColor(catcher_cropped_image, cv2.COLOR_BGR2RGB))
    mp_drawing.draw_landmarks(catcher_cropped_image, catcher_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    
    cv2.imshow('frame', catcher_cropped_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.2)
    
    frame_num += 1
    print(frame_num)
    
    if catcher_results.pose_landmarks:
                       
        # Get the catcher's left hand landmark
        left_hand_landmark = catcher_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                        
        # Store the x, y coordinates
        catcher_left_hand_coords.append((left_hand_landmark.x, left_hand_landmark.y))
        
   


# Plot the left and right knee y-values
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(left_knee_y_values)
plt.title('Left Knee Y-Values Over Time')

plt.subplot(2, 1, 2)
plt.plot(right_knee_y_values)
plt.title('Right Knee Y-Values Over Time')
plt.xlabel('Frame Number')
plt.ylabel('Knee Y-Value')

plt.tight_layout()
plt.show()
                
# Print the catcher's left hand coordinates at the time of peak knee y-values
peak_left_knee_frame = np.argmin(left_knee_y_values)
peak_right_knee_frame = np.argmin(right_knee_y_values)
print(f"Catcher's Left Hand Coordinates at peak Left Knee: {catcher_left_hand_coords[peak_left_knee_frame]}")
print(f"Catcher's Left Hand Coordinates at peak Right Knee: {catcher_left_hand_coords[peak_right_knee_frame]}")    
                