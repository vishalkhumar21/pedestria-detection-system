import os
import streamlit as st
import cv2
import numpy as np

# Set up YOLOv8 model and configuration paths
yolov8_model = "best.pt"
yolov8_config = "yolov8.cfg"

# Load YOLOv8 model and configuration
net = cv2.dnn.readNet(yolov8_model, yolov8_config)

def detect_pedestrians(video_file_path, output_file_path):
    # Open the video file
    cap = cv2.VideoCapture(video_file_path)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file_path, fourcc, 30.0, (640, 360))  # Adjust resolution if needed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect pedestrians in the frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward()

        # Process YOLOv8 output to extract pedestrian detections
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and class_id == 0:  # Assuming class_id 0 represents pedestrians
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    # Calculate bounding box coordinates
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, width, height])

        # Apply non-maximum suppression to remove overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in indices:
            i = i[0]
            x, y, w, h = boxes[i]
            label = str(class_ids[i])
            confidence = confidences[i]

            # Draw bounding box and label
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'Pedestrian {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the frame with detections to the output video
        out.write(frame)

    # Release video objects
    cap.release()
    out.release()

def main():
    st.title('YOLOv8 Pedestrian Detection')
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mpeg"])

    if uploaded_file is not None:
        with open(os.path.join("temp", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully")

        if st.button('Detect Pedestrians'):
            output_file_path = os.path.join("output", uploaded_file.name)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            detect_pedestrians(os.path.join("temp", uploaded_file.name), output_file_path)
            st.success("Pedestrian detection complete!")

            # Display the output video with detections
            video_file = open(output_file_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

if __name__ == '__main__':
    main()
