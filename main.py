import streamlit as st
import numpy as np
import cv2
import cvzone
import tempfile
from collections import defaultdict

# Load the pre-trained MobileNet SSD model and class labels
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',  # Adjust the path as necessary
    'mobilenet_iter_73000.caffemodel'  # Adjust the path as necessary
)

classNames = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
              "sofa", "train", "tvmonitor"]

# Define lane limits and their corresponding colors
limits = {
    'limit1': [935, 90, 1275, 90],
    'limit2': [935, 110, 1275, 110],
    'limit3': [1365, 120, 1365, 360],
    'limit4': [1385, 120, 1385, 360],
    'limit5': [600, 70, 600, 170],
    'limit6': [620, 70, 620, 170],
    'limit7': [450, 500, 1240, 500],
    'limit8': [450, 520, 1240, 520],
}

lane_colors = {
    1: (255, 0, 0),   # Blue
    2: (0, 255, 0),   # Green
    3: (0, 0, 255),   # Red
    4: (255, 255, 0)  # Cyan
}

# Initialize counts
totalCounts = {1: [], 2: [], 3: [], 4: []}

# Simple Centroid Tracker
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = defaultdict(lambda: {"centroid": None, "disappeared": 0})
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = {"centroid": centroid, "disappeared": 0}
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]

    def update(self, inputCentroids):
        if len(inputCentroids) == 0:
            for objectID in list(self.objects.keys()):
                self.objects[objectID]["disappeared"] += 1
                if self.objects[objectID]["disappeared"] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        newObjects = {}
        for i, inputCentroid in enumerate(inputCentroids):
            distances = {objectID: np.linalg.norm(np.array(self.objects[objectID]["centroid"]) - np.array(inputCentroid))
                         for objectID in self.objects.keys()}
            if distances:
                closestObjectID = min(distances, key=distances.get)
                if distances[closestObjectID] < 50:  # threshold to match centroids
                    newObjects[closestObjectID] = {"centroid": inputCentroid, "disappeared": 0}
                else:
                    newObjects[self.nextObjectID] = {"centroid": inputCentroid, "disappeared": 0}
                    self.nextObjectID += 1
            else:
                newObjects[self.nextObjectID] = {"centroid": inputCentroid, "disappeared": 0}
                self.nextObjectID += 1

        self.objects = newObjects
        return self.objects

# Initialize the tracker
ct = CentroidTracker()

def process_frame(img):
    if len(img.shape) == 3:
        height, width, _ = img.shape
    else:
        height, width = img.shape

    # Prepare the image for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    inputCentroids = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if classNames[idx] in ["car", "bus", "motorbike", "truck"]:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x1, y1, x2, y2) = box.astype("int")
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                inputCentroids.append((cx, cy))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    objects = ct.update(inputCentroids)

    for objectID, info in objects.items():
        centroid = info["centroid"]
        cv2.putText(img, f'ID {objectID}', (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        for i in range(1, 5):
            if (limits[f'limit{2*i-1}'][0] < centroid[0] < limits[f'limit{2*i-1}'][2] and
                limits[f'limit{2*i-1}'][1] - 15 < centroid[1] < limits[f'limit{2*i-1}'][1] + 15) or \
               (limits[f'limit{2*i}'][0] < centroid[0] < limits[f'limit{2*i}'][2] and
                limits[f'limit{2*i}'][1] - 15 < centroid[1] < limits[f'limit{2*i}'][1] + 15):
                if totalCounts[i].count(objectID) == 0:
                    totalCounts[i].append(objectID)
                    color = lane_colors[i]
                    cv2.line(img, (limits[f'limit{2*i-1}'][0], limits[f'limit{2*i-1}'][1]),
                             (limits[f'limit{2*i-1}'][2], limits[f'limit{2*i-1}'][3]), color, 3)
                    cv2.line(img, (limits[f'limit{2*i}'][0], limits[f'limit{2*i}'][1]),
                             (limits[f'limit{2*i}'][2], limits[f'limit{2*i}'][3]), color, 3)

    for i in range(1, 5):
        cvzone.putTextRect(img, f' {i}st Lane: {len(totalCounts[i])}', (25, 75 + 70 * (i-1)), 2, thickness=2, colorR=lane_colors[i], colorT=(15, 15, 15))

    return img

st.title("Traffic Flow Monitoring")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()
