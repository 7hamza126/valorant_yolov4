import cv2
import numpy as np
import win32con
import mss
import time  # Added for FPS calculation
import win32gui

# Load YOLOv4-tiny model with CUDA acceleration if available
net = cv2.dnn.readNet("yolov4-tiny-custom_final.weights", "yolov4-tiny-custom.cfg")


# Configuration
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
ACTIVE_WINDOW = True
INPUT_SIZE = (416, 416)

# Precompute screen dimensions and capture region
with mss.mss() as sct:
    monitor = {
        "top": sct.monitors[0]["height"] // 2 - 206,
        "left": sct.monitors[0]["width"] // 2 - 206,
        "width": 412,
        "height": 412,
    }

def detect_enemy(image):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, INPUT_SIZE, swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    h, w = image.shape[:2]
    boxes, confidences = [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            confidence = scores[0]
            if confidence > CONF_THRESHOLD:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)
                x = int(center_x - bw / 2)
                y = int(center_y - bh / 2)
                boxes.append([x, y, bw, bh])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    final_boxes = []

    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            x, y, bw, bh = boxes[i]
            final_boxes.append([x, y, bw, bh])
            if ACTIVE_WINDOW:
                cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                label = f"ENEMY: {confidences[i]:.2f}"
                cv2.putText(image, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    return image


# Initialize screen capture and FPS variables
sct = mss.mss()
prev_time = 0
new_time = 0

# Main loop
while True:
    # Start FPS timer
    new_time = time.time()

    # Capture screen
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Process frame
    processed = detect_enemy(frame)

    # Calculate FPS
    fps = 1 / (new_time - prev_time)
    prev_time = new_time

    # Display FPS counter
    if ACTIVE_WINDOW:
        cv2.putText(processed, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Detected ENEMY", processed)
        hwnd = win32gui.FindWindow(None, "Detected ENEMY")
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                              win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()