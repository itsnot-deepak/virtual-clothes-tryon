from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import mediapipe as mp
import numpy as np
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
shirt_image = cv2.imread(r'C:\Users\depak\Downloads\shirt2.png', cv2.IMREAD_UNCHANGED)
if shirt_image.shape[2] == 3:
    shirt_image = cv2.cvtColor(shirt_image, cv2.COLOR_BGR2BGRA)
def home(request):
    return render(request, 'tryon/home.html')
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            if result.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                frame = overlay_shirt(frame, shirt_image, result.pose_landmarks.landmark)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def overlay_shirt(frame, shirt_image, landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    shoulder_width = int(np.linalg.norm(np.array([left_shoulder.x, left_shoulder.y]) - 
                                        np.array([right_shoulder.x, right_shoulder.y])) * frame.shape[1])
    body_height = int(np.linalg.norm(np.array([left_shoulder.x, left_shoulder.y]) - 
                                     np.array([left_hip.x, left_hip.y])) * frame.shape[0])

    shirt_aspect_ratio = shirt_image.shape[1] / shirt_image.shape[0]
    resized_shirt_height = int(body_height * 1.5)
    resized_shirt_width = int(resized_shirt_height * shirt_aspect_ratio)
    resized_shirt = cv2.resize(shirt_image, (resized_shirt_width, resized_shirt_height))
    top_left_x = int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1] - resized_shirt_width / 2)
    top_left_y = int(left_shoulder.y * frame.shape[0] - resized_shirt_height * 0.1)  
    for i in range(resized_shirt.shape[0]):
        for j in range(resized_shirt.shape[1]):
            y = top_left_y + i
            x = top_left_x + j
            if y >= frame.shape[0] or x >= frame.shape[1] or y < 0 or x < 0:
                continue
            alpha_s = resized_shirt[i, j, 3] / 255.0 
            alpha_f = 1.0 - alpha_s  
            frame[y, x] = (alpha_s * resized_shirt[i, j, :3] + alpha_f * frame[y, x]).astype(np.uint8)
    return frame

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
