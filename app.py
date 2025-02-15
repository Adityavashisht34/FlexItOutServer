from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import cv2
import numpy as np
import base64
import mediapipe as mp
import os
import logging
import datetime
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def process_frame(frame, exercise_type):
    try:
        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process with MediaPipe Pose
        results = pose.process(image)

        if not results.pose_landmarks:
            return {'reps': 0}

        landmarks = results.pose_landmarks.landmark

        if exercise_type == 'Pushups':
            # Pushup logic
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            angle = calculate_angle(left_shoulder, left_elbow, [left_elbow[0], left_elbow[1] - 0.1])
            if angle > 100:
                return {'reps': 1, 'stage': 'up'}
            if angle < 55:
                return {'reps': 1, 'stage': 'down'}

        elif exercise_type == 'Squats':
            # Squat logic
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            angle = calculate_angle(left_hip, left_knee, left_ankle)
            if angle > 150:
                return {'reps': 1, 'stage': 'up'}
            if angle < 110:
                return {'reps': 1, 'stage': 'down'}

        elif exercise_type == 'Bicep Curls':
            # Bicep curl logic
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            if angle > 160:
                return {'reps': 1, 'stage': 'down'}
            if angle < 30:
                return {'reps': 1, 'stage': 'up'}

        return {'reps': 0}

    except Exception as e:
        print(f"Error processing frame: {e}")
        return {'reps': 0}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    print("Server is running and healthy")
    return jsonify({
        'status': 'healthy',
        'message': 'Server is running successfully',
        'timestamp': datetime.datetime.now().isoformat()
    }), 200

@app.route('/process_frame', methods=['POST'])
@limiter.limit("10 per second")
def process_frame_endpoint():
    try:
        print("Received frame processing request")
        data = request.json
        if not data or 'frame' not in data or 'exercise_type' not in data:
            logger.warning("Invalid request data")
            return jsonify({'error': 'Invalid request data'}), 400

        frame_data = data['frame']
        exercise_type = data['exercise_type']

        if exercise_type not in ['Pushups', 'Squats', 'Bicep Curls']:
            logger.warning(f"Invalid exercise type: {exercise_type}")
            return jsonify({'error': 'Invalid exercise type'}), 400

        # Decode base64 frame
        frame_bytes = base64.b64decode(frame_data)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, flags=cv2.IMREAD_COLOR)

        # Process frame with AI model
        result = process_frame(frame, exercise_type)

        logger.info(f"Successfully processed frame for {exercise_type}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
