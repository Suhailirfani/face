import os
import base64
import numpy as np
import cv2
from django.conf import settings
from django.contrib.auth.models import User

# Try to import the primary face_recognition library
try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False

# Load OpenCV's Haar Cascade classifier for face detection (fallback)
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def decode_image(image_data):
    """
    Decodes image data from various formats (base64 string, bytes, or file-like object)
    and returns a standard OpenCV BGR image (numpy array).
    """
    if not image_data:
        return None
    
    try:
        # Case 1: Base64 string (e.g. data:image/jpeg;base64,...)
        if isinstance(image_data, str):
            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]
            img_bytes = base64.b64decode(image_data)
        # Case 2: Django UploadedFile or file-like object
        elif hasattr(image_data, 'read'):
            image_data.seek(0)
            img_bytes = image_data.read()
        # Case 3: Bytes object
        elif isinstance(image_data, bytes):
            img_bytes = image_data
        else:
            return None
            
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def detect_and_extract_face(image_np):
    """
    Detects the first face in the image.
    If using face_recognition: returns the 128-D encoding list of floats.
    If using OpenCV fallback: returns the cropped grayscale face image (numpy array) and the bounding box.
    """
    if image_np is None:
        return None
        
    if HAS_FACE_RECOGNITION:
        # Convert BGR (OpenCV format) to RGB (face_recognition format)
        rgb_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        # Find all face locations and encodings
        face_locations = face_recognition.face_locations(rgb_img)
        if not face_locations:
            return None
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        if face_encodings:
            # Return the first face's encoding as a list of floats
            return list(face_encodings[0])
        return None
    else:
        # Fallback to OpenCV Haar Cascade detection
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        if len(faces) == 0:
            return None
            
        # Get the largest face by area
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        # Crop the face and resize it to a standard size (e.g., 200x200) for LBPH consistency
        face_crop = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, (200, 200))
        return face_resized

def verify_face_present(image_np):
    """
    Simple utility to verify if a face is present in the image.
    Returns True if face is present, False otherwise.
    """
    if image_np is None:
        return False
        
    if HAS_FACE_RECOGNITION:
        rgb_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        return len(face_locations) > 0
    else:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        return len(faces) > 0

def authenticate_by_face(login_image_data, confidence_threshold=70.0):
    """
    Matches the login face image against registered users.
    Returns the User object if a match is found, or None.
    """
    from accounts.models import UserProfile  # Lazy import to avoid circular dependency
    
    login_img = decode_image(login_image_data)
    if login_img is None:
        return None, "Invalid image format or failed to capture."
        
    if HAS_FACE_RECOGNITION:
        # Extract the encoding for the login face
        login_encoding = detect_and_extract_face(login_img)
        if login_encoding is None:
            return None, "No face detected in the camera frame."
            
        # Fetch all user profiles with valid encodings
        profiles = UserProfile.objects.exclude(face_encoding__isnull=True)
        if not profiles.exists():
            return None, "No registered users in the database."
            
        known_encodings = []
        user_map = []
        for p in profiles:
            known_encodings.append(np.array(p.face_encoding))
            user_map.append(p.user)
            
        # Compare faces
        matches = face_recognition.compare_faces(known_encodings, np.array(login_encoding), tolerance=0.6)
        face_distances = face_recognition.face_distance(known_encodings, np.array(login_encoding))
        
        best_match_idx = np.argmin(face_distances)
        if matches[best_match_idx]:
            matched_user = user_map[best_match_idx]
            return matched_user, f"Welcome back, {matched_user.username}!"
        else:
            return None, "Face did not match any registered user."
            
    else:
        # OpenCV Fallback - Train LBPH recognizer on the fly
        login_face = detect_and_extract_face(login_img)
        if login_face is None:
            return None, "No face detected in the camera frame."
            
        profiles = UserProfile.objects.all()
        if not profiles.exists():
            return None, "No registered users in the database."
            
        training_faces = []
        labels = []
        user_map = {}
        
        label_counter = 1
        for p in profiles:
            if not p.face_image:
                continue
            # Read the profile image from media folder
            profile_img_path = p.face_image.path
            if not os.path.exists(profile_img_path):
                continue
                
            img = cv2.imread(profile_img_path)
            face_crop = detect_and_extract_face(img)
            if face_crop is not None:
                training_faces.append(face_crop)
                labels.append(label_counter)
                user_map[label_counter] = p.user
                label_counter += 1
                
        if not training_faces:
            return None, "No valid faces found in registered user images."
            
        # Create and train LBPH Face Recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(training_faces, np.array(labels, dtype=np.int32))
        
        # Predict
        label, confidence = recognizer.predict(login_face)
        print(f"OpenCV Face Match - Predicted label: {label}, Confidence: {confidence:.2f}")
        
        # In LBPH, confidence represents distance (lower is better).
        # Typically, a confidence value under 70-80 indicates a good match.
        if confidence < confidence_threshold:
            matched_user = user_map.get(label)
            if matched_user:
                return matched_user, f"Welcome back, {matched_user.username}! (Match confidence: {confidence:.1f})"
        
        return None, "Face did not match any registered user."
