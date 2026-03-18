'''
Facial detection using OpenCV:
- No deeplearning training going on. using pre-trained models for face detection.
- Detect faces and the eyes
- https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
- Cascade clasifiers are trained on 100s of images that contain the object of interest and other images without the objects
- Haar features are used to perform Adabosst training followed by Cascading classification

'''

import math
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow importing from Facial_Rec_Development
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from Facial_Rec_Development.ImageProcessor import ImagePreprocessor

_default_cascade_dir = _repo_root / "Facial_Rec_Development" / "haarcascades_models"
_image_preprocessor = ImagePreprocessor(
    face_cascade_path=str(_default_cascade_dir / "haarcascade_frontalface_default.xml"),
    eye_cascade_path=str(_default_cascade_dir / "haarcascade_eye.xml"),
)


def detect_faces_and_eyes(img, gray, face_cascade, eye_cascade, scale_factor=1.3, min_neighbors=5):
    """
    Common function to detect faces and eyes in an image and draw bounding boxes.
    
    Args:
        img: Color image (BGR format) to draw bounding boxes on
        gray: Grayscale image for detection
        face_cascade: CascadeClassifier for face detection
        eye_cascade: CascadeClassifier for eye detection
        scale_factor: Scale factor for detectMultiScale (default: 1.3)
        min_neighbors: Minimum neighbors for detectMultiScale (default: 5)
    
    Returns:
        img: Image with bounding boxes drawn around faces and eyes
    """
    # First detect face and then look for eyes inside the face.
    # Multiscale refers to detecting objects (faces) at multiple scales.
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
    # Above faces returns a list of rectangles. For each face, it returns
    # a tuple of (x, y, w, h) values that define the rectangle.

    # For each detected face now detect eyes.
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)   # Draw blue bounding box around the face
        roi_gray = gray[y:y+h, x:x+w]  # Original gray image but only the detected face part
        roi_color = img[y:y+h, x:x+w]  # Original color image but only the detected face part. For display purposes
        eyes = eye_cascade.detectMultiScale(roi_gray)  # Use the gray face image to detect eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)  # Draw green bounding boxes around the eyes
    
    return img

def plain_static_image_analysis(
    img_path: str,
    face_cascade_path: str = '../../haarcascades_models/haarcascade_frontalface_default.xml',
    eye_cascade_path: str = '../../haarcascades_models/haarcascade_eye.xml'
):
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use the common function to detect faces and eyes
    img = detect_faces_and_eyes(img, gray, face_cascade, eye_cascade)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def euclidean_distance(a, b):
        x1 = a[0]; y1 = a[1]
        x2 = b[0]; y2 = b[1]
        return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def display_intermediate_stage(img, stage_name, wait_time=0):
    """
    Display an image at an intermediate processing stage.
    
    Args:
        img: Image to display (numpy array in BGR format)
        stage_name: String describing the current stage (e.g., "Post Face Detection", "Post Eye Detection")
        wait_time: Time in milliseconds to wait. 0 means wait for key press, -1 means don't wait (default: 0)
    
    Returns:
        None
    """
    window_title = f"Intermediate Stage: {stage_name}"
    cv2.imshow(window_title, img)
    if wait_time >= 0:
        cv2.waitKey(wait_time)
    # Note: Windows are destroyed at the end of the main function

def facial_alignment(
    img_path: str,
    face_cascade_path: str = '../../haarcascades_models/haarcascade_frontalface_default.xml',
    eye_cascade_path: str = '../../haarcascades_models/haarcascade_eye.xml'
):
    face_detector = cv2.CascadeClassifier(face_cascade_path)
    eye_detector = cv2.CascadeClassifier(eye_cascade_path)

    # ------ Reading the image ------
    img = cv2.imread(img_path)
    # display_intermediate_stage(img, "Original Image", wait_time=0)

    # ------ Face detection ------
    faces = face_detector.detectMultiScale(img, 1.3, 5)

    if len(faces) == 0:
        return None
        # raise ValueError(f"Expected at least 1 face, but no face detected")
        
    # Draw face detection on original image for visualization
    img_with_face_detection = img.copy()
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(img_with_face_detection, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
    # display_intermediate_stage(img_with_face_detection, "Post Face Detection", wait_time=0)
    
    face_x, face_y, face_w, face_h = faces[0]
    img = img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
    
    # Image to perform overlays on
    img_to_edit = img.copy()
    # display_intermediate_stage(img, "Cropped Face Region", wait_time=0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ------ Eye detection ------
    eyes = eye_detector.detectMultiScale(img_gray)
    
    # Sort eyes by area (width * height) in descending order and keep only the two largest
    if len(eyes) > 2:
        # Calculate area for each eye and sort by area (largest first)
        eyes_with_area = [(eye, eye[2] * eye[3]) for eye in eyes]  # (x, y, w, h) -> area = w * h
        eyes_with_area.sort(key=lambda x: x[1], reverse=True)  # Sort by area, descending
        eyes = [eye for eye, _ in eyes_with_area[:2]]  # Keep only the two largest
    
    # Ensure we have at least 2 eyes, otherwise raise an error
    elif len(eyes) < 2:
        return None
        # raise ValueError(f"Expected at least 2 eyes, but only {len(eyes)} eye(s) detected")
 
    index = 0
    color = (0, 255, 0)  # Green color for eye bounding boxes
    for (eye_x, eye_y, eye_w, eye_h) in eyes:
        if index == 0:
            eye_1 = (eye_x, eye_y, eye_w, eye_h)
        elif index == 1:
            eye_2 = (eye_x, eye_y, eye_w, eye_h)
        
        cv2.rectangle(img_to_edit,(eye_x, eye_y),(eye_x+eye_w, eye_y+eye_h), color, 2)
        index = index + 1
    
    # display_intermediate_stage(img_to_edit, "Post Eye Detection", wait_time=0)

    # Determine left eye vs. right eye
    if eye_1[0] < eye_2[0]:
        left_eye = eye_1
        right_eye = eye_2
    else:
        left_eye = eye_2
        right_eye = eye_1

    print(f"left_eye bbox  (x,y,w,h): {left_eye}")
    print(f"right_eye bbox (x,y,w,h): {right_eye}")

    # ------ Detect center of the eyes ------
    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
    
    right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
    right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]

    print(f"left_eye center  (cx,cy): {left_eye_center}")
    print(f"right_eye center (cx,cy): {right_eye_center}")
    
    cv2.circle(img_to_edit, left_eye_center, 2, (255, 0, 0) , 2)
    cv2.circle(img_to_edit, right_eye_center, 2, (255, 0, 0) , 2)
    cv2.line(img_to_edit,right_eye_center, left_eye_center,(67,67,67),2)

    # ------ Determine direction of rotation ------
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 # rotate same direction to clock
        print("rotate to clock direction")
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 # rotate inverse direction of clock
        print("rotate to inverse clock direction")
 
    cv2.circle(img_to_edit, point_3rd, 2, (255, 0, 0) , 2)
    
    cv2.line(img_to_edit,right_eye_center, left_eye_center,(67,67,67),2)
    cv2.line(img_to_edit,left_eye_center, point_3rd,(67,67,67),2)
    cv2.line(img_to_edit,right_eye_center, point_3rd,(67,67,67),2)

    # display_intermediate_stage(img_to_edit, "Eye Centers and Alignment Lines", wait_time=0)

    # Trigonometry to calculate angle    
    a = euclidean_distance(left_eye_center, point_3rd)
    b = euclidean_distance(right_eye_center, left_eye_center)
    c = euclidean_distance(right_eye_center, point_3rd)

    cos_a = (b*b + c*c - a*a)/(2*b*c)
    print("cos(a) = ", cos_a)
    
    angle = np.arccos(cos_a)
    print("angle: ", angle," in radian")
    
    angle = (angle * 180) / math.pi
    print("angle: ", angle," in degree")

    if direction == -1:
        angle = 90 - angle

    from PIL import Image
    # Rotate only the extracted face region (img) instead of the full image
    new_img = Image.fromarray(img)
    new_img = np.array(new_img.rotate(direction * angle))

    # display_intermediate_stage(new_img, "Final Rotated Image", wait_time=0)
    # cv2.destroyAllWindows()
    
    # Return the rotated, aligned face for facial recognition
    return new_img

def facial_alignment_from_array(img, face_detector, eye_detector):
    """
    Perform facial alignment on a numpy array image (BGR format).
    Delegates to Facial_Rec_Development.ImageProcessor.ImagePreprocessor.

    Args:
        img: Numpy array image in BGR format
        face_detector: CascadeClassifier for face detection
        eye_detector: CascadeClassifier for eye detection

    Returns:
        tuple: (new_img, face_bbox, left_eye_abs, right_eye_abs) where:
            - new_img: Rotated, aligned face as numpy array, or None if face/eyes not detected properly
            - face_bbox: Tuple (x, y, w, h) of face bounding box in original image, or None if detection failed
            - left_eye_abs: Tuple (x, y, w, h) of left eye in absolute coordinates, or None if detection failed
            - right_eye_abs: Tuple (x, y, w, h) of right eye in absolute coordinates, or None if detection failed
    """
    return _image_preprocessor.facial_alignment_from_array(img, face_detector, eye_detector)


def capture_aligned_face_from_camera(
    camera_index=0,
    face_cascade_path: str = '../../haarcascades_models/haarcascade_frontalface_default.xml',
    eye_cascade_path: str = '../../haarcascades_models/haarcascade_eye.xml',
    display_preview=True,
    stability_frames=5,
    position_threshold=0.15,
    use_picamera: bool = False,
):
    """
    Capture video from camera and stop when a properly detected face is found and stable.
    Core logic lives in Facial_Rec_Development.ImageProcessor.ImagePreprocessor; this adds
    optional UI when display_preview=True.

    Args:
        camera_index: Index of the camera to use (default: 0)
        face_cascade_path: Path to face cascade classifier XML file (used only for display_preview UI)
        eye_cascade_path: Path to eye cascade classifier XML file (used only for display_preview UI)
        display_preview: Whether to display the camera feed with face detection overlay (default: True)
        stability_frames: Number of consecutive frames the face must be stable (default: 5)
        position_threshold: Threshold for position stability as fraction of face size (default: 0.15 = 15%)

    Returns:
        aligned_face: Rotated, aligned face as numpy array, or None if camera fails or no face detected
    """
    if not display_preview:
        return _image_preprocessor.capture_aligned_face_from_camera(
            camera_index=camera_index,
            stability_frames=stability_frames,
            position_threshold=position_threshold,
            use_picamera=use_picamera,
        )

    # With display: use generator and add UI
    print(f"Camera opened. Looking for a stable face ({stability_frames} consecutive frames)... (Press 'q' to quit)")
    face_detector = cv2.CascadeClassifier(face_cascade_path)
    eye_detector = cv2.CascadeClassifier(eye_cascade_path)
    last_aligned_face = None
    gen = _image_preprocessor.capture_aligned_face_frames(
        camera_index=camera_index,
        stability_frames=stability_frames,
        position_threshold=position_threshold,
        use_picamera=use_picamera,
    )
    try:
        for frame, aligned_face, face_bbox, left_eye_abs, right_eye_abs, stable_frame_count in gen:
            # Draw face detection on frame for preview
            faces = face_detector.detectMultiScale(frame, 1.3, 5)
            preview_frame = frame.copy()
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(preview_frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            if left_eye_abs is not None and right_eye_abs is not None:
                lex, ley, lew, leh = left_eye_abs
                cv2.rectangle(preview_frame, (lex, ley), (lex + lew, ley + leh), (0, 255, 0), 2)
                rex, rey, rew, reh = right_eye_abs
                cv2.rectangle(preview_frame, (rex, rey), (rex + rew, rey + reh), (0, 255, 0), 2)
            cv2.putText(
                preview_frame, f"Stable frames: {stable_frame_count}/{stability_frames}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.imshow('Camera Feed - Looking for Stable Face (Press q to quit)', preview_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User quit before face was detected")
                last_aligned_face = None
                break
            if aligned_face is not None:
                last_aligned_face = aligned_face
            if stable_frame_count >= stability_frames:
                print(f"Face detected and stable for {stability_frames} consecutive frames!")
                break
    finally:
        gen.close()
    cv2.destroyAllWindows()
    return last_aligned_face

def resize_aligned_face(aligned_face, target_size):
    """
    Resize an aligned face image to a specified target size.
    
    Args:
        aligned_face: Numpy array image (output from facial_alignment function)
        target_size: Tuple of (width, height) for the target size (e.g., (152, 152) or (160, 160))
    
    Returns:
        resized_face: Resized numpy array image with the specified dimensions
    """
    width, height = target_size
    resized_face = cv2.resize(aligned_face, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_face


########################################################

def camera_test():
    # Check if system can detect camera and what is the source number
    cams_test = 10
    for i in range(0, cams_test):
        cap = cv2.VideoCapture(i)
        test, frame = cap.read()
        print("i : "+str(i)+" /// result: "+str(test))

# Apply the above logic to a live video
def realtime_video_analysis(
    face_cascade_path: str = '../../haarcascades_models/haarcascade_frontalface_default.xml',
    eye_cascade_path: str = '../../haarcascades_models/haarcascade_eye.xml'
):
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    cap = cv2.VideoCapture(0)

    while 1:
        ret, img = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use the common function to detect faces and eyes
        img = detect_faces_and_eyes(img, gray, face_cascade, eye_cascade)
                
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:      # Press Esc to stop the video
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    # 1. Initial testing with a static image
    # plain_static_image_analysis(img_path='../database/Obama/img1.jpg')

    # 2. Improved testing with a static image + Facial alignment
    # aligned_face = facial_alignment(img_path='../database/Max/img2.jpg')
    # if aligned_face is None:
    #     print("face not detected properly")
    #     exit()
    # print(f"Size of aligned face: {aligned_face.shape}")
    # resized_face = resize_aligned_face(aligned_face, (160, 160))  # For Facenet
    # print(f"Size of resized face: {resized_face.shape}")

    # 3. Testing with a live video
    # camera_test()
    # realtime_video_analysis()
    # On Raspberry Pi, prefer picamera2 if installed.
    import os
    import platform

    use_picamera = False
    if platform.system().lower() == "linux":
        try:
            import picamera2  # noqa: F401

            use_picamera = True
        except Exception:
            use_picamera = False

    aligned_face = capture_aligned_face_from_camera(use_picamera=use_picamera)
    if aligned_face is None:
        print("face not detected properly")
        exit()
    # Show the captured face on laptops/desktop sessions.
    # Suppress OpenCV windows only for headless Linux (common when SSH'd into a Pi).
    show_windows = True
    if platform.system().lower() == "linux" and not os.environ.get("DISPLAY"):
        show_windows = False
    if show_windows:
        cv2.imshow('aligned_face', aligned_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(f"Size of aligned face: {aligned_face.shape}")
    resized_face = resize_aligned_face(aligned_face, (160, 160))  # For Facenet
    print(f"Size of resized face: {resized_face.shape}")