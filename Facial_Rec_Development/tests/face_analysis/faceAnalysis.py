'''
Facial detection using OpenCV:
- No deeplearning training going on. using pre-trained models for face detection.
- Detect faces and the eyes
- https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
- Cascade clasifiers are trained on 100s of images that contain the object of interest and other images without the objects
- Haar features are used to perform Adabosst training followed by Cascading classification

'''

import numpy as np
import cv2
import math

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

def static_image_analysis(
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

def static_image_analysis_approach_2(
    img_path: str,
    face_cascade_path: str = '../../haarcascades_models/haarcascade_frontalface_default.xml',
    eye_cascade_path: str = '../../haarcascades_models/haarcascade_eye.xml'
):
    face_detector = cv2.CascadeClassifier(face_cascade_path)
    eye_detector = cv2.CascadeClassifier(eye_cascade_path)

    # ------ Reading the image ------
    img = cv2.imread(img_path)
    img_raw = img.copy()
    display_intermediate_stage(img, "Original Image", wait_time=0)

    # ------ Face detection ------
    faces = face_detector.detectMultiScale(img, 1.3, 5)
    # Draw face detection on original image for visualization
    img_with_face_detection = img.copy()
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(img_with_face_detection, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
    display_intermediate_stage(img_with_face_detection, "Post Face Detection", wait_time=0)
    
    face_x, face_y, face_w, face_h = faces[0]
    img = img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
    display_intermediate_stage(img, "Cropped Face Region", wait_time=0)
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
        raise ValueError(f"Expected at least 2 eyes, but only {len(eyes)} eye(s) detected")
 
    index = 0
    color = (0, 255, 0)  # Green color for eye bounding boxes
    for (eye_x, eye_y, eye_w, eye_h) in eyes:
        if index == 0:
            eye_1 = (eye_x, eye_y, eye_w, eye_h)
        elif index == 1:
            eye_2 = (eye_x, eye_y, eye_w, eye_h)
        
        cv2.rectangle(img,(eye_x, eye_y),(eye_x+eye_w, eye_y+eye_h), color, 2)
        index = index + 1
    
    display_intermediate_stage(img, "Post Eye Detection", wait_time=0)

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
    
    cv2.circle(img, left_eye_center, 2, (255, 0, 0) , 2)
    cv2.circle(img, right_eye_center, 2, (255, 0, 0) , 2)
    cv2.line(img,right_eye_center, left_eye_center,(67,67,67),2)

    # ------ Determine direction of rotation ------
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 # rotate same direction to clock
        print("rotate to clock direction")
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 # rotate inverse direction of clock
        print("rotate to inverse clock direction")
 
    cv2.circle(img, point_3rd, 2, (255, 0, 0) , 2)
    
    cv2.line(img,right_eye_center, left_eye_center,(67,67,67),2)
    cv2.line(img,left_eye_center, point_3rd,(67,67,67),2)
    cv2.line(img,right_eye_center, point_3rd,(67,67,67),2)

    display_intermediate_stage(img, "Eye Centers and Alignment Lines", wait_time=0)

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
    new_img = Image.fromarray(img_raw)
    new_img = np.array(new_img.rotate(direction * angle))

    display_intermediate_stage(new_img, "Final Rotated Image", wait_time=0)
    cv2.destroyAllWindows()


########################################################

def camera_test():
    # Check if your system can detect camera and what is the source number
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
    # static_image_analysis(img_path='../database/Obama/img1.jpg')
    static_image_analysis_approach_2(img_path='../database/tiltedhead.jpg')
    # camera_test()
    # realtime_video_analysis()