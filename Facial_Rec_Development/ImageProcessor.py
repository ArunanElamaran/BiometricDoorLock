"""
Image preprocessing for face recognition: load/resize/normalize and optional
facial alignment (detect face + eyes, rotate to align eyes) for path or array input.
"""

import math
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


def _euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    x1, y1 = a[0], a[1]
    x2, y2 = b[0], b[1]
    return math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))


class ImagePreprocessor:
    """
    Converts raw face images to fixed-size tensors.
    Supports:
    - Load from path: resize, optional ImageNet normalize, output tensor.
    - From array (e.g. camera frame): optional face+eye detection and alignment,
      then same resize/normalize pipeline.
    """

    def __init__(
        self,
        image_size: int = 112,
        normalize: bool = True,
        face_cascade_path: Optional[str] = None,
        eye_cascade_path: Optional[str] = None,
        face_box_expand: float = 0.2,  # Expand face box on all sides
    ):
        self.image_size = image_size
        self.normalize = normalize
        self.face_box_expand = face_box_expand
        self._transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        self._mean = (0.485, 0.456, 0.406)
        self._std = (0.229, 0.224, 0.225)

        # Default cascade paths relative to this file
        _base = Path(__file__).resolve().parent
        self._face_cascade_path = face_cascade_path or str(
            _base / "haarcascades_models" / "haarcascade_frontalface_default.xml"
        )
        self._eye_cascade_path = eye_cascade_path or str(
            _base / "haarcascades_models" / "haarcascade_eye.xml"
        )
        self._face_detector: Optional[cv2.CascadeClassifier] = None
        self._eye_detector: Optional[cv2.CascadeClassifier] = None

    def _get_detectors(
        self,
    ) -> Tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
        if self._face_detector is None:
            self._face_detector = cv2.CascadeClassifier(self._face_cascade_path)
        if self._eye_detector is None:
            self._eye_detector = cv2.CascadeClassifier(self._eye_cascade_path)
        return self._face_detector, self._eye_detector

    def facial_alignment_from_array(
        self,
        img: np.ndarray,
        face_detector: Optional[cv2.CascadeClassifier] = None,
        eye_detector: Optional[cv2.CascadeClassifier] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
        """
        Perform facial alignment on a numpy array image (BGR format).
        Detects face and eyes, rotates so eyes are horizontal, returns aligned face crop.

        Args:
            img: Numpy array image in BGR format (e.g. from cv2.imread or camera).
            face_detector: CascadeClassifier for face detection (uses internal default if None).
            eye_detector: CascadeClassifier for eye detection (uses internal default if None).

        Returns:
            (new_img, face_bbox, left_eye_abs, right_eye_abs):
            - new_img: Rotated, aligned face as BGR numpy array, or None if detection failed.
            - face_bbox: (x, y, w, h) in original image, or None.
            - left_eye_abs / right_eye_abs: (x, y, w, h) in original image, or None.
        """
        fd, ed = face_detector, eye_detector
        if fd is None or ed is None:
            fd, ed = self._get_detectors()

        # ------ Face detection ------
        faces = fd.detectMultiScale(img, 1.3, 5)
        if len(faces) == 0:
            return None, None, None, None

        img_height, img_width = img.shape[:2]

        # ------ Face bounding box (expand equally on all four edges) ------
        face_x, face_y, face_w, face_h = faces[0]
        expand_w = face_w * self.face_box_expand
        expand_h = face_h * self.face_box_expand
        face_x = int(face_x - expand_w)
        face_y = int(face_y - expand_h)
        face_w = int(face_w + 2 * expand_w)
        face_h = int(face_h + 2 * expand_h)
        # Clamp to image bounds
        face_x = max(0, face_x)
        face_y = max(0, face_y)
        if face_x + face_w > img_width:
            face_w = img_width - face_x
        if face_y + face_h > img_height:
            face_h = img_height - face_y
        face_bbox = (face_x, face_y, face_w, face_h)

        # ------ Face region ------
        img_face = img[int(face_y) : int(face_y + face_h), int(face_x) : int(face_x + face_w)]

        # ------ Eye detection ------
        img_gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
        eyes = ed.detectMultiScale(img_gray)

        # Sort eyes by area (width * height) in descending order and keep only the two largest
        if len(eyes) > 2:
            # Calculate area for each eye and sort by area (largest first)
            eyes_with_area = [(eye, eye[2] * eye[3]) for eye in eyes]   # (x, y, w, h) -> area = w * h
            eyes_with_area.sort(key=lambda x: x[1], reverse=True)        # Sort by area, descending
            eyes = [eye for eye, _ in eyes_with_area[:2]]                  # Keep only the two largest

        # Ensure we have at least 2 eyes, otherwise raise an error
        if len(eyes) < 2:
            return None, None, None, None

        # ------ Left and right eye ------
        eye_1 = (eyes[0][0], eyes[0][1], eyes[0][2], eyes[0][3])
        eye_2 = (eyes[1][0], eyes[1][1], eyes[1][2], eyes[1][3])
        if eye_1[0] < eye_2[0]:
            left_eye, right_eye = eye_1, eye_2
        else:
            left_eye, right_eye = eye_2, eye_1

        # Reject if the two eye boxes overlap
        lx1, ly1, lw, lh = left_eye[0], left_eye[1], left_eye[2], left_eye[3]
        rx1, ry1, rw, rh = right_eye[0], right_eye[1], right_eye[2], right_eye[3]
        overlap_x = max(0, min(lx1 + lw, rx1 + rw) - max(lx1, rx1))
        overlap_y = max(0, min(ly1 + lh, ry1 + rh) - max(ly1, ry1))
        if overlap_x > 0 and overlap_y > 0:
            return None, None, None, None

        # ------ Left and right eye center ------
        left_eye_center = (
            int(left_eye[0] + (left_eye[2] / 2)),
            int(left_eye[1] + (left_eye[3] / 2)),
        )
        right_eye_center = (
            int(right_eye[0] + (right_eye[2] / 2)),
            int(right_eye[1] + (right_eye[3] / 2)),
        )
        left_eye_x, left_eye_y = left_eye_center[0], left_eye_center[1]
        right_eye_x, right_eye_y = right_eye_center[0], right_eye_center[1]

        # ------ Determine direction of rotation ------
        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1 # rotate clockwise
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1 # rotate counterclockwise

        # ------ Trigonometry to calculate angle ------
        a = _euclidean_distance(left_eye_center, point_3rd)
        b = _euclidean_distance(right_eye_center, left_eye_center)
        c = _euclidean_distance(right_eye_center, point_3rd)
        # Reject degenerate case: coincident eyes or point_3rd same as eye (would divide by zero)
        if b < 1e-6 or c < 1e-6:
            return None, None, None, None
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(np.clip(cos_a, -1.0, 1.0))
        angle_deg = (angle * 180) / math.pi
        if direction == -1:
            angle_deg = 90 - angle_deg

        # ------ Calculate rotation angle in radians for expansion calculation ------
        angle_rad = math.radians(angle_deg)

        # Calculate expansion needed to avoid black corners after rotation
        # When rotating a rectangle, the bounding box dimensions become:
        # new_width = width * |cos(θ)| + height * |sin(θ)|
        # new_height = width * |sin(θ)| + height * |cos(θ)|
        # We need to expand the input so that after rotation, we can crop back to original size
        cos_angle = abs(math.cos(angle_rad))
        sin_angle = abs(math.sin(angle_rad))

        # Calculate how much larger the bounding box will be after rotation
        rotated_w = face_w * cos_angle + face_h * sin_angle
        rotated_h = face_w * sin_angle + face_h * cos_angle

        # Calculate padding needed (extra space on each side)
        pad_w = (rotated_w - face_w) / 2
        pad_h = (rotated_h - face_h) / 2

        # Add some extra padding for safety (10% margin)
        pad_w = int(pad_w * 1.1)
        pad_h = int(pad_h * 1.1)

        # Expand the bounding box with bounds checking (for rotation padding)
        # Calculate expanded region coordinates
        expanded_x = max(0, face_x - pad_w)
        expanded_y = max(0, face_y - pad_h)
        expanded_w = min(img_width - expanded_x, face_w + 2 * pad_w)
        expanded_h = min(img_height - expanded_y, face_h + 2 * pad_h)

        # Adjust if we hit the image boundaries
        if expanded_x + expanded_w > img_width:
            expanded_w = img_width - expanded_x
        if expanded_y + expanded_h > img_height:
            expanded_h = img_height - expanded_y

        # Crop the expanded region
        img_expanded = img[
            int(expanded_y) : int(expanded_y + expanded_h),
            int(expanded_x) : int(expanded_x + expanded_w),
        ]

        # Rotate the expanded region
        img_expanded_pil = Image.fromarray(img_expanded)
        img_rotated = np.array(img_expanded_pil.rotate(direction * angle_deg))

        # Calculate the center of the rotated image and crop back to original face size
        # The center of the rotated expanded region corresponds to the center of the original face
        rot_h, rot_w = img_rotated.shape[:2]
        center_x, center_y = rot_w // 2, rot_h // 2

        # Crop from center to get original face size
        crop_x = center_x - face_w // 2
        crop_y = center_y - face_h // 2

        # Ensure we don't go out of bounds
        crop_x = max(0, min(crop_x, rot_w - face_w))
        crop_y = max(0, min(crop_y, rot_h - face_h))
        new_img = img_rotated[crop_y : crop_y + face_h, crop_x : crop_x + face_w]

        # Ensure we got the right size (in case of boundary issues)
        if new_img.shape[0] != face_h or new_img.shape[1] != face_w:
            new_img = cv2.resize(new_img, (face_w, face_h), interpolation=cv2.INTER_LINEAR)

        # Convert eye coordinates from face-relative to absolute coordinates in original image
        left_eye_abs = (face_x + left_eye[0], face_y + left_eye[1], left_eye[2], left_eye[3])
        right_eye_abs = (face_x + right_eye[0], face_y + right_eye[1], right_eye[2], right_eye[3])

        # Return the rotated, aligned face, face bounding box, and eye coordinates for facial recognition
        return new_img, face_bbox, left_eye_abs, right_eye_abs

    def _array_to_tensor(self, img_bgr: np.ndarray) -> torch.Tensor:
        """Convert BGR numpy array to normalized tensor [1, 3, image_size, image_size]."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        x = self._transform(pil).unsqueeze(0)
        if self.normalize:
            x = TF.normalize(x, self._mean, self._std)
        return x

    def __call__(self, image_path: str, align: bool = True) -> torch.Tensor:
        """
        Full pipeline from path: open image -> [optional alignment] -> resize, to tensor, normalize.
        If align=True: run facial alignment first; raises ValueError if alignment fails.
        If align=False: skip alignment and resize the full image (e.g. for training on pre-cropped faces).
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        if align:
            aligned, _, _, _ = self.facial_alignment_from_array(img_bgr)
            if aligned is None:
                raise ValueError(
                    f"Face alignment failed for {image_path}: no face or fewer than two eyes detected."
                )
            return self._array_to_tensor(aligned)
        return self._array_to_tensor(img_bgr)

    def preprocess_from_array(
        self,
        img_bgr: np.ndarray,
        face_detector: Optional[cv2.CascadeClassifier] = None,
        eye_detector: Optional[cv2.CascadeClassifier] = None,
    ) -> torch.Tensor:
        """
        Same pipeline as __call__ but for a BGR numpy array (e.g. camera frame):
        facial alignment -> resize, to tensor, normalize.
        Raises ValueError if alignment fails (no face or fewer than two eyes detected).
        """
        aligned, _, _, _ = self.facial_alignment_from_array(
            img_bgr, face_detector, eye_detector
        )
        if aligned is None:
            raise ValueError(
                "Face alignment failed on frame: no face or fewer than two eyes detected."
            )
        return self._array_to_tensor(aligned)

    def capture_aligned_face_frames(
        self,
        camera_index: int = 0,
        stability_frames: int = 5,
        position_threshold: float = 0.15,
    ) -> Generator[
        Tuple[
            np.ndarray,
            Optional[np.ndarray],
            Optional[Tuple[int, int, int, int]],
            Optional[Tuple[int, int, int, int]],
            Optional[Tuple[int, int, int, int]],
            int,
        ],
        None,
        None,
    ]:
        """
        Generator that captures camera frames and yields (frame, aligned_face, face_bbox,
        left_eye_abs, right_eye_abs, stable_frame_count) each time. Caller can use this
        for UI or to run until stability. Stops when stable_frame_count >= stability_frames.
        """
        face_detector, eye_detector = self._get_detectors()
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return

        previous_face_center = None
        previous_left_eye_center = None
        previous_right_eye_center = None
        stable_frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                aligned_face, face_bbox, left_eye_abs, right_eye_abs = self.facial_alignment_from_array(
                    frame, face_detector, eye_detector
                )

                if aligned_face is not None and face_bbox is not None:
                    face_x, face_y, face_w, face_h = face_bbox
                    current_face_center = (face_x + face_w / 2, face_y + face_h / 2)
                    face_size_avg = (face_w + face_h) / 2
                    threshold_distance = face_size_avg * position_threshold

                    current_left_eye_center = (
                        left_eye_abs[0] + left_eye_abs[2] / 2,
                        left_eye_abs[1] + left_eye_abs[3] / 2,
                    )
                    current_right_eye_center = (
                        right_eye_abs[0] + right_eye_abs[2] / 2,
                        right_eye_abs[1] + right_eye_abs[3] / 2,
                    )

                    if previous_face_center is not None:
                        center_distance = _euclidean_distance(
                            current_face_center, previous_face_center
                        )
                        left_eye_distance = _euclidean_distance(
                            current_left_eye_center, previous_left_eye_center
                        )
                        right_eye_distance = _euclidean_distance(
                            current_right_eye_center, previous_right_eye_center
                        )
                        if (
                            center_distance <= threshold_distance
                            and left_eye_distance <= threshold_distance
                            and right_eye_distance <= threshold_distance
                        ):
                            stable_frame_count += 1
                        else:
                            stable_frame_count = 1
                            previous_face_center = current_face_center
                            previous_left_eye_center = current_left_eye_center
                            previous_right_eye_center = current_right_eye_center
                    else:
                        previous_face_center = current_face_center
                        previous_left_eye_center = current_left_eye_center
                        previous_right_eye_center = current_right_eye_center
                        stable_frame_count = 1

                    yield frame, aligned_face, face_bbox, left_eye_abs, right_eye_abs, stable_frame_count
                    if stable_frame_count >= stability_frames:
                        break
                else:
                    stable_frame_count = 0
                    previous_face_center = None
                    previous_left_eye_center = None
                    previous_right_eye_center = None
                    yield frame, None, None, None, None, 0
        finally:
            cap.release()

    def capture_aligned_face_from_camera(
        self,
        camera_index: int = 0,
        stability_frames: int = 5,
        position_threshold: float = 0.15,
    ) -> Optional[np.ndarray]:
        """
        Capture video from camera and return when a detected face has been stable
        for the required number of consecutive frames. No UI or display.

        Args:
            camera_index: Index of the camera to use (default: 0).
            stability_frames: Number of consecutive frames the face must be stable (default: 5).
            position_threshold: Position stability as fraction of face size (default: 0.15 = 15%).

        Returns:
            Aligned face as BGR numpy array, or None if camera fails or no stable face detected.
        """
        last_aligned_face = None
        for _frame, aligned_face, _bbox, _le, _re, count in self.capture_aligned_face_frames(
            camera_index, stability_frames, position_threshold
        ):
            if aligned_face is not None:
                last_aligned_face = aligned_face
            if count >= stability_frames:
                break
        return last_aligned_face
