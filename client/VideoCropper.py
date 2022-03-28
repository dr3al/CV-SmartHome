from enum import Enum
from typing import Union, Callable, List
from os import path
import numpy as np
import dlib
import cv2

DLIB_MODELS_PATH = "/Users/bizy1/PycharmProjects/CV-SmartHome/dlib_models/"


def dlib_numpy_rect_converter(main_data, convert_to, additional_data=None):
    if not isinstance(main_data, convert_to):
        if convert_to == np.ndarray:
            if isinstance(main_data, dlib.full_object_detection):
                return np.array([(part.x, part.y) for part in main_data.parts()], dtype=np.float32)
            elif isinstance(main_data, dlib.rectangle):
                return np.array((main_data.left(), main_data.top(), main_data.right(), main_data.bottom()),
                                dtype=np.float32)
        elif convert_to == dlib.rectangle:
            return dlib.rectangle(*main_data)
        elif convert_to == dlib.full_object_detection:
            return dlib.full_object_detection(rect=main_data,
                                              parts=dlib.points([dlib.point(point) for point in additional_data]))
    else:
        return main_data


class Cropper:
    def __init__(self, frame: np.ndarray, detector_type: str = "dlib", landmarks_type: str = "dlib",
                 recognizer_type: str = "dlib"):

        self.frame: Union[str, int, np.ndarray] = frame
        self.faces: Union[None, List[Union[dlib.rectangle, np.ndarray]], dlib.rectangles] = None
        self.face_images: Union[None, List[np.ndarray]] = None
        self.landmarks: Union[None, List[np.ndarray]] = None

        self.detector_type: str = detector_type
        self.landmarks_type: str = landmarks_type
        self.recognizer_type: str = recognizer_type

        self.detector: Union[None, Callable] = None
        self.landmark_finder: Union[None, Callable] = None
        self.recognizer: Union[None, Callable] = None

        self.setup()

    def reset(self):
        self.faces: Union[None, List[Union[dlib.rectangle, np.ndarray]], dlib.rectangles] = None
        self.face_images: Union[None, List[np.ndarray]] = None
        self.landmarks: Union[None, List[np.ndarray]] = None

    def setup(self):
        if self.detector_type == "dlib":
            self.detector = dlib.get_frontal_face_detector()
        else:
            raise NotImplementedError

        if self.landmarks_type == "dlib":
            self.landmark_finder = dlib.shape_predictor(DLIB_MODELS_PATH + "shape_predictor_68_face_landmarks_GTX.dat")
        else:
            raise NotImplementedError

        if self.recognizer_type == "dlib":
            self.recognizer = dlib.face_recognition_model_v1(DLIB_MODELS_PATH + "dlib_face_recognition_resnet_model_v1.dat")
        else:
            raise NotImplementedError

    def detect(self):
        self.faces = self.detector(self.frame)
        if self.faces is not None:
            return True
        return False

    def find_landmarks(self):
        self.landmarks = []
        for face in self.faces:
            rect = dlib_numpy_rect_converter(face, convert_to=dlib.rectangle)
            self.landmarks.append(dlib_numpy_rect_converter(self.landmark_finder(self.frame, rect),
                                                            convert_to=np.ndarray))
        return True

    def get_normalized_faces(self):
        self.face_images = []
        for i, face in enumerate(self.faces):
            landmarks = self.landmarks[i]
            self.face_images.append(dlib.get_face_chip(self.frame,
                                                       dlib_numpy_rect_converter(self.faces[i],
                                                                                 convert_to=dlib.full_object_detection,
                                                                                 additional_data=landmarks)))

    def run(self):
        # Run all the stages from the scratch
        self.reset()
        self.detect()
        self.find_landmarks()
        self.get_normalized_faces()

    def show(self, cap: str = ""):
        for i, face in enumerate(self.faces):
            x0, y0, x1, y1 = map(int, dlib_numpy_rect_converter(face, convert_to=np.ndarray))
            print(x0, y0, x1, y1)
            cv2.imshow(f"{cap} face {i + 1}", self.frame[y0:y1, x0:x1].copy())
            cv2.waitKey(0)

    @staticmethod
    def prettify(img, label, x1, x2, y1, y2, color):
        # For bounding box
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # For the text background
        # Finds space required by the text so that we can put a background with that amount of width.
        (w, h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Prints the text.
        img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        img = cv2.putText(img, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return img

    def show_rectangles(self, cap: str = ""):
        image = self.frame.copy()

        for i, face in enumerate(self.faces):
            x0, y0, x1, y1 = map(int, dlib_numpy_rect_converter(face, convert_to=np.ndarray))
            print(x0, y0, x1, y1)
            image = self.prettify(image, f"People Number {i}", x0, x1, y0, y1, (255, 128, 0))
            # image = cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 3)

        return image

    def frameArray(self, photoPath: str = "", cap: str = ""):
        for i, face in enumerate(self.faces):
            x0, y0, x1, y1 = map(int, dlib_numpy_rect_converter(face, convert_to=np.ndarray))
            picture = cv2.imwrite(photoPath, self.frame[y0:y1, x0:x1].copy())
            return picture

    def __call__(self, frame: np.ndarray = None):
        if frame is not None:
            self.frame = frame
        self.run()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        fw = Cropper(frame)
        fw()
        image = fw.show_rectangles()

        cv2.imshow('frame', image)
        if cv2.waitKey(1) == ord('q'):
            break
