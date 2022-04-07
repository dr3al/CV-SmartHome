from enum import Enum

import cv2
from threading import Thread, main_thread
from time import sleep
from server.config import CV_Config
from requests import post
import numpy as np
from os import path

settings = CV_Config()
server = "http://89.248.193.55:7778/users/recognize"
cascade_model_path = path.join(path.dirname(__file__), "models", "haarcascade_frontalface_alt.xml")


class WorkerType(Enum):
    LANDMARK_WORKER = "landmark_worker"
    CONNECT_WORKER = "connect_worker"


class Worker(Thread):
    def __init__(self, w_type: WorkerType, root=None):
        super().__init__()

        self.name = "Worker"
        self.frame = None
        self.face_locations = []
        self.persons = []
        self.facenet = cv2.CascadeClassifier(cascade_model_path)
        self.w_type = w_type

        self.root = root

    def landmark_worker(self):
        while main_thread().is_alive():
            if self.frame is not None:
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                self.face_locations = self.facenet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                for (i), (x, y, w, h) in enumerate(self.face_locations):
                    cropped_image = self.frame[y:y + h, x:x + w]
                    cv2.imwrite(f"cropped_image_{i}.jpg", cropped_image)

                self.frame = None

            sleep(.00001)

    def connect_worker(self):
        while main_thread().is_alive():
            face_loc = self.root.face_locations if self.root else self.face_locations
            persons = self.root.persons if self.root else self.persons

            for (i), (x, y, w, h) in enumerate(face_loc):
                try:
                    with open(f"cropped_image_{i}.jpg", "rb") as f:
                        file = f.read()

                    if not file:
                        raise AssertionError

                except:
                    continue

                response = post(server, files={"photo1.jpg": file}, headers={"authorization": f"Bearer {settings.secret_token}"})

                print(response.json())

                if response.json()["status"] == "bad":
                    continue

                if response.json()["response"]["identity"] is None:
                    pass

                else:
                    first_name = response.json()["response"]["identity"]["first_name"]
                    last_name = response.json()["response"]["identity"]["last_name"]
                    distance = response.json()["response"]["identity"]["distance"]

                    if persons:
                        try:
                            persons[i] = [first_name, last_name, distance]

                        except:
                            persons.append([first_name, last_name, distance])

                        else:
                            pass

                    else:
                        persons.append([first_name, last_name, distance])

                    if i + 1 == len(face_loc):
                        persons = persons[:i + 1]

            sleep(.00001)

    def run(self):
        if self.w_type == WorkerType.CONNECT_WORKER:
            self.connect_worker()

        elif self.w_type == WorkerType.LANDMARK_WORKER:
            self.landmark_worker()

        else:
            raise ValueError("Не указан тип воркера. Проверьте возможные типы в @class WorkerType")


def prettify(img, label, x1, x2, y1, y2, color):
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    (w, h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

    img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    img = cv2.putText(img, label, (x1, y1 - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img


def white_balance_loops(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            l, a, b = result[x, y, :]
            # fix for CV correction
            l *= 100 / 255.0
            result[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
            result[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def increase_brightness(img, value=40):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


video_capture = cv2.VideoCapture(0)
# video_capture = cv2.VideoCapture("/Users/bizy1/PycharmProjects/CV-SmartHome/client/video2.mp4")

landmark_worker = Worker(WorkerType.LANDMARK_WORKER)
connect_worker = Worker(WorkerType.CONNECT_WORKER, root=landmark_worker)

landmark_worker.start()
connect_worker.start()

crop_image = 0

while video_capture.isOpened():
    ret, frame = video_capture.read()
    # frame = cv2.resize(frame, (640, 1214))
    frame = cv2.flip(frame, 1)
    # frame = increase_brightness(frame)

    view_image = frame
    process_image = frame.copy()

    landmark_worker.frame = process_image

    # for (i), (top, right, bottom, left) in enumerate(worker.face_locations):
    #     cropped_image = frame[top:bottom, left:right]
    #     cv2.imwrite(f"cropped_image_{crop_image}.jpg", cropped_image)
    #     crop_image += 1

    for (i), (x, y, w, h) in enumerate(landmark_worker.face_locations):

        if not landmark_worker.persons:
            view_image = prettify(view_image, f"Person Number {i}", x, x+w, y, y+h, (255, 128, 0))

        else:
            first_name = landmark_worker.persons[0][0]
            last_name = landmark_worker.persons[0][1]
            distance = landmark_worker.persons[0][2]

            view_image = prettify(view_image, f"{first_name} {last_name}: {distance}", x, x+w, y, y+h, (255, 128, 0))

        # cropped_image = cv2.imread(f"cropped_image_{i}.jpg")
        # try:
        #     view_image[0:cropped_image.shape[0], view_image.shape[1] - cropped_image.shape[1]:view_image.shape[1]] = cropped_image
        #
        # except:
        #     pass

    cv2.imshow('Video', view_image)
    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
