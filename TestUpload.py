import cv2
# import face_recognition
# import dlib
from threading import Thread, main_thread
from time import sleep
from config import CV_Config
from requests import post
import numpy as np

settings = CV_Config()
server = "http://127.0.0.1:7777/users/recognize"


class Worker(Thread):
    def __init__(self):
        super().__init__()

        self.name = "Worker-1"
        self.frame = None
        self.face_locations = []
        self.persons = []
        self.facenet = cv2.CascadeClassifier("/Users/bizy1/PycharmProjects/CV-SmartHome/client/haarcascade_frontalface_alt.xml")

    def run(self):
        while main_thread().is_alive():
            if self.frame is not None:
                # rgb_frame = frame[:, :, :1]

                # self.face_locations = face_recognition.face_locations(rgb_frame)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.face_locations = self.facenet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                for (i), (x, y, w, h) in enumerate(self.face_locations):
                    cropped_image = frame[x:x+w, y:y+h]
                    cv2.imwrite(f"cropped_image_{i}.jpg", cropped_image)
                    response = post(server, files={"photo1.jpg": open(f"cropped_image_{i}.jpg", "rb").read()},
                                    headers={"authorization": f"Bearer {settings.secret_token}"})

                    print(response.json())

                    if response.json()["status"] == "bad":
                        continue

                    if response.json()["response"]["identity"] is None:
                        pass
                        # print(f"Person Number {i}")

                    else:
                        first_name = response.json()["response"]["identity"]["first_name"]
                        last_name = response.json()["response"]["identity"]["last_name"]
                        distance = response.json()["response"]["identity"]["distance"]

                        # print(f"{first_name} {last_name}: {distance}")

                        if self.persons:
                            try:
                                self.persons[i] = [first_name, last_name, distance]

                            except:
                                self.persons.append([first_name, last_name, distance])

                            else:
                                pass

                        else:
                            self.persons.append([first_name, last_name, distance])

                        if i + 1 == len(self.face_locations):
                            self.persons = self.persons[:i + 1]

                # print("Persons: ", self.persons)

                self.frame = None

            sleep(.00001)


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
worker = Worker()
worker.start()
crop_image = 0

while video_capture.isOpened():
    ret, frame = video_capture.read()
    # frame = cv2.resize(frame, (640, 1214))
    frame = cv2.flip(frame, 1)
    # frame = increase_brightness(frame)

    view_image = frame
    process_image = frame

    worker.frame = process_image

    # for (i), (top, right, bottom, left) in enumerate(worker.face_locations):
    #     cropped_image = frame[top:bottom, left:right]
    #     cv2.imwrite(f"cropped_image_{crop_image}.jpg", cropped_image)
    #     crop_image += 1

    for (i), (x, y, w, h) in enumerate(worker.face_locations):

        if not worker.persons:
            view_image = prettify(view_image, f"Person Number {i}", x, x+w, y, y+h, (255, 128, 0))

        else:
            first_name = worker.persons[0][0]
            last_name = worker.persons[0][1]
            distance = worker.persons[0][2]

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
