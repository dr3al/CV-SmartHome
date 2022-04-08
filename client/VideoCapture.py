import cv2
import face_recognition
import dlib
from threading import Thread, main_thread
from time import sleep


class Worker(Thread):
    def __init__(self):
        super().__init__()

        self.name = "Worker-1"
        self.frame = None
        self.face_locations = []

    def run(self):
        while main_thread().is_alive():
            if self.frame is not None:
                rgb_frame = frame[:, :, :1]

                self.face_locations = face_recognition.face_locations(rgb_frame)
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


video_capture = cv2.VideoCapture("/Users/bizy1/PycharmProjects/CV-SH/client/aram.mp4")
worker = Worker()
worker.start()
crop_image = 0

while video_capture.isOpened():
    ret, frame = video_capture.read()
    # frame = cv2.resize(frame, (640, 1214))

    view_image = frame
    process_image = frame

    worker.frame = process_image

    for (i), (top, right, bottom, left) in enumerate(worker.face_locations):
        cropped_image = frame[top:bottom, left:right]
        cv2.imwrite(f"cropped_image_{crop_image}.jpg", cropped_image)
        crop_image += 1

    for (i), (top, right, bottom, left) in enumerate(worker.face_locations):
        view_image = prettify(view_image, f"Person Number {i}", left, right, top, bottom, (255, 128, 0))

    cv2.imshow('Video', view_image)
    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
