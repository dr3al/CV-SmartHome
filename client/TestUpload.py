from enum import Enum
import cv2
from threading import Thread, main_thread
from time import sleep
from config import CV_Config
from requests import get, post
from os import path

settings = CV_Config()
server_uri = "89.248.193.55:7778"
ping_method = f"http://{server_uri}/"
register_method = f"http://{server_uri}/users/add"
check_method = f"http://{server_uri}/users/get"
add_photos_method = f"http://{server_uri}/users/settings/upload"
recognize_method = f"http://{server_uri}/users/recognize"
headers = {"authorization": f"Bearer {settings.secret_token}"}
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

        self.working = False

    def kill(self):
        self.working = False
        return None

    def renew(self):
        self.working = True
        return None

    def landmark_worker(self):
        while main_thread().is_alive():
            if not self.working:
                sleep(0.00001)
                continue

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
            if not self.working:
                sleep(0.00001)
                continue

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

                try:
                    response = post(recognize_method, files={"photo1.jpg": file}, headers=headers)

                except:
                    sleep(1)
                    continue

                # print(response.json())

                try:
                    response.json()

                except:
                    sleep(1)
                    continue

                if response.json()["status"] == "bad":
                    continue

                if response.json()["response"]["identity"] is None:
                    first_name = "~"
                    last_name = "Unknown"
                    distance = -1

                else:
                    first_name = response.json()["response"]["identity"]["first_name"]
                    last_name = response.json()["response"]["identity"]["last_name"]
                    distance = response.json()["response"]["identity"]["distance"]

                try:
                    persons[i] = [first_name, last_name, distance]

                except:
                    persons.append([first_name, last_name, distance])

                else:
                    pass

                    # if i + 1 == len(face_loc):
                    #     persons = persons[:i + 1]

            sleep(.00001)

    def run(self):
        if self.w_type == WorkerType.CONNECT_WORKER:
            self.connect_worker()

        elif self.w_type == WorkerType.LANDMARK_WORKER:
            self.landmark_worker()

        else:
            raise ValueError("Не указан тип воркера. Проверьте возможные типы в @class WorkerType")


class Capture(Thread):
    def __init__(self, connect_w: Worker, landmark_w: Worker):
        super().__init__()

        self.name = "Capture"
        self.connect_w = connect_w
        self.landmark_w = landmark_w

        self.working = True

        self.video_capture = cv2.VideoCapture(0)

    def prettify(self, img, label, x1, x2, y1, y2, color):
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        (w, h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        img = cv2.putText(img, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return img

    def kill(self):
        self.working = False
        return None

    def restart_landmark(self):
        self.landmark_w.kill()
        self.landmark_w.renew()

    def stop_landmark(self):
        self.landmark_w.kill()

    def restart_connect(self):
        self.connect_w.kill()
        self.connect_w.renew()

    def stop_connect(self):
        self.connect_w.kill()

    def clear_landmarks(self):
        self.landmark_w.face_locations = []
        return True

    def run(self):
        while self.video_capture.isOpened() and self.working and main_thread().is_alive():
            ret, frame = self.video_capture.read()
            frame = cv2.flip(frame, 1)

            view_image = frame.copy()
            process_image = frame.copy()

            self.landmark_w.frame = process_image

            for (i), (x, y, w, h) in enumerate(self.landmark_w.face_locations):
                if not landmark_worker.persons:
                    view_image = self.prettify(view_image, f"Person Number {i}", x, x + w, y, y + h, (255, 128, 0))

                else:
                    try:
                        local_person = self.landmark_w.persons[i]

                    except:
                        continue

                    else:
                        first_name = local_person[0]
                        last_name = local_person[1]
                        distance = local_person[2]

                        view_image = self.prettify(view_image, f"{first_name} {last_name}: {distance}", x, x + w, y, y + h,
                                              (255, 128, 0))

            cv2.imshow('Video', view_image)
            if cv2.waitKey(1) == ord('q'):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()


landmark_worker = Worker(WorkerType.LANDMARK_WORKER)
connect_worker = Worker(WorkerType.CONNECT_WORKER, root=landmark_worker)
capture_worker = Capture(connect_worker, landmark_worker)

capture_worker.start()
landmark_worker.start()
connect_worker.start()

while True:
    commands = ["test_mode", "register", "add_photos", "ping"]

    cmds = input("(mode) >> ")
    args = cmds.split(" ")
    try:
        cmd = args[0]

    except:
        print("No commands passed.")
        continue

    if cmd not in commands:
        print("Invalid command passed.")
        continue

    if cmd == "ping":
        print("Connecting to server...")
        try:
            response = get(ping_method, headers=headers, timeout=5).json()

        except:
            print("Server is unavailable.")
            continue

        else:
            print(response["response"]["message"])
            continue

    if cmd == "register":
        print("Connecting to server...")
        try:
            response = get(ping_method, headers=headers, timeout=5).json()

        except:
            print("Server is unavailable.")
            continue

        else:
            print(response["response"]["message"])

        username = input("(Enter username) >> ")
        first_name = input("(Enter First Name) >> ")
        last_name = input("(Enter Last Name) >> ")

        data = {"first_name": first_name, "last_name": last_name, "username": username}
        response = post(register_method, data=data, headers=headers).json()

        if response["status"] == "bad":
            print("User is already exists.")
            continue

        else:
            print(f"Successfully added ({username}) -> {first_name} {last_name}")
            continue

    if cmd == "add_photos":
        print("Connecting to server...")
        try:
            response = get(ping_method, headers=headers, timeout=5).json()

        except:
            print("Server is unavailable.")
            continue

        else:
            print(response["response"]["message"])

        username = input("(Enter username) >> ")
        user_data = {"username": username}
        response = get(check_method, data=user_data, headers=headers).json()

        if response["status"] == "bad":
            print(f"User with (username) -> {username} was not found.")
            continue

        else:
            first_name = response["response"]["identity"]["first_name"]
            last_name =  response["response"]["identity"]["last_name"]
            print(f"Successfully joined thread with ({username}) -> {first_name} {last_name}")

        # Restart landmark worker
        capture_worker.restart_landmark()

        print("Adding Photos. When you are ready, just click [Enter] in console. To exit mode, enter [exit]")

        mode = input("[ADD PHOTOS] ")

        while mode != "exit":
            for (i), (x, y, w, h) in enumerate(capture_worker.landmark_w.face_locations):
                try:
                    with open(f"cropped_image_{i}.jpg", "rb") as f:
                        file = f.read()

                    if not file:
                        raise AssertionError

                except:
                    continue

                data = {"username": username}
                response = post(add_photos_method, data=data, files={"photo1.jpg": file}, headers=headers)

            mode = input("[ADD PHOTOS] ")

        print("Exiting [ADD PHOTOS] mode...")
        capture_worker.stop_landmark()
        capture_worker.clear_landmarks()

        continue

    if cmd == "test_mode":
        print("Connecting to server...")
        try:
            response = get(ping_method, headers=headers, timeout=5).json()

        except:
            print("Server is unavailable.")
            continue

        else:
            print(response["response"]["message"])

        capture_worker.restart_landmark()
        capture_worker.restart_connect()

        print("Test mode. When you are ready, just click [Enter] in console. To exit mode, just press [Enter]")
        input(">> ")

        capture_worker.stop_connect()
        capture_worker.stop_landmark()
        capture_worker.clear_landmarks()

        continue
