from configparser import ConfigParser


class CV_Config(ConfigParser):
    def __init__(self):
        super().__init__()

        self.CONFIG_PATH = "/Users/bizy1/PycharmProjects/CV-SmartHome/config.ini"
        self.read("config.ini")

        # Models config
        self.face_recognition_path = self.get("MODELS", "FACE_RECOGNITION_PATH")
        self.face_detection_paht = self.get("MODELS", "FACE_DETECTION_PATH")

        # Server config
        self.server_port = self.get("SERVER", "SERVER_PORT")
        self.uploads_path = self.get("SERVER", "UPLOADS_PATH")

        # Recognition config
        self.threshold = self.get("RECOGNITION", "THRESHOLD")
        self.neighbours = self.get("RECOGNITION", "NEIGHBOURS")

        # Client config
        self.crop_size_x = self.get("CLIENT", "CROP_SIZE_X")
        self.crop_size_y = self.get("CLIENT", "CROP_SIZE_Y")
