from configparser import ConfigParser


class CV_Config(ConfigParser):
    def __init__(self):
        super().__init__()

        self.CONFIG_PATH = "/Users/bizy1/PycharmProjects/CV-SmartHome/config.ini"
        self.read(self.CONFIG_PATH)

        # Models config
        self.face_recognition_path = self.get("MODELS", "FACE_RECOGNITION_PATH")
        self.face_detection_path = self.get("MODELS", "FACE_DETECTION_PATH")

        # Server config
        self.server_host = self.get("SERVER", "SERVER_HOST")
        self.server_port = self.get("SERVER", "SERVER_PORT")
        self.uploads_path = self.get("SERVER", "UPLOADS_PATH")
        self.secret_token = self.get("SERVER", "SECRET_TOKEN")
        self.users_database = self.get("SERVER", "USERS_DATABASE")
        self.faiss_database = self.get("SERVER", "FAISS_DATABASE")

        # Recognition config
        self.threshold = self.get("RECOGNITION", "THRESHOLD")
        self.neighbours = self.get("RECOGNITION", "NEIGHBOURS")

        # Client config
        self.crop_size_x = self.get("CLIENT", "CROP_SIZE_X")
        self.crop_size_y = self.get("CLIENT", "CROP_SIZE_Y")
