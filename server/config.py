from configparser import ConfigParser
from os import path


class CV_Config(ConfigParser):
    def __init__(self):
        super().__init__()

        self.CONFIG_PATH = path.join(path.dirname(__file__), "config.ini")
        self.read(self.CONFIG_PATH)

        # Models config
        self.face_recognition_path = path.join(path.dirname(__file__), "models")

        # Server config
        self.server_host = self.get("SERVER", "SERVER_HOST")
        self.server_port = int(self.get("SERVER", "SERVER_PORT"))
        self.secret_token = self.get("SERVER", "SECRET_TOKEN")

        self.uploads_path = path.join(path.dirname(__file__), "uploads")
        self.users_database = path.join(path.dirname(__file__), "users_database.sqlite")
        self.faiss_database = path.join(path.dirname(__file__), "users.index")

        # Recognition config
        self.threshold = float(self.get("RECOGNITION", "THRESHOLD"))
        self.neighbours = int(self.get("RECOGNITION", "NEIGHBOURS"))
