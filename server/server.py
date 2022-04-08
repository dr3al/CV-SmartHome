import json
import typing as t
from flask import Flask, request, jsonify, Response, abort, request_started
from config import CV_Config
from errors import CV_Errors as CVE
from database import Users, Photos, engine
from sqlmodel import Session, select
from time import time
from re import sub, match
from datetime import datetime
from model import CV_Model as CVM
# from dlib_model import Dlib_Model as CVM
from uuid import uuid4 as uuid


class CV_Server(Flask):
    def __init__(self, import_name):
        super().__init__(import_name)
        self.secret_key = "!TOP_SECRET_KEY"

        # self.face_model = CVM("shape_predictor_68_face_landmarks_GTX.dat", "dlib_face_recognition_resnet_model_v1.dat")
        self.face_model = CVM("siamese_network_weights.h5")

        self.settings = CV_Config()
        self.routes()

    @staticmethod
    def success(response: dict, code=200, execution_time=None):
        response = {
                "status": "ok",
                "response": response
             }

        if execution_time is not None:
            response["execution_time"] = round(abs(time() - execution_time), 3)

        return Response(response=json.dumps(response),
                        status=code, content_type='application/json', direct_passthrough=True)

    @staticmethod
    def failed(error_message, error_code, arguments, code=400):
        return Response(response=json.dumps(
            {
                "status": "bad",
                "error": {
                    "error_message": error_message,
                    "error_code": error_code
                },
                "arguments": arguments
            }
        ), status=code, content_type='application/json', direct_passthrough=True)

    @staticmethod
    def data_processing():
        data = request.args.to_dict() or request.data or request.form or {}

        return data

    def before_something(self, sender, **extra):
        data = self.data_processing()

        if self.settings.secret_token:
            token_header = request.headers.get("authorization", None)
            if token_header != f"Bearer {self.settings.secret_token}":
                return abort(self.failed(CVE.NOT_AUTHORIZED.value,
                                         CVE.NOT_AUTHORIZED_CODE.value,
                                         [x for x in data.keys()]))

    def routes(self):
        request_started.connect(self.before_something, self)

        @self.errorhandler(404)
        def not_found(e):
            data = self.data_processing()

            return self.failed(CVE.NOT_FOUND.value, CVE.NOT_FOUND_CODE.value,
                               [x for x in data.keys()], CVE.NOT_FOUND_CODE.value)

        @self.errorhandler(405)
        def not_found(e):
            data = self.data_processing()

            return self.failed(CVE.METHOD_NOT_ALLOWED.value, CVE.METHOD_NOT_ALLOWED_CODE.value,
                               [x for x in data.keys()], CVE.METHOD_NOT_ALLOWED_CODE.value)

        @self.route("/", methods=["GET", "POST"])
        def index():
            return self.success({"message": "Hello, School X 2022!"})

        @self.route("/users/recognize", methods=["POST"])
        def users_recognize():
            start_t = time()

            data = self.data_processing()

            if not request.files:
                return self.failed(
                    CVE.NO_FILES_SEND.value,
                    CVE.NO_FILES_SEND_CODE.value,
                    [x for x in data.keys()],
                    CVE.NO_FILES_SEND_CODE.value
                )

            allowed_files = [x for x in request.files if match(r""".*\.jpg""", request .files[x].filename)]

            if not allowed_files:
                return self.failed(
                    CVE.NO_FILES_SEND.value,
                    CVE.NO_FILES_SEND_CODE.value,
                    [x for x in data.keys()],
                    CVE.NO_FILES_SEND_CODE.value
                )

            file = request.files[allowed_files[0]]
            new_filename = self.settings.uploads_path + f"/{str(uuid())}.jpg"
            file.save(new_filename)

            if self.face_model.check_correct_size(new_filename):
                return self.failed(
                    CVE.NO_FILES_SEND.value,
                    CVE.NO_FILES_SEND_CODE.value,
                    [x for x in data.keys()],
                    CVE.NO_FILES_SEND_CODE.value
                )

            photo_id, distance = self.face_model.recognize_vector(new_filename)

            if photo_id is None and distance is None:
                return self.success({
                    "identity": None
                }, execution_time=start_t)

            session = Session(engine)
            find_photo = select(Photos).where(Photos.faiss_id == photo_id)
            find_photo = session.exec(find_photo).one_or_none()

            find_user = select(Users).where(Users.id == find_photo.user_id)
            find_user = session.exec(find_user).one_or_none()

            response = {
                "identity": {
                    "first_name": find_user.first_name,
                    "last_name": find_user.last_name,
                    "username": find_user.username,
                    "distance": distance
                }
            }

            session.close()

            return self.success(response, execution_time=start_t)

        @self.route("/users/add", methods=["POST"])
        def users_add():
            start_t = time()

            data = self.data_processing()

            if "first_name" not in data or "last_name" not in data or "username" not in data:
                return self.failed(CVE.NOT_ENOUGH_ARGS.value,
                                   CVE.NOT_ENOUGH_ARGS_CODE.value,
                                   [x for x in data.values()],
                                   CVE.NOT_ENOUGH_ARGS_CODE.value)

            first_name = sub(r"[^а-яА-Яa-zA-Z]", "", data["first_name"])
            last_name = sub(r"[^а-яА-Яa-zA-Z]", "", data["last_name"])
            username = sub(r"[^a-zA-Z0-9]", "", data["username"])

            session = Session(engine)
            find_user = select(Users).where(Users.username == username)
            find_user = session.exec(find_user).one_or_none()

            if find_user:
                session.close()

                return self.failed(
                    CVE.ALREADY_EXISTS.value,
                    CVE.ALREADY_EXISTS_CODE.value,
                    [x for x in data.values()],
                    CVE.ALREADY_EXISTS_CODE.value
                )

            new_user = Users(username=username, first_name=first_name, last_name=last_name)
            session.add(new_user)
            session.commit()
            session.refresh(new_user)
            session.close()

            return self.success(
                {
                    "user": {
                        "id": new_user.id,
                        "username": new_user.username,
                        "first_name": new_user.first_name,
                        "last_name": new_user.last_name
                    }
                }, execution_time=start_t
            )

        @self.route("/users/remove", methods=["POST"])
        def users_remove():
            return self.success({})

        @self.route("/users/access/enable", methods=["POST"])
        def users_access_enable():
            return self.success({})

        @self.route("/users/access/disable", methods=["POST"])
        def users_access_disable():
            return self.success({})

        @self.route("/users/settings/update", methods=["POST"])
        def users_settings_update():
            return self.success({})

        @self.route("/users/get", methods=["GET"])
        def users_get():
            start_t = time()

            data = self.data_processing()

            if "username" not in data:
                return self.failed(
                    CVE.NOT_ENOUGH_ARGS.value,
                    CVE.NOT_ENOUGH_ARGS_CODE.value,
                    [],
                    CVE.NOT_ENOUGH_ARGS_CODE.value
                )

            session = Session(engine)
            find_user = select(Users).where(Users.username == data["username"])
            find_user = session.exec(find_user).one_or_none()

            print(find_user)

            if not find_user:
                session.close()

                return self.failed(
                    CVE.NOT_FOUND.value,
                    CVE.NOT_FOUND_CODE.value,
                    [x for x in data.values()],
                    CVE.NOT_FOUND_CODE.value
                )

            else:
                response = self.success({
                    "identity": {
                        "username": find_user.username,
                        "first_name": find_user.first_name,
                        "last_name": find_user.last_name
                    }
                }, execution_time=start_t)

                session.close()

                return response

        @self.route("/users/settings/upload", methods=["POST"])
        def users_settings_upload():
            start_t = time()

            data = self.data_processing()

            if "username" not in data:
                return self.failed(
                    CVE.NOT_ENOUGH_ARGS.value,
                    CVE.NOT_ENOUGH_ARGS_CODE.value,
                    [],
                    CVE.NOT_ENOUGH_ARGS_CODE.value
                )

            session = Session(engine)
            find_user = select(Users).where(Users.username == data["username"])
            find_user = session.exec(find_user).one_or_none()

            print(find_user)

            if not find_user:
                session.close()

                return self.failed(
                    CVE.NOT_FOUND.value,
                    CVE.NOT_FOUND_CODE.value,
                    [x for x in data.values()],
                    CVE.NOT_FOUND_CODE.value
                )

            if not request.files:
                return self.failed(
                    CVE.NO_FILES_SEND.value,
                    CVE.NO_FILES_SEND_CODE.value,
                    [x for x in data.keys()],
                    CVE.NO_FILES_SEND_CODE.value
                )

            allowed_files = [x for x in request.files if match(r""".*\.jpg""", request .files[x].filename)]

            if not allowed_files:
                return self.failed(
                    CVE.NO_FILES_SEND.value,
                    CVE.NO_FILES_SEND_CODE.value,
                    [x for x in data.keys()],
                    CVE.NO_FILES_SEND_CODE.value
                )

            add_photos = []

            for i, filename in enumerate(allowed_files):
                file = request.files[filename]
                new_filename = self.settings.uploads_path + f"/{str(uuid())}.jpg"
                file.save(new_filename)

                if self.face_model.check_correct_size(new_filename):
                    return self.failed(
                        CVE.NO_FILES_SEND.value,
                        CVE.NO_FILES_SEND_CODE.value,
                        [x for x in data.keys()],
                        CVE.NO_FILES_SEND_CODE.value
                    )

                faiss_id = self.face_model.add_to_storage(new_filename)

                photo = Photos(faiss_id=faiss_id, user_id=find_user.id,
                               photo_path=new_filename, uploaded_at=int(time()))

                add_photos.append(photo)

            session.add_all(add_photos)
            session.commit()
            session.close()

            self.face_model.dump_storage()

            return self.success({
                "files": allowed_files,
                "username": data["username"]
            }, execution_time=start_t)

    def run(self, *args, **kwargs) -> None:

        super().run(*args, **kwargs, threaded=True)
