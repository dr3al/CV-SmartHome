import typing as t

from flask import Flask, request, jsonify


class CV_Server(Flask):
    def __init__(self, import_name: str, static_url_path: t.Optional[str] = __name__,
                 static_folder: t.Optional[str] = "static", static_host: t.Optional[str] = None,
                 host_matching: bool = False, subdomain_matching: bool = False,
                 template_folder: t.Optional[str] = "templates", instance_path: t.Optional[str] = None,
                 instance_relative_config: bool = False, root_path: t.Optional[str] = None):
        super().__init__(import_name, static_url_path, static_folder, static_host, host_matching, subdomain_matching,
                         template_folder, instance_path, instance_relative_config, root_path)

    def routes(self):
        @self.route("/", methods=["GET", "POST"])
        def index():
            return "Hello, SchoolX 2022!", 200

        @self.route("/users/recognize", methods=["POST"])
        def users_recognize():
            pass

        @self.route("/users/add", methods=["POST"])
        def users_add():
            pass

        @self.route("/users/remove", methods=["POST"])
        def users_remove():
            pass

        @self.route("/users/access/enable", methods=["POST"])
        def users_access_enable():
            pass

        @self.route("/users/access/disable", methods=["POST"])
        def users_access_disable():
            pass
