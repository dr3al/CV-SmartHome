import gevent.pywsgi
from server import CV_Server
from config import CV_Config

config = CV_Config()

app_server = gevent.pywsgi.WSGIServer((config.server_host, int(config.server_port)), CV_Server(__name__))

app_server.serve_forever()
