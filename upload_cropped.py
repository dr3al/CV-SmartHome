from requests import post, get
from config import CV_Config

settings = CV_Config()

images = [f"cropped_image_{i}.jpg" for i in range(62)]
images_dict = {f"photo{i}.jpg": open(name, "rb").read() for i, name in enumerate(images)}

data = {"username": "Aram"}
response = post("http://127.0.0.1:7777/users/settings/upload", data=data, files=images_dict, headers=
{"authorization": f"Bearer {settings.secret_token}"})

print(response.json())