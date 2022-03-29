from requests import get, post
from config import CV_Config

settings = CV_Config()
image = "cropped_image_0.jpg"

data = {"username": "Ar4ikov"}
response = post("http://127.0.0.1:7777/users/settings/upload",
                data=data, files={"photo1.jpg": open(image, "rb").read()}, headers={"authorization": f"Bearer {settings.secret_token}"})

print(response.json())