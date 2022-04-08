from requests import get, post
from config import CV_Config

secret_token = "254972abs3:24cf35bcvrnbrfcn35mjttgriei8y"

data = {"first_name": "Aram", "last_name": "Bidoev", "username": "Aram"}
response = post("http://127.0.0.1:7778/users/add", data=data, headers={"authorization": f"Bearer {secret_token}"})

# image1 = "cropped_image_43.jpg"
# image2 = "cropped_image_6.jpg"
# image3 = "cropped_image_26.jpg"
# image4 = "cropped_image_64.jpg"
# image5 = "cropped_image_49.jpg"
# data = {"username": "Ar4ikov"}
# response = post("http://127.0.0.1:7777/users/settings/upload",
#                 data=data, files={"photo1.jpg": open(image1, "rb").read(),
#                                   "photo2.jpg": open(image2, "rb").read(),
#                                   "photo3.jpg": open(image3, "rb").read(),
#                                   "photo4.jpg": open(image4, "rb").read(),
#                                   "photo5.jpg": open(image5, "rb").read()}, headers={"authorization": f"Bearer {settings.secret_token}"})


# image = "cropped_image_24.jpg"
# response = post("http://127.0.0.1:7777/users/recognize",
#                 files={"photo1.jpg": open(image, "rb").read()}, headers={"authorization": f"Bearer {settings.secret_token}"})

print(response.json())
