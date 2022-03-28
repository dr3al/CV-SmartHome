import cv2
import face_recognition

video_capture = cv2.VideoCapture(0)

face_locations = []


def prettify(img, label, x1, x2, y1, y2, color):
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    (w, h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

    img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    img = cv2.putText(img, label, (x1, y1 - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img


while True:
    ret, frame = video_capture.read()
    image = frame

    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)

    for (i), (top, right, bottom, left) in enumerate(face_locations):
        image = prettify(image, f"Person Number {i}", left, top, right, bottom, (255, 128, 0))

    cv2.imshow('Video', image)
    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
