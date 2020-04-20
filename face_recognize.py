import cv2
import face_recognition
from urllib.request import Request
from urllib.request import urlopen
from urllib.error import HTTPError
import numpy as np
import time

def url_to_image(url):
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'}
    req = Request(url, headers=header)
    try:
        resp = urlopen(req)
    except HTTPError as err:
        if err.code == 404:
            return None
        elif err.code == 403:
            time.sleep(2)
            return None
        else:
            raise
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def face_detect(url):
    #Input: URL of picture
    #Output: cutout pictures of face
    face_list = []
    face_cascade_path = './haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    image = url_to_image(url)
    if image is None:
        return None
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray,scaleFactor=1.2, minNeighbors=2)
    shape = image.shape
    img_h,img_w = shape[0],shape[1]
    for x,y,w,h in faces:
        face = image[max(y-40,0): min(y+h+40, img_h), max(x-40,0): min(x+w+40, img_w)]
        face_list.append(face)
        
    return face_list

def face_recog(face):
    #Input: cutout picture of face
    #Output: whether it's target or not

    #load some images that contain target's face
    sample_image = face_recognition.load_image_file("sample_image/Scarlett_Johansson_1761.jpg")
    sample_image1 = face_recognition.load_image_file("sample_image/Scarlett_Johansson_1814.jpg")
    sample_image2 = face_recognition.load_image_file("sample_image/Scarlett_Johansson_1859.jpg")
    sample_image3 = face_recognition.load_image_file("sample_image/Scarlett_Johansson_1868.jpg")
    sample_image4 = face_recognition.load_image_file("sample_image/Scarlett_Johansson_1836.jpg")

    sample_image = face_recognition.face_encodings(sample_image)[0]
    sample_image1 = face_recognition.face_encodings(sample_image1)[0]
    sample_image2 = face_recognition.face_encodings(sample_image2)[0]
    sample_image3 = face_recognition.face_encodings(sample_image3)[0]
    sample_image4 = face_recognition.face_encodings(sample_image4)[0]

    try:
        unknown_image = face_recognition.face_encodings(face)[0]
    except:
        return [False]
    
    known_faces = [
        sample_image,
        sample_image1,
        sample_image2,
        sample_image3,
        sample_image4
    ]

    results = face_recognition.compare_faces(known_faces, unknown_image)
    return results
