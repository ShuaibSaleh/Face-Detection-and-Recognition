import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import csv




def detect_faces(source: np.ndarray, scale_factor: float = 1.1, min_size: int = 50) -> list:

    src = np.copy(source)
    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        pass

    # set project directory to python system path
    repo_root = os.path.dirname(os.getcwd())
    sys.path.append(repo_root)

    cascade_path = "haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier(cascade_path)

    # face detection in the image
    faces = face_cascade.detectMultiScale(
        image=src,
        scaleFactor=scale_factor,
        minNeighbors=5,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return faces


# rectangle drawing for each detected face
def draw_faces(source: np.ndarray, faces: list, thickness: int = 10) -> np.ndarray:

    src = np.copy(source)

    # rectangle drawing around the face
    for (x, y, w, h) in faces:
        cv2.rectangle(img=src, pt1=(x, y), pt2=(x + w, y + h),
                      color=(0, 255, 0), thickness=thickness)

    return src



# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in an image and return the cropped face images
def crop_face(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    
    # Crop and store the detected faces
    cropped_face=[]
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        cropped_face.append(face)
    
    return cropped_face

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

def ImageRecognition(path,mean_face,principal_components):

    # Read the image 
    inputimage = cv2.imread(path)
    # Crop the face and save it
    inputimage = cv2.resize(inputimage,(300,300))
    cropped_face = crop_face(inputimage)
    cropped_face = np.array(cropped_face[0])
    cv2.imwrite(f'static/images/input/cropped_face.jpg', cropped_face)

    # Read the cropped face image in gray scale
    cropped_face_gray = cv2.imread(f'static/images/input/cropped_face.jpg',0)

    # Resize the image
    cropped_face_gray = cv2.resize(cropped_face_gray,(100,100))


    # Convert the image to a NumPy array
    image_array = np.array(cropped_face_gray)

    # Reshape the 2D images into 1D feature vectors
    image_array = image_array.reshape(1, -1)

    # Normalize the feature vectors
    difference_face = image_array - mean_face

    # Project the training data onto the principal components
    image_array_pca = np.dot(difference_face, principal_components)

    # Load the saved KNN model
    knn_model = load('faceRecognition')

    # Predict the label for the input image
    prediction=knn_model.predict(image_array_pca)

    return prediction
