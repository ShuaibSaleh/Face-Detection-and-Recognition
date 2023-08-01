from flask import Flask, render_template, request
from flask_cors import CORS
import os
import base64  # convert from string to  bits
import json
import numpy as np
import time
import calendar
import image as img1
import functions as fn
import matplotlib.pyplot as plt
import json
import cv2
import csv


app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

CORS(app)


@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("facedetection.html")


@app.route("/facedetection", methods=["GET", "POST"])
def facedetection():
    if request.method == "POST":
        image_data = base64.b64decode(
            request.form["image_data"].split(',')[1])

        img_path = img1.saveImage(image_data, "faceDetect_img")

        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detected_faces = fn.detect_faces(source=image)
        faced_image = fn.draw_faces(source=image_rgb, faces=detected_faces)

        # print(f"found {len(detected_faces)} faces")
        # print(detected_faces)

        # cv2.imwrite('./static/images/output/output.jpg', faced_image)

        plt.imsave('./static/images/output/output.jpg', faced_image)

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)

        output_path = './static/images/output/output.jpg'

        mean_face = np.array(np.float32(fn.read_csv_file('mean_face.csv'))).reshape(1,-1)
        principal_components = np.array(np.float32(fn.read_csv_file('principal_components.csv')))
        recognized_face = fn.ImageRecognition(img_path,mean_face,principal_components)
        

        # return json.dumps({1: f'test'})
        return json.dumps({1: f'<img src="{output_path}?t={time_stamp}" id="ApplyEdges" alt="" >',
                           2: f'<p class="btn btn-success">Recognized Face For: {recognized_face[0]} </p>'})

    else:
        return render_template("facedetection.html")


if __name__ == "__main__":
    app.run(debug=True,port=5052)
