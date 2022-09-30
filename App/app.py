import io
from operator import truediv
import os
import json
from PIL import Image

import torch
from flask import Flask, jsonify, url_for, render_template, request, redirect

app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# finds the model inside your directory automatically - works only if there is one model
def find_model():
    for f  in os.listdir():
        if f.endswith(".pt"):
            return f
    print("please place a model file in this directory!")

model_name = find_model()
print ("Model name:",model_name)
model = torch.hub.load("WongKinYiu/yolov7", 'custom',model_name)
model.conf = 0.25  # confidence threshold (0-1)
model.iou = 0.45
model.eval()

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images
# Inference
    results = model(imgs, size=640)  # includes NMS
    return results

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return

        non_reclyable = ['foam', 'plastic', 'other']
        recyclable = ['glass', 'metal', 'paper', 'carton']

        msg = ""
        img_bytes = file.read()
        results = get_prediction(img_bytes)
        results.save(save_dir='static')
        filename = 'image0.jpg'
        results_df = results.pandas().xyxy
        if results_df is not None:
            pred_name = results_df[0].name
            for waste in pred_name:
                if waste in recyclable:
                    msg += "Detected waste contains <mark class=\"green\"> " + str(waste) + ", put this in recycle bin. </mark>\n"
                if waste in non_reclyable:
                    msg += "Detected waste contains <mark class=\"red\"> " + str(waste) + ", do not put this in recycle bin. </mark>\n"
        msg = msg.replace('\n', '<br>')
        return render_template('result.html',result_image = filename, model_name = model_name, msg = msg)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)
