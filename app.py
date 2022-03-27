from __future__ import division, print_function
import os
from keras.models import load_model
from flask import Flask,request, render_template
from werkzeug.utils import secure_filename
import numpy as np
from keras.preprocessing import image


app = Flask(__name__)
MODEL_PATH = 'Gmodel.h5'
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    test_image = image.load_img(img_path, target_size=(256, 256))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    if result[0][0] == 1:
        return "Glaucoma"
    else:
        return "Not Glaucoma"


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        print(preds)
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)

