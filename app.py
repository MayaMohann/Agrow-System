from flask import Flask, redirect, render_template, request
from markupsafe import Markup
import numpy as np
import pandas as pd
import joblib
import requests
from dotenv import load_dotenv
import os
import tensorflow as tf
import cv2
from PIL import Image

crop_recom_model = joblib.load('model/crop_model.joblib')
fert_pred=joblib.load('model/fertilizer_model.joblib')

disease_pred = tf.keras.models.load_model("D:/PROJECTS/AGROWW/Trained_model.keras")

# import pickle

# with open('model/Ensemble.joblib', 'rb') as file:
#     crop_recom_model = pickle.load(file)


app = Flask(__name__)
load_dotenv()

# Configure upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static', 'Images_to_test')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def home():
    title='Home'
    return render_template('index.html',title=title)

@ app.route('/crop-recommend')
def crop_recommend():
    title='Crop Recommendation'
    return render_template('crop_recom.html',title=title)

@app.route('/fertilizer')
def fertilizer_recom():
    title='Fertilizer'
    return render_template('fertilizer.html',title=title)


@app.route('/weather')
def weather():
    title='Weather Report'
    return render_template('weather.html',title=title)


@app.route('/disease')
def disease():
    title='Disease Prediction'
    return render_template('disease.html',title=title)


@ app.route('/crop-predict',methods=['POST'])
def crop_prediction():
    title='Crop Recommendation'

    if request.method=='POST':
        N=float(request.form['N'])
        P=float(request.form['P'])
        K=float(request.form['K'])
        temperature=float(request.form['temperature'])
        humidity=float(request.form['humidity'])
        ph=float(request.form['ph'])
        rainfall=float(request.form['rainfall'])
        prediction = crop_recom_model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
        final_predict=prediction[0]
        
        return render_template('result.html', prediction = final_predict,title=title)
        
    
@app.route('/fertilizer-predict', methods=['POST'])
def fert():
    title = 'Fertilizer Recommendation'

    if request.method == 'POST':
        try:
            Temparature = int(request.form['Temparature'])
            Humidity = int(request.form['Humidity'])
            Moisture = int(request.form['Moisture'])
            Nitrogen = int(request.form['Nitrogen'])
            Potassium = int(request.form['Potassium'])
            Phosphorous = int(request.form['Phosphorus'])
        except KeyError as e:
            return render_template('error.html', message=f'Missing form field: {e}', title=title)
        except ValueError:
            return render_template('error.html', message='Invalid form data. Please enter numeric values.', title=title)

        prediction1 = fert_pred.predict([[Temparature, Humidity, Moisture, Nitrogen, Potassium, Phosphorous]])
        final_predict = prediction1[0]

        return render_template('Fert_res.html', prediction1=final_predict, title=title)
    else:
        return render_template('error.html', message='Invalid request method. Please use POST.', title=title)



@app.route('/weather-predict', methods=['POST'])
def weather_fetch():
    title = 'Fertilizer Recommendation'

    if request.method == 'POST':
        city = request.form['city']
        country = request.form['country']
        api_key = os.getenv("WEATHER_API_KEY")
        weather_url = requests.get(f'http://api.openweathermap.org/data/2.5/weather?appid={api_key}&q={city},{country}&units=Metric')
        weather_data = weather_url.json()
        temp = round(weather_data['main']['temp'])
        humidity = weather_data['main']['humidity']
        description = weather_data['weather'][0]['description']

        return render_template('weather_result.html', temp = temp, humidity = humidity, city=city, description = description, title=title)
    else:
        return render_template('error.html', message='Invalid request method. Please use POST.', title=title)
    


@app.route('/disease_prediction', methods=['POST'])
def disease_prediction():
    title = 'Disease Prediction'
    
    if 'leaf_image' not in request.files:
        return render_template('error.html', message='No file uploaded.', title=title)
    
    file = request.files['leaf_image']
    if file.filename == '':
        return render_template('error.html', message='No file selected.', title=title)
    
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
    
        # Get the predicted class
        # Load the image (use one of the two approaches)
        # Approach 1: Using file path
        image = tf.keras.preprocessing.image.load_img(file_path, target_size=(128, 128))

        # Approach 2: Using PIL directly
        # image = Image.open(file)
        # image = image.resize((128, 128))

        # Convert image to array
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict
        prediction = disease_pred.predict(image_array)

        result_index = np.argmax(prediction)

        class_name = ['Apple___Apple_scab',
                    'Apple___Black_rot',
                    'Apple___Cedar_apple_rust',
                    'Apple___healthy',
                    'Background_without_leaves',
                    'Blueberry___healthy',
                    'Cherry___Powdery_mildew',
                    'Cherry___healthy',
                    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn___Common_rust',
                    'Corn___Northern_Leaf_Blight',
                    'Corn___healthy',
                    'Grape___Black_rot',
                    'Grape___Esca_(Black_Measles)',
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)',
                    'Peach___Bacterial_spot',
                    'Peach___healthy',
                    'Pepper,_bell___Bacterial_spot',
                    'Pepper,_bell___healthy',
                    'Potato___Early_blight',
                    'Potato___Late_blight',
                    'Potato___healthy',
                    'Raspberry___healthy',
                    'Soybean___healthy',
                    'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch',
                    'Strawberry___healthy',
                    'Tomato___Bacterial_spot',
                    'Tomato___Early_blight',
                    'Tomato___Late_blight',
                    'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot',
                    'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                    'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy']


        # Get the predicted class name
        model_prediction = class_name[result_index]
        print(result_index)

        # Return the result
        return render_template('disease_result.html', result = model_prediction, title=title)
    except Exception as e:
        return render_template('error.html', message=str(e), title=title)



if __name__=='__main__':
    app.run(debug=False)









