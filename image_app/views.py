from django.shortcuts import render
from .forms import ImageUploadForm
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
import os
from django.conf import settings
from .pydrive_2 import GoogleDriveAPI

drive = GoogleDriveAPI()

drive.download_from_file_name("models","face_detection_model.h5",".\\image_app\\models")

# モデルを読み込む
model_path = '.\\image_app\\models\\face_detection_model.h5'
model = load_model(model_path)

def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img, axis=0)

def detect_and_crop_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    cropped_faces = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_preprocessed = preprocess_image(face)
        prediction = model.predict(face_preprocessed)[0][0]
        
        if prediction > 0.5:
            _, buffer = cv2.imencode('.jpg', face)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            cropped_faces.append(base64_image)
    
    return cropped_faces

def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_np = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
            cropped_faces = detect_and_crop_faces(image_np)
            
            context = {
                'form': form,
                'cropped_faces': cropped_faces,
                'faces_count': len(cropped_faces),
                'debug_info': f"処理された画像: {image.name}, 検出された顔: {len(cropped_faces)}個"
            }
            
            if not cropped_faces:
                context['error_message'] = '顔が検出されませんでした。別の画像を試してください。'
            
            return render(request, 'index.html', context)
    else:
        form = ImageUploadForm()
    
    return render(request, 'index.html', {'form': form})