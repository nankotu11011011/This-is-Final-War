import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.dirname(script_dir)


def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img, axis=0)

def detect_and_crop_faces(image_path, model, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        print(f"エラー: 画像の読み込みに失敗しました: {image_path}")
        return

    # 顔検出器の初期化
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # グレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 顔の検出
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("顔が検出されませんでした。")
        return
    
    print(f"{len(faces)}個の顔が検出されました。")
    
    for i, (x, y, w, h) in enumerate(faces):
        face = img[y:y+h, x:x+w]
        face_preprocessed = preprocess_image(face)
        prediction = model.predict(face_preprocessed)[0][0]
        
        print(f"顔 {i+1} である確率: {prediction:.2f}")
        
        if prediction > 0.5:  # 顔である確率が50%以上の場合のみ保存
            # 切り抜いた顔を保存
            output_path = os.path.join(output_dir, f'cropped_face_{i+1}.jpg')
            cv2.imwrite(output_path, face)
            print(f"切り抜いた顔 {i+1} を保存しました: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python predict.py <画像のパス>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = os.path.join(project_root,'models', 'face_detection_model.h5')
    
    # 出力ディレクトリの作成
    output_dir = os.path.join('output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model = load_model(model_path)
    
    detect_and_crop_faces(image_path, model, output_dir)