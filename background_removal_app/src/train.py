import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.dirname(script_dir)

def load_images_and_labels(face_directory, non_face_directory):
    images = []
    labels = []
    
    # 顔画像の読み込み
    for filename in os.listdir(face_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(face_directory, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(1)  # 顔画像のラベルは1
    
    # 非顔画像の読み込み
    for filename in os.listdir(non_face_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(non_face_directory, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(0)  # 非顔画像のラベルは0
    
    return np.array(images), np.array(labels)

def create_or_load_model(model_path):
    if os.path.exists(model_path):
        print("既存のモデルを読み込みます。")
        model = load_model(model_path)
    else:
        print("新しいモデルを作成します。")
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=output)
    
    # ファインチューニングのために一部の層を解凍
    for layer in model.layers[-4:]:  # 最後の4層を訓練可能にする
        layer.trainable = True
    
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='training')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='training')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    print("学習の進捗グラフが training_history.png として保存されました。")

# メインの実行部分
if __name__ == "__main__":
    # データの読み込みと前処理
    face_dir = os.path.join(project_root, 'data', 'faces')
    non_face_dir = os.path.join(project_root, 'data', 'non_faces')
    print(f"顔画像ディレクトリ: {face_dir}")
    print(f"非顔画像ディレクトリ: {non_face_dir}")
    
    X, y = load_images_and_labels(face_dir, non_face_dir)

    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # モデルの作成または読み込み
    model_path = os.path.join(project_root, 'models', 'face_detection_model.h5')
    model = create_or_load_model(model_path)

    # モデルのコンパイルと学習
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

    # 学習履歴の可視化
    plot_training_history(history)

    # 更新されたモデルの保存
    model.save(model_path, include_optimizer=False)

    print(f"モデルの学習が完了し、{model_path}に保存されました。")

    # モデルの評価
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"テストデータでの損失: {loss:.4f}")
    print(f"テストデータでの精度: {accuracy:.4f}")