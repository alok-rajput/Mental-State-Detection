# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

#   2. Install required libraries
!pip install librosa --quiet
!pip install resampy --quiet

#   3. Import necessary libraries
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Bidirectional, Dropout
from tensorflow.keras.utils import to_categorical

#   4. Feature extraction function
def extract_features(file_path, max_pad_len=200):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print(" Error with file:", file_path)
        print("   Exception:", e)
        return None

#   5. Test MFCC on one file
test_file = "/content/drive/MyDrive/speech/Normal/03-01-01-01-01-01-03.wav"
mfcc = extract_features(test_file)
if mfcc is not None:
    print("MFCC shape:", mfcc.shape)
else:
    print(" Could not load test file.")

#  6. Load dataset
def load_data(data_dir):
    features, labels = [], []
    folder_map = {
        "Stressed": "Stressed",
        "Depressed": "Depressed",
        "Normal": "Normal"
    }
    for label, folder_name in folder_map.items():
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f" Folder not found: {folder_path}")
            continue
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                file_path = os.path.join(folder_path, file)
                mfccs = extract_features(file_path)
                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(label)
    return np.array(features), np.array(labels)

# 7. Load data
data_dir = "/content/drive/MyDrive/speech"
print("📁 Subfolders in speech directory:", os.listdir(data_dir))
X, y = load_data(data_dir)

#  8. Check data and preprocess
if len(X) == 0 or len(y) == 0:
    print("❌ No data loaded. Please check your folder structure.")
else:
    print("Data loaded:", X.shape, y.shape)

    # Encode labels
    le = LabelEncoder()
    y_encoded = to_categorical(le.fit_transform(y))

    # Transpose to (samples, timesteps, features)
    X = np.transpose(X, (0, 2, 1))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    #  9. Build BiGRU model
    def build_bigru_model(input_shape, num_classes):
        model = Sequential([
            Bidirectional(GRU(64, return_sequences=False), input_shape=input_shape),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    model = build_bigru_model(X_train.shape[1:], y_train.shape[1])
    model.summary()

    #  10. Train model
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)

    #  11. Evaluate and plot confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    #  12. Classification report
    try:
        report = classification_report(y_true, y_pred_classes, target_names=le.classes_, output_dict=True)

        print("\n🔍 Classification Report:\n")
        for label in le.classes_:
            print(f"Class: {label}")
            print(f"  Precision: {report[label]['precision']:.4f}")
            print(f"  Recall:    {report[label]['recall']:.4f}")
            print(f"  F1 Score:  {report[label]['f1-score']:.4f}")
            print(f"  Support:   {report[label]['support']}")
            print("-" * 30)

        print("\n📊 Overall Accuracy:", report['accuracy'])
    except NameError:
        print("❌ Variables y_true, y_pred_classes, or le not defined. Please run the previous cell first.")
