import cv2 
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load the dataset
data = pd.read_csv('styles.csv')

# Image preprocessing functions
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        return image
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def load_images_from_folder(folder_path, data_df, max_samples=None):
    images = []
    labels = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    print(f"Loading images from {folder_path}")
    processed = 0
    
    for filename in tqdm(os.listdir(folder_path)):
        if max_samples and processed >= max_samples:
            break
            
        if filename.lower().endswith(valid_extensions):
            img_id = os.path.splitext(filename)[0]
            try:
                img_id = int(img_id)  # assuming filenames are IDs
            except:
                continue
                
            if img_id in data_df['id'].values:
                img_path = os.path.join(folder_path, filename)
                img = preprocess_image(img_path)
                if img is not None:
                    images.append(img)
                    # Get the baseColour for this image ID
                    label = data_df[data_df['id'] == img_id]['baseColour'].values[0]
                    labels.append(label)
                    processed += 1
    
    return np.array(images), np.array(labels)

# Skin tone detection functions (not used in training but useful for inference)
def detect_skin_tone(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
    skin = cv2.bitwise_and(image, image, mask=mask)
    skin_pixels = skin[np.where(mask != 0)]
    return np.mean(skin_pixels, axis=0) if len(skin_pixels) > 0 else np.array([0, 0, 0])

def get_closest_skin_tone(skin_mean):
    if len(skin_mean) == 0:
        return "Unknown"
    r, g, b = skin_mean
    if r > 200 and g > 150 and b > 100:
        return "Light"
    elif r > 150 and g > 100 and b > 80:
        return "Medium"
    else:
        return "Dark"

# Model creation
def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions)

# Main training function
def train_model():
    # Load and prepare data
    X, y = load_images_from_folder('images', data, max_samples=1000)  # Limit samples for testing
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    num_classes = len(encoder.classes_)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2)
    
    # Create and compile model
    model = create_model(num_classes)
    model.compile(optimizer=Adam(0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=40,  # Reduced for testing
        batch_size=32
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print(f'Test accuracy: {test_acc}')
    
    # Save the model
    model.save('fashion_recommender.h5')
    joblib.dump(encoder, 'label_encoder.pkl')
    print("Model saved as fashion_recommender.h5")
    
    return model, encoder

# Run the training
if __name__ == "__main__":
    model, encoder = train_model()