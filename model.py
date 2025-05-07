import tensorflow as tf
from tensorflow.keras import layers
from config import MODEL_CONFIG
import numpy as np
import pandas as pd
from PIL import Image
import os
import logging
from data import RealTimeData
from sklearn.model_selection import train_test_split


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TC_Model:
    def __init__(self, weights_path='ri_model_weights.weights.h5'):
        self.model = self.build_advanced_model()
        self.weights_path = weights_path
        # Load pre-trained weights if they exist
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            logger.info(f"Loaded weights from {weights_path}")
    
    def build_advanced_model(self):
        """Builds a CNN model for RI detection"""
        inputs = tf.keras.Input(shape=MODEL_CONFIG['input_shape'])
        
        # Feature extraction
        x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2,2)(x)
        
        x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2,2)(x)
        
        x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2,2)(x)
        
        # Attention mechanism
        attention = layers.Conv2D(1, (1,1), activation='sigmoid')(x)
        x = layers.multiply([x, attention])
        
        # Classification head
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        return model
    
    def train(self, dataset_path='cyclone_ri_prediction_dataset.csv', image_dir='insat3d_raw_cyclone_ds', batch_size=32, epochs=20):
        """Train the model using the dataset and save weights"""
        logger.info("Loading dataset for training...")
        df = pd.read_csv(dataset_path)
        
        # Initialize data fetcher for synthetic images
        data_fetcher = RealTimeData()
        
        # Load images and labels
        images = []
        labels = []
        label_column = 'RI' if 'RI' in df.columns else 'is_RI'  # Handle both dataset versions
        for _, row in df.iterrows():
            img_path = os.path.join(image_dir, row['img_name'])
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((256, 256))
                images.append(np.array(img) / 255.0)
                labels.append(1.0 if row[label_column] else 0.0)
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                # Use synthetic image as fallback
                synthetic_img = data_fetcher._generate_synthetic_cyclone()
                images.append(synthetic_img)
                labels.append(1.0 if row[label_column] else 0.0)
                logger.info(f"Using synthetic image for {row['img_name']}")
        
        images = np.array(images)
        labels = np.array(labels)
        
        logger.info(f"Loaded {len(images)} images for training (including synthetic)")
        
        if len(images) == 0:
            logger.error("No images loaded for training. Cannot proceed.")
            raise ValueError("No images available for training.")
        
        # Split into train and validation (80-20 split)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(
                    self.weights_path, save_best_only=True, save_weights_only=True
                )
            ]
        )
        
        logger.info("Training completed. Weights saved.")
        return history
    
    def predict(self, image):
        """Predict RI probability for a single image"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        return float(self.model.predict(image, verbose=0)[0][0])
    
    def predict_with_location(self, image, lat, lon):
        """Predicts RI probability with location context"""
        base_prob = self.predict(image)
        
        # Adjust probability based on location
        location_factor = self._get_location_factor(lat, lon)
        adjusted_prob = (base_prob * (1 - MODEL_CONFIG['location_weight']) + 
                        location_factor * MODEL_CONFIG['location_weight'])
        
        return float(adjusted_prob)
    
    def _get_location_factor(self, lat, lon):
        """Returns location-based RI probability factor (0-1)"""
        if 65 <= lon <= 85 and 5 <= lat <= 15:  # Arabian Sea
            return 0.8
        elif 85 <= lon <= 95 and 10 <= lat <= 20:  # Bay of Bengal
            return 0.7
        return 0.3

# Test function
def test_model():
    """Test the model with a sample image from the dataset or synthetic image"""
    model = TC_Model()
    data_fetcher = RealTimeData()
    
    try:
        df = pd.read_csv('balanced_cyclone_ri_prediction_dataset.csv')
        sample_row = df.iloc[0]
        img_path = os.path.join('insat3d_raw_cyclone_ds', sample_row['img_name'])
        
        img = Image.open(img_path).convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
    except Exception as e:
        logger.warning(f"Failed to load test image {img_path}: {e}")
        img_array = data_fetcher._generate_synthetic_cyclone()
        logger.info("Using synthetic image for testing")
    
    print("Testing basic prediction...")
    prob = model.predict(img_array)
    print(f"Base probability: {prob:.2f}")
    
    print("\nTesting location-aware prediction...")
    prob_loc = model.predict_with_location(img_array, 15.0, 80.0)  # Bay of Bengal
    print(f"Location-adjusted probability: {prob_loc:.2f}")

if __name__ == "__main__":
    # Train the model
    model = TC_Model()
    model.train()
    # Test the model
    test_model()