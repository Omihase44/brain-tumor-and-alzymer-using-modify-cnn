import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import json
import os
from PIL import Image
import logging

class AlzheimerMulticlassClassifier:
    def __init__(self, model_path="alz_model_improved_efficient.h5", accuracy_file="models/model_accuracy.json"):
        self.model_path = model_path
        self.accuracy_file = accuracy_file
        self.model = None
        self.class_names = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
        self.target_size = (224, 224)
        self.load_model()

    def get_model_info(self):
        """Get model accuracy and metadata"""
        try:
            with open(self.accuracy_file, "r") as f:
                data = json.load(f)
                return data.get("alzheimer_multiclass", {})
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                logging.info(f"Loaded Alzheimer multiclass model from {self.model_path}")
            else:
                logging.warning(f"Model file {self.model_path} not found, creating new model")
                self.create_model()
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            self.create_model()

    def create_model(self):
        """Create EfficientNetB0 model for Alzheimer classification"""
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Add custom head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(4, activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logging.info("Created EfficientNetB0 Alzheimer multiclass classification model")

    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize image
        image = image.resize(self.target_size)

        # Convert to array and normalize
        img_array = np.array(image) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image_path):
        """Predict Alzheimer'"'"'s disease stage from image"""
        try:
            # Open and preprocess image
            image = Image.open(image_path)
            processed_image = self.preprocess_image(image)

            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]

            # Get model accuracy info
            model_info = self.get_model_info()
            model_accuracy = model_info.get("validation_accuracy", 0.0)

            result = {
                "prediction": self.class_names[predicted_class_idx],
                "confidence": float(confidence),
                "model_accuracy": model_accuracy,
                "all_probabilities": {
                    class_name: float(prob)
                    for class_name, prob in zip(self.class_names, predictions[0])
                }
            }

            return result

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return None

    def train_model(self, train_data_dir, validation_split=0.2, epochs=50, batch_size=32):
        """Train the model on the dataset"""
        try:
            # Create data generators
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=validation_split
            )

            train_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size=self.target_size,
                batch_size=batch_size,
                class_mode="sparse",
                subset="training"
            )

            validation_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size=self.target_size,
                batch_size=batch_size,
                class_mode="sparse",
                subset="validation"
            )

            # Callbacks
            early_stopping = EarlyStopping(
                monitor="val_accuracy",
                patience=10,
                restore_best_weights=True
            )

            reduce_lr = ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.2,
                patience=5,
                min_lr=1e-6
            )

            # If using transfer learning, do initial training with frozen base
            if hasattr(self.model.layers[0], 'trainable') and not self.model.layers[0].trainable:
                # Train the model initially with frozen base
                history = self.model.fit(
                    train_generator,
                    epochs=epochs//2,  # First half with frozen base
                    validation_data=validation_generator,
                    callbacks=[early_stopping, reduce_lr]
                )

                # Unfreeze some layers for fine-tuning
                base_model = self.model.layers[0]  # EfficientNetB0 base
                base_model.trainable = True
                for layer in base_model.layers[:-20]:  # Freeze all but last 20 layers
                    layer.trainable = False

                # Recompile with lower learning rate
                self.model.compile(
                    optimizer=Adam(learning_rate=1e-5),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )

                # Fine-tune
                history_ft = self.model.fit(
                    train_generator,
                    epochs=epochs//2,  # Second half fine-tuning
                    validation_data=validation_generator,
                    callbacks=[early_stopping, reduce_lr]
                )

                # Combine histories
                for key in history.history:
                    history.history[key].extend(history_ft.history[key])
                history.history['val_accuracy'].extend(history_ft.history['val_accuracy'])
                history.history['val_loss'].extend(history_ft.history['val_loss'])
            else:
                # Train normally
                history = self.model.fit(
                    train_generator,
                    epochs=epochs,
                    validation_data=validation_generator,
                    callbacks=[early_stopping, reduce_lr]
                )

            # Save the model
            self.model.save(self.model_path)

            # Evaluate and save accuracy
            val_accuracy = max(history.history["val_accuracy"])
            self.save_accuracy_info(val_accuracy)

            logging.info(f"Model trained successfully. Validation accuracy: {val_accuracy:.4f}")

            return history, val_accuracy

        except Exception as e:
            logging.error(f"Error during training: {e}")
            return None, 0.0

    def save_accuracy_info(self, validation_accuracy):
        """Save model accuracy information"""
        try:
            # Load existing data or create new
            if os.path.exists(self.accuracy_file):
                with open(self.accuracy_file, "r") as f:
                    data = json.load(f)
            else:
                data = {}

            # Update Alzheimer multiclass classifier info
            data["alzheimer_multiclass"] = {
                "validation_accuracy": float(validation_accuracy),
                "model_path": self.model_path,
                "classes": self.class_names,
                "input_shape": self.target_size + (3,),
                "last_updated": str(tf.timestamp())
            }

            # Save back to file
            with open(self.accuracy_file, "w") as f:
                json.dump(data, f, indent=4)

            logging.info(f"Saved accuracy info: {validation_accuracy:.4f}")

        except Exception as e:
            logging.error(f"Error saving accuracy info: {e}")

def main():
    """Main function for training the model"""
    import argparse

    parser = argparse.ArgumentParser(description="Train Alzheimer Multiclass Classification Model")
    parser.add_argument("--data_dir", type=str, default="dataset/alzheimer/Data",
                       help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create classifier
    classifier = AlzheimerMulticlassClassifier()

    # Train the model
    print(f"Training Alzheimer multiclass classifier on data from: {args.data_dir}")
    history, accuracy = classifier.train_model(
        args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    if history:
        print(".4f")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()
