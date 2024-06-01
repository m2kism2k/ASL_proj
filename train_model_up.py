from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow
import mlflow.tensorflow
import mlflow.keras
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import tensorflow as tf


class LossHistory(Callback):
    def __init__(self):
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        self.accuracies.append(logs['accuracy'])
        self.val_accuracies.append(logs['val_accuracy'])
        mlflow.log_metrics({
            f"train_loss_epoch_{epoch + 1}": logs['loss'],
            f"validation_loss_epoch_{epoch + 1}": logs['val_loss'],
            f"train_accuracy_epoch_{epoch + 1}": logs['accuracy'],
            f"validation_accuracy_epoch_{epoch + 1}": logs['val_accuracy']
        })
mlflow_url = "http://mlflow:5002"
# mlflow_url = "http://localhost:5002"
mlflow.set_tracking_uri(uri=mlflow_url)

def train_model(image_height, batch_size, num_epochs, num_classes, learning_rate, seed):
    #End the current MLflow run if there is an active one
    # if mlflow.active_run():
    #     mlflow.end_run()
    #
    # #Start a new nested MLflow run
    # with mlflow.start_run(nested=True):
    mlflow.start_run()

    mlflow.log_params({
            "image_height": image_height,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "num_classes": num_classes,
            "learning_rate": learning_rate,
            "seed": seed,
            "model_architecture": "ResNet50V2",
            "optimizer": "Adam"

        })


    #Set random seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load the ResNet-V2 model without the top classification layer
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(image_height, image_height, 3))
    #Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    # Add custom classification layers on top of the base model
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    #Output Layer
    predictions = Dense(27, activation='softmax')(x) # Assuming you have 27 classes


    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', #Use sparse_categorical_crossentropy
                  metrics=['accuracy'])

    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    validation_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_directory(
        "venv/Train_Alphabet",
        target_size=(image_height, image_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        "venv/Test_Alphabet",
        target_size=(image_height, image_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

   # Define Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    #Define ModelCheckpoint callback
    checkpoint_filepath = "models/best_model.keras" # Ensure the filepath ends with .keras
    model_checkpoint = ModelCheckpoint(checkpoint_filepath,monitor='val_loss', save_best_only=True)

    history = LossHistory()
    model.fit(
        train_generator,
        epochs=num_epochs,
        validation_data=validation_generator,
        callbacks=[history, early_stopping, model_checkpoint]
    )

    # Evaluate the model (you can also use a separate test set)
    test_loss, test_accuracy = model.evaluate(validation_generator, steps=len(validation_generator))
    print(f'Test loss: {test_loss:.4f}')
    print(f'Test accuracy: {test_accuracy:.4f}')

    example_input = next(train_generator)[0]

    #Add a signature to the model
    signature = mlflow.models.signature.infer_signature(example_input, model.predict(example_input))

    #Save the best model for future use
    model_path = "models/mlflow_model.h5"
    model_path1 = "models/mlflow_model.keras"
    model = tf.keras.models.load_model(checkpoint_filepath)
    save_model(model, model_path)
    save_model(model, model_path1)
    mlflow.keras.log_model(model, "models", signature=signature)

    final_metrics = model.evaluate(validation_generator, steps=len(validation_generator))
    mlflow.log_metrics({"final_test_loss": final_metrics[0], "final_test_accuracy": final_metrics[1]})

    # End the MLflow run
    mlflow.end_run()

    return model, history

