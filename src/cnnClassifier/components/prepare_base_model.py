from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    def get_base_model(self):
        """
        Load the EfficientNetB0 base model without top layers.
        """
        print("[INFO] Loading base EfficientNetB0 model...")
        self.model = EfficientNetB0(
            input_shape=tuple(self.config.params_image_size + [3]),
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        print("[INFO] Base model loaded successfully.")

    @staticmethod
    def _prepare_full_model(model, classes, dropout, freeze_all, freeze_till, learning_rate):
        """
        Add custom classification head and compile the model.
        """
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        x = tf.keras.layers.Dropout(dropout)(x)
        output = tf.keras.layers.Dense(units=classes, activation="softmax")(x)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=output)

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        print("[INFO] Full model compiled and ready for saving.")
        return full_model

    def update_base_model(self):
        """
        Adds classification head to base model and saves the full model.
        """
        if self.model is None:
            raise ValueError("Base model has not been loaded. Call get_base_model() first.")

        print("[INFO] Updating base model with classification head...")
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            dropout=self.config.params_dropout,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        print(f"[INFO] Saving full model to: {self.config.updated_base_model_path}")
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the model in TensorFlow SavedModel format (folder-based).
        Avoids JSON serialization issues with EagerTensors.
        """
        path = str(path)
        # Ensure path is folder (not .keras/.h5 file)
        if path.endswith(".keras") or path.endswith(".h5"):
            path = path.replace(".keras", "").replace(".h5", "")

        model.save(path)  # SavedModel format
        print(f"[INFO] Model saved successfully at {path} (SavedModel format)")
