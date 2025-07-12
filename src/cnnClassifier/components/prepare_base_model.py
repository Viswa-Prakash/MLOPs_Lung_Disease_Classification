from pathlib import Path
import tensorflow as tf
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from cnnClassifier import logger

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self) -> tf.keras.Model:
        """
        Load a base model from tf.keras.applications with specified parameters.
        """
        logger.info(f"Loading base model: {self.config.base_model_name}")

        base_model_class = getattr(tf.keras.applications, self.config.base_model_name)
        model = base_model_class(
            include_top=self.config.include_top,
            weights=self.config.weights,
            input_shape=tuple(self.config.input_shape)
        )

        model.trainable = self.config.trainable
        logger.info(f"Base model loaded with trainable = {model.trainable}")

        # Save the base model to .h5
        self._save_model_h5(self.config.base_model_path, model)
        return model

    def update_base_model(self, base_model: tf.keras.Model) -> tf.keras.Model:
        """
        Add classification head to base model and compile with params.
        """
        logger.info("Building updated model with classification head.")

        # Input layer
        inputs = tf.keras.Input(shape=tuple(self.config.input_shape))

        # Base model as feature extractor
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(self.config.dropout)(x)

        # Output head
        outputs = tf.keras.layers.Dense(
            self.config.classes,
            activation=self.config.activation,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer_l2)
        )(x)

        full_model = tf.keras.Model(inputs, outputs)

        # Optimizer
        optimizer_class = getattr(tf.keras.optimizers, self.config.optimizer)
        optimizer = optimizer_class(learning_rate=self.config.learning_rate)

        # Handle metrics
        metrics = (
            [self.config.metrics]
            if isinstance(self.config.metrics, str)
            else [str(m) for m in self.config.metrics]
        )

        # Compile model
        full_model.compile(
            optimizer=optimizer,
            loss=str(self.config.loss),
            metrics=metrics
        )

        logger.info("Updated model compiled successfully.")
        full_model.summary()

        # Save the updated model to .h5
        self._save_model_h5(self.config.updated_base_model_path, full_model)
        return full_model

    def _save_model_h5(self, path: Path, model: tf.keras.Model) -> None:
        """
        Save a model in .h5 format.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            if path.exists():
                import shutil
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)

            model.save(str(path), save_format="h5", include_optimizer=False)
            logger.info(f"Model saved successfully at: {path}")
        except Exception as e:
            logger.error(f"Error saving model at {path}: {e}")
            raise e
