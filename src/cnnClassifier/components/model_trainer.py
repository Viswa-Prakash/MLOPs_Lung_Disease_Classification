import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier import logger
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.config.configuration import ConfigurationManager

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_ds = None
        self.val_ds = None

    def build_model(self):
        logger.info("Rebuilding model using PrepareBaseModel logic...")
        prepare_config = ConfigurationManager().get_prepare_base_model_config()
        prepare_model = PrepareBaseModel(config=prepare_config)
        base_model = prepare_model.get_base_model()
        self.model = prepare_model.update_base_model(base_model)
        logger.info("Model rebuilt successfully.")

    def prepare_data(self):
        logger.info("Preparing training and validation datasets...")

        data_dir = self.config.training_data

        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.config.image_height, self.config.image_width),
            batch_size=self.config.batch_size,
            label_mode="int"
        )

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.config.image_height, self.config.image_width),
            batch_size=self.config.batch_size,
            label_mode="int"
        )

        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.prefetch(buffer_size=AUTOTUNE)

        logger.info("Datasets prepared successfully.")

    def get_callbacks(self):
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.trained_model_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        )

        earlystop_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )

        return [checkpoint_cb, earlystop_cb, reduce_lr_cb]

    def train(self):
        logger.info("Started training...")
        self.prepare_data()
        self.build_model()
        callbacks = self.get_callbacks()

        self.model.fit(
            self.train_ds,
            epochs=self.config.epochs,
            validation_data=self.val_ds,
            callbacks=callbacks
        )

        logger.info(f"Best model saved at: {self.config.trained_model_path}")
        logger.info("Training completed.")
