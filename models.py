"""
Model definitions for the federated learning experiments
"""

import tensorflow as tf
from tensorflow.keras import layers
from config import WIDTH, HEIGHT, CHANNELS, NUM_CLASSES


def create_compiled_keras_model():
    """
    Crée un modèle CNN compilé pour CIFAR-10

    Returns:
        model: modèle CNN compilé
    """
    model = tf.keras.models.Sequential([
        # Première couche convolutionnelle
        tf.keras.layers.Input(shape=(WIDTH, HEIGHT, CHANNELS)),
        tf.keras.layers.Conv2D(
            32, (5, 5),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Deuxième couche convolutionnelle
        tf.keras.layers.Conv2D(
            64, (5, 5),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Couches denses
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)
    ])

    # Compilation du modèle
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer="adam",
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

    return model


def create_shadow_model():
    """
    Crée un modèle shadow pour les attaques d'inférence de membership
    Architecture identique au modèle cible mais avec dropout

    Returns:
        model: modèle shadow compilé
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(WIDTH, HEIGHT, CHANNELS)),
        tf.keras.layers.Conv2D(
            32, (5, 5),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(
            64, (5, 5),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        "adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def create_attack_model():
    """
    Crée un modèle d'attaque pour prédire l'appartenance

    Returns:
        model: modèle d'attaque compilé
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            128,
            activation="relu",
            input_shape=(NUM_CLASSES,)
        ),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        "adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def create_adaboost_attack_model():
    """
    Crée un modèle d'attaque basé sur AdaBoost

    Returns:
        model: modèle AdaBoost
    """
    from sklearn.ensemble import AdaBoostClassifier

    model = AdaBoostClassifier(n_estimators=250, random_state=0)
    return model