"""
Data preparation module for federated learning experiments
"""

import numpy as np
import tensorflow as tf
from config import CLIENTS, SIZE


def get_data(source):
    """
    Divise les données en sous-ensembles pour chaque client

    Args:
        source: tuple (X, y) des données d'entraînement

    Returns:
        all_data: toutes les données combinées
        split_data: données divisées par client
        external_data: données externes restantes
    """
    # Normaliser les données d'entrée
    X_normalized = np.array(source[0][:SIZE * CLIENTS] / 255, dtype=np.float32)
    y_categorical = tf.keras.utils.to_categorical(source[1][:SIZE * CLIENTS])

    all_data = (X_normalized, y_categorical)

    # Diviser les données par client
    split_data = []
    for client_id in range(CLIENTS):
        start_idx = client_id * SIZE
        end_idx = (client_id + 1) * SIZE
        client_data = (
            all_data[0][start_idx:end_idx],
            all_data[1][start_idx:end_idx]
        )
        split_data.append(client_data)

    # Données externes (pour les attaques)
    external_X = np.array(source[0][SIZE * CLIENTS:] / 255, dtype=np.float32)
    external_y = tf.keras.utils.to_categorical(source[1][SIZE * CLIENTS:])
    external_data = (external_X, external_y)

    return all_data, split_data, external_data


def prepare_cifar10_data():
    """
    Prépare les données CIFAR-10 pour l'expérimentation

    Returns:
        cifar_train_data: données d'entraînement combinées
        cifar_train_fed_data: données d'entraînement divisées par client
        cifar_test_data: données de test
        attacker_data: données pour l'attaquant
    """
    # Charger CIFAR-10
    (cifar_train_X, cifar_train_y), (cifar_test_X, cifar_test_y) = tf.keras.datasets.cifar10.load_data()

    # Préparer les données de test
    cifar_test_data = (
        np.array(cifar_test_X / 255, dtype=np.float32),
        tf.keras.utils.to_categorical(cifar_test_y)
    )

    # Préparer les données d'entraînement
    cifar_train_data, cifar_train_fed_data, attacker_data = get_data(
        (cifar_train_X, cifar_train_y)
    )

    return cifar_train_data, cifar_train_fed_data, cifar_test_data, attacker_data


def prepare_shadow_data(test_data, train_size=5000):
    """
    Prépare les données pour l'entraînement des modèles shadow

    Args:
        test_data: données de test CIFAR-10
        train_size: taille de l'ensemble d'entraînement

    Returns:
        shadow_train_data: données d'entraînement pour les modèles shadow
        shadow_test_data: données de test pour les modèles shadow
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        test_data[0], test_data[1],
        train_size=train_size,
        test_size=len(test_data[0]) - train_size,
        random_state=42
    )

    return (X_train, y_train), (X_test, y_test)