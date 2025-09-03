"""
Centralized training implementation
"""

import os
import datetime
from models import create_compiled_keras_model
from config import EPOCHS_CENTRALIZED, BATCH_SIZE, MODEL_DIR, LOG_DIR


def train_centralized_model(train_data, test_data, save_model=True):
    """
    Entraîne un modèle de manière centralisée

    Args:
        train_data: données d'entraînement (X, y)
        test_data: données de test (X, y)
        save_model: sauvegarder le modèle ou non

    Returns:
        model: modèle entraîné
        history: historique d'entraînement
    """
    print("=== ENTRAÎNEMENT CENTRALISÉ ===")
    print(f"Données d'entraînement: {train_data[0].shape}")
    print(f"Données de test: {test_data[0].shape}")

    # Créer le modèle
    model = create_compiled_keras_model()

    # Entraîner le modèle
    history = model.fit(
        train_data[0], train_data[1],
        validation_data=test_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_CENTRALIZED,
        verbose=1
    )

    # Sauvegarder le modèle
    if save_model:
        model_path = os.path.join(MODEL_DIR, "centralized.h5")
        model.save(model_path)
        print(f"Modèle sauvegardé: {model_path}")

    # Sauvegarder les logs
    save_training_logs(history, "Centralized", "IDD")

    return model, history


def save_training_logs(history, method_name, data_distribution):
    """
    Sauvegarde les logs d'entraînement

    Args:
        history: historique d'entraînement Keras
        method_name: nom de la méthode
        data_distribution: distribution des données
    """
    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    log_path = os.path.join(LOG_DIR, f"{timestamp}.txt")

    try:
        with open(log_path, 'w') as log_file:
            log_file.write(f"CIFAR-10, {method_name}, {data_distribution}, batch_size: {BATCH_SIZE}\n")
            log_file.write(f"Train Loss = {history.history['loss']}\n")
            log_file.write(f"Val Loss = {history.history['val_loss']}\n")
            log_file.write(f"Val Accuracy = {history.history['val_categorical_accuracy']}\n")

        print(f"Logs sauvegardés: {log_path}")
    except IOError:
        print("Erreur lors de la sauvegarde des logs")


def evaluate_centralized_model(model, test_data):
    """
    Évalue le modèle centralisé

    Args:
        model: modèle entraîné
        test_data: données de test

    Returns:
        loss: perte sur les données de test
        accuracy: précision sur les données de test
    """
    print("\n=== ÉVALUATION MODÈLE CENTRALISÉ ===")

    loss, accuracy = model.evaluate(
        test_data[0], test_data[1],
        verbose=0
    )

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    return loss, accuracy