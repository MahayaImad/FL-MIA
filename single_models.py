"""
Individual model training for each client
"""

import os
import numpy as np
import datetime
from models import create_compiled_keras_model
from config import EPOCHS_CENTRALIZED, BATCH_SIZE, MODEL_DIR, LOG_DIR, CLIENTS


def train_single_models(fed_data, test_data, save_models=True):
    """
    Entraîne un modèle individuel pour chaque client

    Args:
        fed_data: données fédérées (liste de tuples (X, y) pour chaque client)
        test_data: données de test
        save_models: sauvegarder les modèles ou non

    Returns:
        models: liste des modèles entraînés
        histories: liste des historiques d'entraînement
    """
    print("=== ENTRAÎNEMENT DE MODÈLES INDIVIDUELS ===")

    models = []
    histories = []

    for client_id in range(CLIENTS):
        print(f"\nEntraînement du modèle pour le client {client_id}")
        print(f"Données client {client_id}: {fed_data[client_id][0].shape}")

        # Créer un nouveau modèle pour ce client
        model = create_compiled_keras_model()

        # Entraîner le modèle
        history = model.fit(
            fed_data[client_id][0], fed_data[client_id][1],
            validation_data=test_data,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS_CENTRALIZED,
            verbose=1
        )

        # Sauvegarder le modèle
        if save_models:
            model_path = os.path.join(MODEL_DIR, f"individual_{client_id}.h5")
            model.save(model_path)
            print(f"Modèle client {client_id} sauvegardé: {model_path}")

        models.append(model)
        histories.append(history)

        # Sauvegarder les logs pour ce client
        save_individual_logs(history, client_id)

    # Calculer et sauvegarder les métriques moyennes
    calculate_average_metrics(histories)

    return models, histories


def save_individual_logs(history, client_id):
    """
    Sauvegarde les logs pour un client individuel

    Args:
        history: historique d'entraînement
        client_id: ID du client
    """
    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    log_path = os.path.join(LOG_DIR, f"individual_{client_id}_{timestamp}.txt")

    try:
        with open(log_path, 'w') as log_file:
            log_file.write(f"CIFAR-10, Individual Client {client_id}, IDD, batch_size: {BATCH_SIZE}\n")
            log_file.write(f"Train Loss = {history.history['loss']}\n")
            log_file.write(f"Val Loss = {history.history['val_loss']}\n")
            log_file.write(f"Val Accuracy = {history.history['val_categorical_accuracy']}\n")

        print(f"Logs client {client_id} sauvegardés: {log_path}")
    except IOError:
        print(f"Erreur lors de la sauvegarde des logs pour le client {client_id}")


def calculate_average_metrics(histories):
    """
    Calcule les métriques moyennes pour tous les clients

    Args:
        histories: liste des historiques d'entraînement
    """
    print("\n=== CALCUL DES MÉTRIQUES MOYENNES ===")

    # Calculer les moyennes
    avg_train_loss = np.mean([h.history['loss'] for h in histories], axis=0)
    avg_val_loss = np.mean([h.history['val_loss'] for h in histories], axis=0)
    avg_val_accuracy = np.mean([h.history['val_categorical_accuracy'] for h in histories], axis=0)

    # Sauvegarder les métriques moyennes
    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    log_path = os.path.join(LOG_DIR, f"average_individual_{timestamp}.txt")

    try:
        with open(log_path, 'w') as log_file:
            log_file.write(f"CIFAR-10, Individual Average, IDD, batch_size: {BATCH_SIZE}\n")
            log_file.write(f"Avg Train Loss = {avg_train_loss.tolist()}\n")
            log_file.write(f"Avg Val Loss = {avg_val_loss.tolist()}\n")
            log_file.write(f"Avg Val Accuracy = {avg_val_accuracy.tolist()}\n")

        print(f"Métriques moyennes sauvegardées: {log_path}")
    except IOError:
        print("Erreur lors de la sauvegarde des métriques moyennes")

    # Afficher les résultats finaux
    print(f"Perte d'entraînement moyenne finale: {avg_train_loss[-1]:.4f}")
    print(f"Perte de validation moyenne finale: {avg_val_loss[-1]:.4f}")
    print(f"Précision de validation moyenne finale: {avg_val_accuracy[-1]:.4f}")


def evaluate_individual_models(models, test_data):
    """
    Évalue chaque modèle individuel

    Args:
        models: liste des modèles entraînés
        test_data: données de test

    Returns:
        results: liste des résultats (loss, accuracy) pour chaque modèle
    """
    print("\n=== ÉVALUATION DES MODÈLES INDIVIDUELS ===")

    results = []

    for client_id, model in enumerate(models):
        loss, accuracy = model.evaluate(
            test_data[0], test_data[1],
            verbose=0
        )

        results.append((loss, accuracy))
        print(f"Client {client_id} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Calculer les moyennes
    avg_loss = np.mean([r[0] for r in results])
    avg_accuracy = np.mean([r[1] for r in results])

    print(f"\nMoyenne - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    return results