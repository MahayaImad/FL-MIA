"""
Cyclic weight transfer implementation
"""

import os
import datetime
from models import create_compiled_keras_model
from config import BATCH_SIZE, MODEL_DIR, LOG_DIR, CLIENTS


def train_cyclic_weight_transfer(fed_data, test_data, rounds=6, epochs_per_round=2, save_model=True):
    """
    Entraîne un modèle avec transfert cyclique de poids

    Args:
        fed_data: données fédérées (liste de tuples (X, y) pour chaque client)
        test_data: données de test
        rounds: nombre de tours d'entraînement
        epochs_per_round: nombre d'époques par tour
        save_model: sauvegarder le modèle final ou non

    Returns:
        model: modèle final entraîné
        train_losses: pertes d'entraînement par tour
        test_losses: pertes de test par tour
        test_accuracies: précisions de test par tour
    """
    print("=== ENTRAÎNEMENT AVEC TRANSFERT CYCLIQUE DE POIDS ===")
    print(f"Nombre de tours: {rounds}")
    print(f"Époques par tour: {epochs_per_round}")
    print(f"Nombre de clients: {CLIENTS}")

    # Créer le modèle initial
    model = create_compiled_keras_model()

    # Stocker les métriques
    train_losses = []
    test_losses = []
    test_accuracies = []

    # Entraînement cyclique
    for round_num in range(rounds):
        print(f"\n--- Tour {round_num + 1}/{rounds} ---")

        # Entraîner séquentiellement sur chaque client
        for client_id in range(CLIENTS):
            print(f"Entraînement sur le client {client_id}")

            # Entraîner le modèle sur les données de ce client
            history = model.fit(
                fed_data[client_id][0], fed_data[client_id][1],
                validation_data=test_data,
                batch_size=BATCH_SIZE,
                epochs=epochs_per_round,
                verbose=1
            )

            # Stocker les métriques du dernier epoch
            train_losses.extend(history.history['loss'])
            test_losses.extend(history.history['val_loss'])
            test_accuracies.extend(history.history['val_categorical_accuracy'])

    # Sauvegarder le modèle final
    if save_model:
        model_path = os.path.join(MODEL_DIR, "cyclic_weight_transfer.h5")
        model.save(model_path)
        print(f"Modèle sauvegardé: {model_path}")

    # Sauvegarder les logs
    save_cyclic_logs(train_losses, test_losses, test_accuracies)

    return model, train_losses, test_losses, test_accuracies


def save_cyclic_logs(train_losses, test_losses, test_accuracies):
    """
    Sauvegarde les logs d'entraînement cyclique

    Args:
        train_losses: pertes d'entraînement
        test_losses: pertes de test
        test_accuracies: précisions de test
    """
    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    log_path = os.path.join(LOG_DIR, f"cyclic_weight_transfer_{timestamp}.txt")

    try:
        with open(log_path, 'w') as log_file:
            log_file.write(f"CIFAR-10, Cyclic Weight Transfer, IDD, batch_size: {BATCH_SIZE}\n")
            log_file.write(f"Train Loss = {train_losses}\n")
            log_file.write(f"Test Loss = {test_losses}\n")
            log_file.write(f"Test Accuracy = {test_accuracies}\n")

        print(f"Logs sauvegardés: {log_path}")
    except IOError:
        print("Erreur lors de la sauvegarde des logs")


def evaluate_cyclic_model(model, test_data):
    """
    Évalue le modèle entraîné avec transfert cyclique

    Args:
        model: modèle entraîné
        test_data: données de test

    Returns:
        loss: perte sur les données de test
        accuracy: précision sur les données de test
    """
    print("\n=== ÉVALUATION MODÈLE CYCLIQUE ===")

    loss, accuracy = model.evaluate(
        test_data[0], test_data[1],
        verbose=0
    )

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    return loss, accuracy