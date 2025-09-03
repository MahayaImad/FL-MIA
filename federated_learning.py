"""
Federated learning implementation using FedAvg algorithm
"""

import os
import numpy as np
import datetime
from models import create_compiled_keras_model
from config import BATCH_SIZE, MODEL_DIR, LOG_DIR, CLIENTS


def train_federated_model(fed_data, train_data, test_data, rounds=36, local_epochs=1, save_model=True):
    """
    Entraîne un modèle avec l'algorithme FedAvg

    Args:
        fed_data: données fédérées (liste de tuples (X, y) pour chaque client)
        train_data: données d'entraînement complètes (pour évaluation)
        test_data: données de test
        rounds: nombre de tours de communication
        local_epochs: nombre d'époques locales par tour
        save_model: sauvegarder le modèle final ou non

    Returns:
        global_model: modèle global final
        train_losses: pertes d'entraînement par tour
        test_losses: pertes de test par tour
        test_accuracies: précisions de test par tour
    """
    print("=== ENTRAÎNEMENT FÉDÉRÉ (FedAvg) ===")
    print(f"Nombre de tours: {rounds}")
    print(f"Époques locales par tour: {local_epochs}")
    print(f"Nombre de clients: {CLIENTS}")

    # Créer le modèle global initial
    global_model = create_compiled_keras_model()

    # Stocker les métriques
    train_losses = []
    test_losses = []
    test_accuracies = []

    # Boucle principale FedAvg
    for round_num in range(rounds):
        print(f"\n--- Tour {round_num + 1}/{rounds} ---")

        # Stocker les deltas de poids
        weight_deltas = []

        # Entraînement local pour chaque client
        for client_id in range(CLIENTS):
            print(f"Entraînement local client {client_id}")

            # Créer un modèle local et copier les poids globaux
            local_model = create_compiled_keras_model()
            local_model.set_weights(global_model.get_weights())

            # Entraîner localement
            local_model.fit(
                fed_data[client_id][0], fed_data[client_id][1],
                batch_size=BATCH_SIZE,
                epochs=local_epochs,
                verbose=0
            )

            # Calculer le delta des poids
            global_weights = np.array(global_model.get_weights(), dtype=object)
            local_weights = np.array(local_model.get_weights(), dtype=object)
            delta = global_weights - local_weights
            weight_deltas.append(delta)

        # Agrégation FedAvg (moyenne des deltas)
        avg_delta = np.mean(weight_deltas, axis=0)
        new_weights = np.array(global_model.get_weights(), dtype=object) - avg_delta
        global_model.set_weights(new_weights)

        # Évaluation du modèle global
        train_loss = global_model.evaluate(train_data[0], train_data[1], verbose=0)[0]
        test_loss, test_accuracy = global_model.evaluate(test_data[0], test_data[1], verbose=0)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

    # Sauvegarder le modèle final
    if save_model:
        model_path = os.path.join(MODEL_DIR, "federated_model.h5")
        global_model.save(model_path)
        print(f"Modèle fédéré sauvegardé: {model_path}")

    # Sauvegarder les logs
    save_federated_logs(train_losses, test_losses, test_accuracies)

    return global_model, train_losses, test_losses, test_accuracies


def save_federated_logs(train_losses, test_losses, test_accuracies):
    """
    Sauvegarde les logs d'entraînement fédéré

    Args:
        train_losses: pertes d'entraînement
        test_losses: pertes de test
        test_accuracies: précisions de test
    """
    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    log_path = os.path.join(LOG_DIR, f"federated_learning_{timestamp}.txt")

    try:
        with open(log_path, 'w') as log_file:
            log_file.write(f"CIFAR-10, Federated Learning (FedAvg), IDD, batch_size: {BATCH_SIZE}\n")
            log_file.write(f"Train Loss = {train_losses}\n")
            log_file.write(f"Test Loss = {test_losses}\n")
            log_file.write(f"Test Accuracy = {test_accuracies}\n")

        print(f"Logs sauvegardés: {log_path}")
    except IOError:
        print("Erreur lors de la sauvegarde des logs")


def evaluate_federated_model(model, test_data):
    """
    Évalue le modèle fédéré final

    Args:
        model: modèle fédéré entraîné
        test_data: données de test

    Returns:
        loss: perte sur les données de test
        accuracy: précision sur les données de test
    """
    print("\n=== ÉVALUATION MODÈLE FÉDÉRÉ ===")

    loss, accuracy = model.evaluate(
        test_data[0], test_data[1],
        verbose=0
    )

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    return loss, accuracy


def compare_federated_vs_centralized(fed_model, centralized_model, test_data):
    """
    Compare les performances entre le modèle fédéré et centralisé

    Args:
        fed_model: modèle fédéré
        centralized_model: modèle centralisé
        test_data: données de test

    Returns:
        comparison: dictionnaire avec les résultats de comparaison
    """
    print("\n=== COMPARAISON FÉDÉRÉ VS CENTRALISÉ ===")

    # Évaluer le modèle fédéré
    fed_loss, fed_accuracy = fed_model.evaluate(test_data[0], test_data[1], verbose=0)

    # Évaluer le modèle centralisé
    cent_loss, cent_accuracy = centralized_model.evaluate(test_data[0], test_data[1], verbose=0)

    print(f"Modèle fédéré - Loss: {fed_loss:.4f}, Accuracy: {fed_accuracy:.4f}")
    print(f"Modèle centralisé - Loss: {cent_loss:.4f}, Accuracy: {cent_accuracy:.4f}")

    accuracy_diff = fed_accuracy - cent_accuracy
    loss_diff = fed_loss - cent_loss

    print(f"Différence de précision: {accuracy_diff:.4f}")
    print(f"Différence de perte: {loss_diff:.4f}")

    comparison = {
        'federated_loss': fed_loss,
        'federated_accuracy': fed_accuracy,
        'centralized_loss': cent_loss,
        'centralized_accuracy': cent_accuracy,
        'accuracy_difference': accuracy_diff,
        'loss_difference': loss_diff
    }

    return comparison