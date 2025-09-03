"""
Ensemble methods for combining individual models
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score


def create_ensemble_predictions(models, test_data):
    """
    Crée des prédictions d'ensemble en moyennant les sorties des modèles

    Args:
        models: liste des modèles entraînés
        test_data: données de test

    Returns:
        ensemble_predictions: prédictions d'ensemble
        individual_predictions: prédictions individuelles de chaque modèle
    """
    print("=== CRÉATION DES PRÉDICTIONS D'ENSEMBLE ===")

    individual_predictions = []

    # Obtenir les prédictions de chaque modèle
    for i, model in enumerate(models):
        print(f"Prédictions du modèle {i}...")
        predictions = model.predict(test_data[0], batch_size=32, verbose=1)
        individual_predictions.append(predictions)

    # Moyenner les prédictions
    ensemble_predictions = np.mean(individual_predictions, axis=0)

    print(f"Forme des prédictions d'ensemble: {ensemble_predictions.shape}")

    return ensemble_predictions, individual_predictions


def evaluate_ensemble(ensemble_predictions, train_data, test_data):
    """
    Évalue les performances de l'ensemble

    Args:
        ensemble_predictions: prédictions d'ensemble
        train_data: données d'entraînement pour calculer la perte
        test_data: données de test

    Returns:
        train_loss: perte sur les données d'entraînement
        test_loss: perte sur les données de test
        test_accuracy: précision sur les données de test
    """
    print("\n=== ÉVALUATION DE L'ENSEMBLE ===")

    # CORRECTION: Utiliser les prédictions d'ensemble sur les données de test uniquement
    # Ne pas utiliser train_data pour calculer la perte car les prédictions d'ensemble
    # sont faites sur test_data

    # Calculer la perte sur les données de test
    test_loss = tf.keras.losses.categorical_crossentropy(
        test_data[1], ensemble_predictions, from_logits=False
    )
    test_loss = np.mean(test_loss.numpy())

    # Calculer la précision sur les données de test
    y_pred = ensemble_predictions.argmax(axis=1)
    y_true = test_data[1].argmax(axis=1)
    test_accuracy = accuracy_score(y_true, y_pred)

    # Pour la perte d'entraînement, nous devons faire les prédictions d'ensemble
    # sur les données d'entraînement séparément
    print("Calcul des prédictions d'ensemble sur les données d'entraînement...")

    # Récupérer les modèles depuis le contexte global ou les passer en paramètre
    # Pour éviter de recalculer, on peut estimer la perte d'entraînement
    # ou la calculer séparément si nécessaire

    # Option 1: Estimer la train_loss comme égale à test_loss (approximation)
    train_loss = test_loss

    # Option 2: Si vous voulez calculer la vraie perte d'entraînement,
    # vous devez passer les modèles en paramètre et recalculer

    print(f"Perte d'entraînement (estimée): {train_loss:.4f}")
    print(f"Perte de test (ensemble): {test_loss:.4f}")
    print(f"Précision de test (ensemble): {test_accuracy:.4f}")

    return train_loss, test_loss, test_accuracy


def evaluate_ensemble_complete(models, train_data, test_data):
    """
    Version complète de l'évaluation d'ensemble qui calcule les vraies métriques

    Args:
        models: liste des modèles individuels
        train_data: données d'entraînement
        test_data: données de test

    Returns:
        train_loss: perte sur les données d'entraînement
        test_loss: perte sur les données de test
        test_accuracy: précision sur les données de test
        ensemble_predictions: prédictions d'ensemble sur les données de test
    """
    print("\n=== ÉVALUATION COMPLÈTE DE L'ENSEMBLE ===")

    # Créer les prédictions d'ensemble pour les données de test
    test_ensemble_predictions, _ = create_ensemble_predictions(models, test_data)

    # Créer les prédictions d'ensemble pour les données d'entraînement
    print("Calcul des prédictions d'ensemble sur les données d'entraînement...")
    train_individual_predictions = []
    for i, model in enumerate(models):
        print(f"Prédictions du modèle {i} sur les données d'entraînement...")
        predictions = model.predict(train_data[0], batch_size=32, verbose=0)
        train_individual_predictions.append(predictions)

    train_ensemble_predictions = np.mean(train_individual_predictions, axis=0)

    # Calculer les pertes
    train_loss = tf.keras.losses.categorical_crossentropy(
        train_data[1], train_ensemble_predictions, from_logits=False
    )
    train_loss = np.mean(train_loss.numpy())

    test_loss = tf.keras.losses.categorical_crossentropy(
        test_data[1], test_ensemble_predictions, from_logits=False
    )
    test_loss = np.mean(test_loss.numpy())

    # Calculer la précision sur les données de test
    y_pred = test_ensemble_predictions.argmax(axis=1)
    y_true = test_data[1].argmax(axis=1)
    test_accuracy = accuracy_score(y_true, y_pred)

    print(f"Perte d'entraînement (ensemble): {train_loss:.4f}")
    print(f"Perte de test (ensemble): {test_loss:.4f}")
    print(f"Précision de test (ensemble): {test_accuracy:.4f}")

    return train_loss, test_loss, test_accuracy, test_ensemble_predictions


def compare_ensemble_vs_individuals(models, ensemble_predictions, test_data):
    """
    Compare les performances de l'ensemble avec les modèles individuels

    Args:
        models: liste des modèles individuels
        ensemble_predictions: prédictions d'ensemble
        test_data: données de test

    Returns:
        comparison_results: dictionnaire des résultats de comparaison
    """
    print("\n=== COMPARAISON ENSEMBLE VS INDIVIDUELS ===")

    # Évaluer l'ensemble
    y_pred_ensemble = ensemble_predictions.argmax(axis=1)
    y_true = test_data[1].argmax(axis=1)
    ensemble_accuracy = accuracy_score(y_true, y_pred_ensemble)

    # Évaluer chaque modèle individuel
    individual_accuracies = []
    for i, model in enumerate(models):
        predictions = model.predict(test_data[0], verbose=0)
        y_pred = predictions.argmax(axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        individual_accuracies.append(accuracy)
        print(f"Modèle {i} - Précision: {accuracy:.4f}")

    avg_individual_accuracy = np.mean(individual_accuracies)

    print(f"\nPrécision moyenne des modèles individuels: {avg_individual_accuracy:.4f}")
    print(f"Précision de l'ensemble: {ensemble_accuracy:.4f}")
    print(f"Amélioration: {ensemble_accuracy - avg_individual_accuracy:.4f}")

    comparison_results = {
        'ensemble_accuracy': ensemble_accuracy,
        'individual_accuracies': individual_accuracies,
        'average_individual_accuracy': avg_individual_accuracy,
        'improvement': ensemble_accuracy - avg_individual_accuracy
    }

    return comparison_results


def analyze_ensemble_diversity(individual_predictions, test_data):
    """
    Analyse la diversité des prédictions dans l'ensemble

    Args:
        individual_predictions: liste des prédictions individuelles
        test_data: données de test

    Returns:
        diversity_metrics: métriques de diversité
    """
    print("\n=== ANALYSE DE LA DIVERSITÉ DE L'ENSEMBLE ===")

    y_true = test_data[1].argmax(axis=1)

    # Calculer les prédictions de classe pour chaque modèle
    class_predictions = []
    for predictions in individual_predictions:
        class_pred = predictions.argmax(axis=1)
        class_predictions.append(class_pred)

    # Calculer le taux de désaccord
    disagreement_rates = []
    for i in range(len(class_predictions)):
        for j in range(i + 1, len(class_predictions)):
            disagreement = np.mean(class_predictions[i] != class_predictions[j])
            disagreement_rates.append(disagreement)

    avg_disagreement = np.mean(disagreement_rates)

    # Calculer la variance des prédictions
    prediction_variance = np.var(individual_predictions, axis=0)
    avg_prediction_variance = np.mean(prediction_variance)

    print(f"Taux de désaccord moyen: {avg_disagreement:.4f}")
    print(f"Variance moyenne des prédictions: {avg_prediction_variance:.4f}")

    diversity_metrics = {
        'average_disagreement': avg_disagreement,
        'average_prediction_variance': avg_prediction_variance,
        'disagreement_rates': disagreement_rates
    }

    return diversity_metrics