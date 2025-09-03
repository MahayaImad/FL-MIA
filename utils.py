# utils.py
"""
Utility functions for the federated learning experiments
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from config import LOG_DIR


def plot_training_history(history, title="Training History", save_path=None):
    """
    Trace l'historique d'entraînement

    Args:
        history: historique d'entraînement Keras ou listes de métriques
        title: titre du graphique
        save_path: chemin de sauvegarde (optionnel)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if hasattr(history, 'history'):
        # Historique Keras
        epochs = range(1, len(history.history['loss']) + 1)

        # Perte
        ax1.plot(epochs, history.history['loss'], 'b-', label='Training Loss')
        if 'val_loss' in history.history:
            ax1.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Précision
        if 'categorical_accuracy' in history.history:
            ax2.plot(epochs, history.history['categorical_accuracy'], 'b-', label='Training Accuracy')
        if 'val_categorical_accuracy' in history.history:
            ax2.plot(epochs, history.history['val_categorical_accuracy'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
    else:
        # Listes de métriques
        train_loss, val_loss, val_accuracy = history
        epochs = range(1, len(train_loss) + 1)

        ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
        ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.plot(epochs, val_accuracy, 'g-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Graphique sauvegardé: {save_path}")

    plt.show()


def compare_models_performance(results_dict, metric='accuracy', save_path=None):
    """
    Compare les performances de différents modèles

    Args:
        results_dict: dictionnaire {nom_modèle: (loss, accuracy)}
        metric: métrique à comparer ('accuracy' ou 'loss')
        save_path: chemin de sauvegarde (optionnel)
    """
    models = list(results_dict.keys())

    if metric == 'accuracy':
        values = [results_dict[model][1] for model in models]
        ylabel = 'Accuracy'
        title = 'Model Accuracy Comparison'
    else:
        values = [results_dict[model][0] for model in models]
        ylabel = 'Loss'
        title = 'Model Loss Comparison'

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color=['blue', 'green', 'red', 'orange', 'purple'][:len(models)])

    # Ajouter les valeurs sur les barres
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                 f'{value:.4f}', ha='center', va='bottom')

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Models')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Graphique de comparaison sauvegardé: {save_path}")

    plt.show()


def plot_attack_results(attack_results, save_path=None):
    """
    Trace les résultats d'attaque d'inférence de membership

    Args:
        attack_results: dictionnaire des résultats d'attaque
        save_path: chemin de sauvegarde (optionnel)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Précision globale par type d'attaque
    attack_types = list(attack_results.keys())
    overall_accuracies = [attack_results[att]['overall_accuracy'] for att in attack_types]

    ax1.bar(attack_types, overall_accuracies, color=['red', 'orange', 'blue'][:len(attack_types)])
    ax1.set_title('Attack Success Rate by Type')
    ax1.set_ylabel('Attack Accuracy')
    ax1.set_ylim(0, 1)

    # Ajouter les valeurs sur les barres
    for i, v in enumerate(overall_accuracies):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    # Précision par classe pour chaque type d'attaque
    classes = list(range(10))  # CIFAR-10 classes

    for i, attack_type in enumerate(attack_types):
        class_precisions = attack_results[attack_type]['class_precisions']
        ax2.plot(classes, class_precisions, marker='o', label=attack_type)

    ax2.set_title('Attack Success Rate by Class')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Attack Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Graphique des résultats d'attaque sauvegardé: {save_path}")

    plt.show()


def calculate_privacy_metrics(attack_accuracies):
    """
    Calcule les métriques de confidentialité

    Args:
        attack_accuracies: liste des précisions d'attaque

    Returns:
        privacy_metrics: dictionnaire des métriques
    """
    attack_accuracies = np.array(attack_accuracies)

    # Avantage de l'attaque (attack advantage)
    attack_advantage = attack_accuracies - 0.5

    # Perte de confidentialité (privacy loss)
    privacy_loss = np.maximum(0, attack_advantage)

    privacy_metrics = {
        'average_attack_accuracy': np.mean(attack_accuracies),
        'max_attack_accuracy': np.max(attack_accuracies),
        'min_attack_accuracy': np.min(attack_accuracies),
        'average_attack_advantage': np.mean(attack_advantage),
        'max_attack_advantage': np.max(attack_advantage),
        'average_privacy_loss': np.mean(privacy_loss),
        'max_privacy_loss': np.max(privacy_loss)
    }

    return privacy_metrics


def print_privacy_analysis(privacy_metrics):
    """
    Affiche l'analyse de confidentialité

    Args:
        privacy_metrics: métriques de confidentialité
    """
    print("\n=== ANALYSE DE CONFIDENTIALITÉ ===")
    print(f"Précision d'attaque moyenne: {privacy_metrics['average_attack_accuracy']:.4f}")
    print(f"Précision d'attaque maximale: {privacy_metrics['max_attack_accuracy']:.4f}")
    print(f"Précision d'attaque minimale: {privacy_metrics['min_attack_accuracy']:.4f}")
    print(f"Avantage d'attaque moyen: {privacy_metrics['average_attack_advantage']:.4f}")
    print(f"Avantage d'attaque maximal: {privacy_metrics['max_attack_advantage']:.4f}")
    print(f"Perte de confidentialité moyenne: {privacy_metrics['average_privacy_loss']:.4f}")
    print(f"Perte de confidentialité maximale: {privacy_metrics['max_privacy_loss']:.4f}")

    # Interprétation
    avg_accuracy = privacy_metrics['average_attack_accuracy']
    if avg_accuracy > 0.7:
        print("⚠️  RISQUE ÉLEVÉ: Le modèle est vulnérable aux attaques d'inférence de membership")
    elif avg_accuracy > 0.6:
        print("⚠️  RISQUE MODÉRÉ: Le modèle présente une vulnérabilité modérée")
    elif avg_accuracy > 0.55:
        print("⚠️  RISQUE FAIBLE: Le modèle présente une vulnérabilité faible")
    else:
        print("✅ RISQUE MINIMAL: Le modèle est relativement résistant aux attaques")


def save_experiment_summary(results, filepath=None):
    """
    Sauvegarde un résumé des expériences

    Args:
        results: dictionnaire des résultats
        filepath: chemin de sauvegarde (optionnel)
    """
    if filepath is None:
        filepath = os.path.join(LOG_DIR, "experiment_summary.txt")

    try:
        with open(filepath, 'w') as f:
            f.write("=== RÉSUMÉ DES EXPÉRIENCES ===\n\n")

            # Performances des modèles
            if 'model_performance' in results:
                f.write("PERFORMANCES DES MODÈLES:\n")
                for model_name, (loss, accuracy) in results['model_performance'].items():
                    f.write(f"  {model_name}: Loss={loss:.4f}, Accuracy={accuracy:.4f}\n")
                f.write("\n")

            # Résultats d'attaque
            if 'attack_results' in results:
                f.write("RÉSULTATS D'ATTAQUE:\n")
                for attack_type, attack_data in results['attack_results'].items():
                    f.write(f"  {attack_type}: {attack_data['overall_accuracy']:.4f}\n")
                f.write("\n")

            # Métriques de confidentialité
            if 'privacy_metrics' in results:
                f.write("MÉTRIQUES DE CONFIDENTIALITÉ:\n")
                for metric, value in results['privacy_metrics'].items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")

            # Conclusions
            f.write("CONCLUSIONS:\n")
            f.write("- Comparaison des approches d'apprentissage\n")
            f.write("- Analyse des vulnérabilités de confidentialité\n")
            f.write("- Recommandations pour améliorer la confidentialité\n")

        print(f"Résumé des expériences sauvegardé: {filepath}")
    except IOError:
        print("Erreur lors de la sauvegarde du résumé")


def load_model_safely(model_path):
    """
    Charge un modèle de manière sécurisée

    Args:
        model_path: chemin du modèle

    Returns:
        model: modèle chargé ou None si erreur
    """
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        print(f"Modèle chargé avec succès: {model_path}")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle {model_path}: {e}")
        return None


def create_data_visualization(data, labels, title="Data Visualization", save_path=None):
    """
    Crée une visualisation des données

    Args:
        data: données à visualiser
        labels: étiquettes des données
        title: titre de la visualisation
        save_path: chemin de sauvegarde (optionnel)
    """
    # Afficher quelques exemples d'images CIFAR-10
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.imshow(data[i])
        ax.set_title(f'Label: {labels[i].argmax() if len(labels[i].shape) > 0 else labels[i]}')
        ax.axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Visualisation sauvegardée: {save_path}")

    plt.show()