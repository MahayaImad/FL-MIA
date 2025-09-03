"""
Script principal pour exécuter les expériences d'apprentissage fédéré
et les attaques d'inférence de membership
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Imports des modules locaux
from data_preparation import prepare_cifar10_data, prepare_shadow_data
from centralized_training import train_centralized_model, evaluate_centralized_model
from single_models import train_single_models, evaluate_individual_models
from ensemble_models import create_ensemble_predictions, evaluate_ensemble, compare_ensemble_vs_individuals, analyze_ensemble_diversity
from cyclic_weight_transfer import train_cyclic_weight_transfer, evaluate_cyclic_model
from federated_learning import train_federated_model, evaluate_federated_model, compare_federated_vs_centralized
from membership_inference_attack import (
    perform_outsider_attack, perform_insider_attack, perform_original_mia_attack,
    save_attack_results
)
from utils import (
    plot_training_history, compare_models_performance, plot_attack_results,
    calculate_privacy_metrics, print_privacy_analysis, save_experiment_summary,
    load_model_safely, create_data_visualization
)


def setup_experiment():
    """
    Configure l'environnement d'expérimentation
    """
    print("=== CONFIGURATION DE L'EXPÉRIMENTATION ===")

    # Vérifier les dépendances
    try:
        import tensorflow as tf
        import numpy as np
        import sklearn
        print(f"TensorFlow version: {tf.__version__}")
        print(f"NumPy version: {np.__version__}")
        print(f"Scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        print(f"Erreur d'importation: {e}")
        sys.exit(1)

    # Configurer TensorFlow
    tf.config.experimental.set_memory_growth(
        tf.config.experimental.list_physical_devices('GPU')[0], True
    ) if tf.config.experimental.list_physical_devices('GPU') else None

    # Fixer les graines pour la reproductibilité
    np.random.seed(42)
    tf.random.set_seed(42)

    print("Configuration terminée avec succès")


def run_data_preparation():
    """
    Exécute la préparation des données

    Returns:
        tuple: données préparées
    """
    print("\n=== PRÉPARATION DES DONNÉES ===")

    # Préparer les données CIFAR-10
    cifar_train_data, cifar_train_fed_data, cifar_test_data, attacker_data = prepare_cifar10_data()

    # Afficher les informations sur les données
    print(f"Données d'entraînement: {cifar_train_data[0].shape}")
    print(f"Données de test: {cifar_test_data[0].shape}")
    print(f"Nombre de clients: {len(cifar_train_fed_data)}")
    print(f"Données par client: {cifar_train_fed_data[0][0].shape}")
    print(f"Données attaquant: {attacker_data[0].shape}")

    # Créer une visualisation des données
    create_data_visualization(
        cifar_train_data[0][:10],
        cifar_train_data[1][:10],
        "Échantillons CIFAR-10",
        "visualizations/cifar10_samples.png"
    )

    return cifar_train_data, cifar_train_fed_data, cifar_test_data, attacker_data


def run_training_experiments(cifar_train_data, cifar_train_fed_data, cifar_test_data):
    """
    Exécute les expériences d'entraînement

    Args:
        cifar_train_data: données d'entraînement complètes
        cifar_train_fed_data: données d'entraînement fédérées
        cifar_test_data: données de test

    Returns:
        dict: modèles entraînés et leurs performances
    """
    print("\n=== EXPÉRIENCES D'ENTRAÎNEMENT ===")

    results = {}

    # 1. Entraînement centralisé
    print("\n1. ENTRAÎNEMENT CENTRALISÉ")
    centralized_model, centralized_history = train_centralized_model(
        cifar_train_data, cifar_test_data
    )
    cent_loss, cent_acc = evaluate_centralized_model(centralized_model, cifar_test_data)
    results['centralized'] = {
        'model': centralized_model,
        'history': centralized_history,
        'performance': (cent_loss, cent_acc)
    }

    # 2. Modèles individuels
    print("\n2. MODÈLES INDIVIDUELS")
    individual_models, individual_histories = train_single_models(
        cifar_train_fed_data, cifar_test_data
    )
    individual_results = evaluate_individual_models(individual_models, cifar_test_data)
    avg_ind_loss = np.mean([r[0] for r in individual_results])
    avg_ind_acc = np.mean([r[1] for r in individual_results])
    results['individual'] = {
        'models': individual_models,
        'histories': individual_histories,
        'performance': (avg_ind_loss, avg_ind_acc)
    }

    # 3. Ensemble des modèles individuels
    print("\n3. ENSEMBLE DES MODÈLES")

    # Option 1: Utiliser la fonction corrigée simple
    ensemble_predictions, individual_predictions = create_ensemble_predictions(
        individual_models, cifar_test_data
    )

    # Utiliser la version corrigée de evaluate_ensemble
    ens_train_loss, ens_test_loss, ens_test_acc = evaluate_ensemble(
        ensemble_predictions, cifar_train_data, cifar_test_data
    )

    # Option 2: Utiliser la fonction complète si vous voulez les vraies métriques
    # ens_train_loss, ens_test_loss, ens_test_acc, ensemble_predictions = evaluate_ensemble_complete(
    #     individual_models, cifar_train_data, cifar_test_data
    # )

    results['ensemble'] = {
        'predictions': ensemble_predictions,
        'performance': (ens_test_loss, ens_test_acc)
    }

    # Comparaison ensemble vs individuels
    comparison_results = compare_ensemble_vs_individuals(
        individual_models, ensemble_predictions, cifar_test_data
    )

    # Analyse de la diversité
    diversity_metrics = analyze_ensemble_diversity(
        individual_predictions, cifar_test_data
    )

    # 4. Transfert cyclique de poids
    print("\n4. TRANSFERT CYCLIQUE DE POIDS")
    cyclic_model, cyclic_train_loss, cyclic_test_loss, cyclic_test_acc = train_cyclic_weight_transfer(
        cifar_train_fed_data, cifar_test_data
    )
    cyc_loss, cyc_acc = evaluate_cyclic_model(cyclic_model, cifar_test_data)
    results['cyclic'] = {
        'model': cyclic_model,
        'performance': (cyc_loss, cyc_acc),
        'history': (cyclic_train_loss, cyclic_test_loss, cyclic_test_acc)
    }

    # 5. Apprentissage fédéré
    print("\n5. APPRENTISSAGE FÉDÉRÉ")
    federated_model, fed_train_loss, fed_test_loss, fed_test_acc = train_federated_model(
        cifar_train_fed_data, cifar_train_data, cifar_test_data
    )
    fed_loss, fed_acc = evaluate_federated_model(federated_model, cifar_test_data)
    results['federated'] = {
        'model': federated_model,
        'performance': (fed_loss, fed_acc),
        'history': (fed_train_loss, fed_test_loss, fed_test_acc)
    }

    # Comparaison des performances
    print("\n=== COMPARAISON DES PERFORMANCES ===")
    performance_comparison = {
        'Centralized': results['centralized']['performance'],
        'Individual (Avg)': results['individual']['performance'],
        'Ensemble': results['ensemble']['performance'],
        'Cyclic Transfer': results['cyclic']['performance'],
        'Federated': results['federated']['performance']
    }

    compare_models_performance(performance_comparison, 'accuracy')
    compare_models_performance(performance_comparison, 'loss')

    return results


def run_attack_experiments(models, cifar_train_fed_data, cifar_test_data, attacker_data):
    """
    Exécute les expériences d'attaque d'inférence de membership

    Args:
        models: dictionnaire des modèles entraînés
        cifar_train_fed_data: données d'entraînement fédérées
        cifar_test_data: données de test
        attacker_data: données de l'attaquant

    Returns:
        dict: résultats des attaques
    """
    print("\n=== EXPÉRIENCES D'ATTAQUE D'INFÉRENCE DE MEMBERSHIP ===")

    attack_results = {}

    # Préparer les données externes pour les attaques
    external_data = cifar_test_data

    # Tester différents modèles
    models_to_test = {
        'centralized': models['centralized']['model'],
        'federated': models['federated']['model'],
        'cyclic': models['cyclic']['model']
    }

    for model_name, model in models_to_test.items():
        print(f"\n--- Attaques sur le modèle {model_name.upper()} ---")

        # Données cibles (premier client pour les attaques)
        target_data = cifar_train_fed_data[0]

        # 1. Attaque d'outsider
        print(f"\n1. Attaque d'outsider sur {model_name}")
        try:
            outsider_accuracy, outsider_class_precisions = perform_outsider_attack(
                model, target_data, external_data, attacker_data
            )

            attack_results[f'{model_name}_outsider'] = {
                'overall_accuracy': outsider_accuracy,
                'class_precisions': outsider_class_precisions
            }

            save_attack_results(
                'outsider', model_name, outsider_accuracy, outsider_class_precisions
            )
        except Exception as e:
            print(f"Erreur lors de l'attaque d'outsider: {e}")

        # 2. Attaque d'insider
        print(f"\n2. Attaque d'insider sur {model_name}")
        try:
            # Utiliser les données d'un autre client comme attaquant insider
            insider_data = cifar_train_fed_data[1]  # Client 1 attaque client 0

            insider_accuracy, insider_class_precisions = perform_insider_attack(
                model, target_data, insider_data, external_data
            )

            attack_results[f'{model_name}_insider'] = {
                'overall_accuracy': insider_accuracy,
                'class_precisions': insider_class_precisions
            }

            save_attack_results(
                'insider', model_name, insider_accuracy, insider_class_precisions
            )
        except Exception as e:
            print(f"Erreur lors de l'attaque d'insider: {e}")

    # 3. Attaque MIA originale (seulement sur le modèle centralisé)
    print(f"\n3. Attaque MIA originale sur le modèle centralisé")
    try:
        original_accuracy, original_class_precisions = perform_original_mia_attack(
            models['centralized']['model'],
            cifar_train_fed_data[0],
            cifar_test_data
        )

        attack_results['centralized_original'] = {
            'overall_accuracy': original_accuracy,
            'class_precisions': original_class_precisions
        }

        save_attack_results(
            'original', 'centralized', original_accuracy, original_class_precisions
        )
    except Exception as e:
        print(f"Erreur lors de l'attaque MIA originale: {e}")

    # Visualiser les résultats d'attaque
    if attack_results:
        plot_attack_results(attack_results)

    return attack_results


def run_privacy_analysis(attack_results):
    """
    Exécute l'analyse de confidentialité

    Args:
        attack_results: résultats des attaques

    Returns:
        dict: métriques de confidentialité
    """
    print("\n=== ANALYSE DE CONFIDENTIALITÉ ===")

    # Extraire les précisions d'attaque
    attack_accuracies = [
        result['overall_accuracy'] for result in attack_results.values()
    ]

    # Calculer les métriques de confidentialité
    privacy_metrics = calculate_privacy_metrics(attack_accuracies)

    # Afficher l'analyse
    print_privacy_analysis(privacy_metrics)

    # Analyse comparative par modèle
    print("\n=== ANALYSE COMPARATIVE PAR MODÈLE ===")

    model_privacy = {}
    for attack_name, result in attack_results.items():
        model_name = attack_name.split('_')[0]
        if model_name not in model_privacy:
            model_privacy[model_name] = []
        model_privacy[model_name].append(result['overall_accuracy'])

    for model_name, accuracies in model_privacy.items():
        avg_accuracy = np.mean(accuracies)
        max_accuracy = np.max(accuracies)
        print(f"{model_name.capitalize()}: Précision moyenne = {avg_accuracy:.4f}, Max = {max_accuracy:.4f}")

    return privacy_metrics


def save_all_results(training_results, attack_results, privacy_metrics):
    """
    Sauvegarde tous les résultats des expériences

    Args:
        training_results: résultats d'entraînement
        attack_results: résultats d'attaque
        privacy_metrics: métriques de confidentialité
    """
    # Préparer le dictionnaire de résultats
    all_results = {
        'model_performance': {
            name: data['performance'] for name, data in training_results.items()
            if 'performance' in data
        },
        'attack_results': attack_results,
        'privacy_metrics': privacy_metrics,
        'timestamp': datetime.now().isoformat()
    }

    # Sauvegarder le résumé
    save_experiment_summary(all_results)

    print("\n=== RÉSULTATS FINAUX ===")
    print("Toutes les expériences ont été complétées avec succès!")
    print("Consultez le dossier 'logs' pour les résultats détaillés.")
    print("Consultez le dossier 'models' pour les modèles sauvegardés.")


def main():
    """
    Fonction principale
    """
    parser = argparse.ArgumentParser(description='Expériences d\'apprentissage fédéré et attaques MIA')
    parser.add_argument('--skip-training', action='store_true',
                        help='Ignorer l\'entraînement et charger les modèles existants')
    parser.add_argument('--skip-attacks', action='store_true',
                        help='Ignorer les attaques d\'inférence de membership')
    parser.add_argument('--model-path', type=str, default='models',
                        help='Chemin vers les modèles sauvegardés')
    parser.add_argument('--visualize', action='store_true',
                        help='Créer des visualisations des résultats')

    args = parser.parse_args()

    # Configuration initiale
    setup_experiment()

    # Préparation des données
    cifar_train_data, cifar_train_fed_data, cifar_test_data, attacker_data = run_data_preparation()

    # Expériences d'entraînement
    if not args.skip_training:
        training_results = run_training_experiments(
            cifar_train_data, cifar_train_fed_data, cifar_test_data
        )
    else:
        print("\n=== CHARGEMENT DES MODÈLES EXISTANTS ===")
        training_results = load_existing_models(args.model_path)

    # Expériences d'attaque
    if not args.skip_attacks:
        attack_results = run_attack_experiments(
            training_results, cifar_train_fed_data, cifar_test_data, attacker_data
        )

        # Analyse de confidentialité
        privacy_metrics = run_privacy_analysis(attack_results)
    else:
        attack_results = {}
        privacy_metrics = {}

    # Sauvegarde des résultats
    save_all_results(training_results, attack_results, privacy_metrics)

    # Visualisations supplémentaires
    if args.visualize:
        create_additional_visualizations(training_results, attack_results)


def load_existing_models(model_path):
    """
    Charge les modèles existants

    Args:
        model_path: chemin vers les modèles

    Returns:
        dict: modèles chargés
    """
    models = {}

    model_files = {
        'centralized': 'centralized.h5',
        'federated': 'federated_model.h5',
        'cyclic': 'cyclic_weight_transfer.h5'
    }

    for model_name, filename in model_files.items():
        filepath = os.path.join(model_path, filename)
        model = load_model_safely(filepath)
        if model:
            models[model_name] = {'model': model, 'performance': None}

    return models


def create_additional_visualizations(training_results, attack_results):
    """
    Crée des visualisations supplémentaires

    Args:
        training_results: résultats d'entraînement
        attack_results: résultats d'attaque
    """
    print("\n=== CRÉATION DE VISUALISATIONS SUPPLÉMENTAIRES ===")

    # Créer le dossier de visualisations
    os.makedirs('visualizations', exist_ok=True)

    # Graphiques d'entraînement
    for model_name, data in training_results.items():
        if 'history' in data:
            plot_training_history(
                data['history'],
                f'Training History - {model_name.capitalize()}',
                f'visualizations/training_{model_name}.png'
            )

    # Graphiques d'attaque
    if attack_results:
        plot_attack_results(attack_results, 'visualizations/attack_results.png')

    print("Visualisations créées dans le dossier 'visualizations'")


if __name__ == "__main__":
    main()