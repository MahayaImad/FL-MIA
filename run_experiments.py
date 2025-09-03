# run_experiments.py
"""
Script d'exécution rapide pour les expériences d'apprentissage fédéré
"""

import os
import sys
import time
from datetime import datetime


def check_dependencies():
    """
    Vérifie que toutes les dépendances sont installées
    """
    required_packages = [
        'tensorflow',
        'numpy',
        'sklearn',
        'matplotlib'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Packages manquants:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstallez les packages manquants avec:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("✅ Toutes les dépendances sont installées")
    return True


def setup_directories():
    """
    Crée les dossiers nécessaires
    """
    directories = ['models', 'logs', 'visualizations']

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Dossier créé: {directory}")


# REMPLACER toute la fonction par :
def run_quick_experiment():
    """
    Exécute une expérience rapide avec des paramètres réduits
    """
    print("\n🚀 EXPÉRIENCE RAPIDE - PARAMÈTRES RÉDUITS")
    print("=" * 50)

    # Modifier temporairement la configuration
    try:
        import config
        original_epochs = config.EPOCHS_CENTRALIZED
        original_size = config.SIZE

        # Réduire les paramètres pour un test rapide
        config.EPOCHS_CENTRALIZED = 2
        config.SIZE = 1000

        print(f"📊 Paramètres réduits:")
        print(f"   - Époques: {config.EPOCHS_CENTRALIZED}")
        print(f"   - Taille par client: {config.SIZE}")
        print(f"   - Nombre de clients: {config.CLIENTS}")

        # Exécuter seulement la partie entraînement pour le test rapide
        from data_preparation import prepare_cifar10_data
        from centralized_training import train_centralized_model

        print("\n📥 Préparation des données...")
        cifar_train_data, cifar_train_fed_data, cifar_test_data, attacker_data = prepare_cifar10_data()

        print("\n🧠 Entraînement centralisé...")
        start_time = time.time()
        centralized_model, centralized_history = train_centralized_model(
            cifar_train_data, cifar_test_data
        )
        end_time = time.time()

        print(f"\n✅ Expérience rapide terminée!")
        print(f"⏱️  Temps d'exécution: {end_time - start_time:.2f} secondes")

        # Restaurer les paramètres originaux
        config.EPOCHS_CENTRALIZED = original_epochs
        config.SIZE = original_size

        return True

    except Exception as e:
        print(f"❌ Erreur lors de l'expérience rapide: {e}")
        print(f"📝 Détails: {str(e)}")

        # Restaurer les paramètres en cas d'erreur
        try:
            config.EPOCHS_CENTRALIZED = original_epochs
            config.SIZE = original_size
        except:
            pass

        return False

def run_full_experiment():
    """
    Exécute l'expérience complète
    """
    print("\n🔬 EXPÉRIENCE COMPLÈTE")
    print("=" * 50)
    print("⚠️  Cette expérience peut prendre plusieurs heures...")

    response = input("Continuer? (y/n): ").lower()
    if response != 'y':
        print("Expérience annulée")
        return False

    try:
        from main import main

        start_time = time.time()
        main()
        end_time = time.time()

        print(f"\n⏱️  Temps d'exécution: {end_time - start_time:.2f} secondes")

    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        return False

    return True


def run_only_attacks():
    """
    Exécute seulement les attaques sur des modèles existants
    """
    print("\n🎯 ATTAQUES SEULEMENT")
    print("=" * 50)

    # Vérifier si les modèles existent
    model_files = ['centralized.h5', 'federated_model.h5', 'cyclic_weight_transfer.h5']
    missing_models = []

    for model_file in model_files:
        if not os.path.exists(os.path.join('models', model_file)):
            missing_models.append(model_file)

    if missing_models:
        print("❌ Modèles manquants:")
        for model in missing_models:
            print(f"  - {model}")
        print("\nEntraînez d'abord les modèles ou utilisez l'option 'full'")
        return False

    try:
        import sys
        sys.argv = ['main.py', '--skip-training']

        from main import main

        start_time = time.time()
        main()
        end_time = time.time()

        print(f"\n⏱️  Temps d'exécution: {end_time - start_time:.2f} secondes")

    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        return False

    return True


def show_results():
    """
    Affiche les résultats des expériences
    """
    print("\n📊 RÉSULTATS DES EXPÉRIENCES")
    print("=" * 50)

    # Vérifier les fichiers de logs
    if os.path.exists('logs'):
        log_files = os.listdir('logs')
        if log_files:
            print(f"📄 {len(log_files)} fichiers de logs trouvés:")
            for log_file in sorted(log_files)[-5:]:  # Afficher les 5 derniers
                print(f"  - {log_file}")
        else:
            print("❌ Aucun fichier de log trouvé")

    # Vérifier les modèles
    if os.path.exists('models'):
        model_files = os.listdir('models')
        if model_files:
            print(f"\n🤖 {len(model_files)} modèles trouvés:")
            for model_file in model_files:
                print(f"  - {model_file}")
        else:
            print("❌ Aucun modèle trouvé")

    # Vérifier les visualisations
    if os.path.exists('visualizations'):
        viz_files = os.listdir('visualizations')
        if viz_files:
            print(f"\n📈 {len(viz_files)} visualisations trouvées:")
            for viz_file in viz_files:
                print(f"  - {viz_file}")


def main():
    """
    Menu principal
    """
    print("🧪 EXPÉRIENCES D'APPRENTISSAGE FÉDÉRÉ ET ATTAQUES MIA")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Vérifier les dépendances
    if not check_dependencies():
        return

    # Créer les dossiers
    setup_directories()

    # Menu principal
    while True:
        print("\n🎯 MENU PRINCIPAL")
        print("=" * 30)
        print("1. 🚀 Expérience rapide (paramètres réduits)")
        print("2. 🔬 Expérience complète")
        print("3. 🎯 Attaques seulement (modèles existants)")
        print("4. 📊 Afficher les résultats")
        print("5. 🔧 Informations système")
        print("6. ❌ Quitter")

        choice = input("\nChoisissez une option (1-6): ").strip()

        if choice == '1':
            run_quick_experiment()
        elif choice == '2':
            run_full_experiment()
        elif choice == '3':
            run_only_attacks()
        elif choice == '4':
            show_results()
        elif choice == '5':
            show_system_info()
        elif choice == '6':
            print("👋 Au revoir!")
            break
        else:
            print("❌ Option invalide")


def show_system_info():
    """
    Affiche les informations système
    """
    print("\n💻 INFORMATIONS SYSTÈME")
    print("=" * 30)

    try:
        import platform
        print(f"🐍 Python: {platform.python_version()}")
        print(f"💾 Système: {platform.system()} {platform.release()}")
        print(f"🏗️  Architecture: {platform.architecture()[0]}")

        import tensorflow as tf
        print(f"🧠 TensorFlow: {tf.__version__}")

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"🎮 GPU disponible: {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("🎮 GPU: Non disponible (CPU seulement)")

        import numpy as np
        print(f"🔢 NumPy: {np.__version__}")

        import sklearn
        print(f"🤖 Scikit-learn: {sklearn.__version__}")

    except Exception as e:
        print(f"❌ Erreur lors de la récupération des informations: {e}")


if __name__ == "__main__":
    main()