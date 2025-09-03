"""
Membership inference attack implementation
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
import datetime

from models import create_shadow_model, create_attack_model, create_adaboost_attack_model
from config import EPOCHS_SHADOW, EPOCHS_ATTACK, BATCH_SIZE, MODEL_DIR, LOG_DIR, NUM_CLASSES, SIZE


def prepare_attack_data(model, data_in, data_out):
    """
    Prépare les données pour l'attaque d'inférence de membership

    Args:
        model: modèle cible
        data_in: données d'entraînement du modèle (membres)
        data_out: données non vues par le modèle (non-membres)

    Returns:
        attack_data: données formatées pour l'attaque
        membership_labels: étiquettes de membership (1=membre, 0=non-membre)
    """
    X_in, y_in = data_in
    X_out, y_out = data_out

    # Obtenir les prédictions du modèle cible
    pred_in = model.predict(X_in)
    pred_out = model.predict(X_out)

    # Combiner prédictions et étiquettes vraies
    X_attack = np.concatenate([
        np.concatenate([pred_in, pred_out], axis=0),
        np.concatenate([y_in, y_out], axis=0)
    ], axis=1)

    # Étiquettes de membership
    y_attack = np.concatenate([
        np.ones(len(X_in)),  # Membres
        np.zeros(len(X_out))  # Non-membres
    ])

    return X_attack, y_attack


class ShadowModelBundle:
    """
    Bundle de modèles shadow pour l'attaque d'inférence de membership
    """

    def __init__(self, model_fn, shadow_dataset_size, num_models):
        self.model_fn = model_fn
        self.shadow_dataset_size = shadow_dataset_size
        self.num_models = num_models
        self.models = []

    def fit_transform(self, X_train, y_train, fit_kwargs=None):
        """
        Entraîne les modèles shadow et génère les données d'attaque

        Args:
            X_train: données d'entraînement
            y_train: étiquettes d'entraînement
            fit_kwargs: arguments pour l'entraînement

        Returns:
            X_shadow: données d'attaque
            y_shadow: étiquettes de membership
        """
        if fit_kwargs is None:
            fit_kwargs = {}

        X_shadow_list = []
        y_shadow_list = []

        for i in range(self.num_models):
            print(f"Entraînement du modèle shadow {i + 1}/{self.num_models}")

            # Échantillonner les données pour ce modèle shadow
            indices = np.random.choice(
                len(X_train),
                size=self.shadow_dataset_size,
                replace=False
            )

            X_shadow_train = X_train[indices]
            y_shadow_train = y_train[indices]

            # Données non vues par ce modèle shadow
            remaining_indices = np.setdiff1d(np.arange(len(X_train)), indices)
            X_shadow_out = X_train[remaining_indices[:self.shadow_dataset_size]]
            y_shadow_out = y_train[remaining_indices[:self.shadow_dataset_size]]

            # Créer et entraîner le modèle shadow
            shadow_model = self.model_fn()
            shadow_model.fit(
                X_shadow_train, y_shadow_train,
                **fit_kwargs
            )

            self.models.append(shadow_model)

            # Préparer les données d'attaque pour ce modèle
            X_attack, y_attack = prepare_attack_data(
                shadow_model,
                (X_shadow_train, y_shadow_train),
                (X_shadow_out, y_shadow_out)
            )

            X_shadow_list.append(X_attack)
            y_shadow_list.append(y_attack)

        # Combiner toutes les données d'attaque
        X_shadow = np.concatenate(X_shadow_list, axis=0)
        y_shadow = np.concatenate(y_shadow_list, axis=0)

        return X_shadow, y_shadow


class AttackModelBundle:
    """
    Bundle de modèles d'attaque par classe
    """

    def __init__(self, attack_model_fn, num_classes):
        self.attack_model_fn = attack_model_fn
        self.num_classes = num_classes
        self.models = {}

    def fit(self, X_shadow, y_shadow, fit_kwargs=None):
        """
        Entraîne un modèle d'attaque pour chaque classe

        Args:
            X_shadow: données d'attaque
            y_shadow: étiquettes de membership
            fit_kwargs: arguments pour l'entraînement
        """
        if fit_kwargs is None:
            fit_kwargs = {}

        for class_id in range(self.num_classes):
            print(f"Entraînement du modèle d'attaque pour la classe {class_id}")

            # Filtrer les données pour cette classe
            class_indices = np.where(X_shadow[:, -self.num_classes:].argmax(axis=1) == class_id)[0]

            if len(class_indices) > 0:
                X_class = X_shadow[class_indices, :self.num_classes]  # Seulement les prédictions
                y_class = y_shadow[class_indices]

                # Créer et entraîner le modèle d'attaque
                attack_model = self.attack_model_fn()

                if hasattr(attack_model, 'compile'):
                    # Modèle Keras
                    attack_model.fit(X_class, y_class, **fit_kwargs)
                else:
                    # Modèle sklearn
                    attack_model.fit(X_class, y_class)

                self.models[class_id] = attack_model

    def predict(self, X_attack):
        """
        Fait des prédictions d'attaque

        Args:
            X_attack: données d'attaque

        Returns:
            predictions: prédictions de membership
        """
        predictions = np.zeros(len(X_attack))

        for class_id in range(self.num_classes):
            # Filtrer les données pour cette classe
            class_indices = np.where(X_attack[:, -self.num_classes:].argmax(axis=1) == class_id)[0]

            if len(class_indices) > 0 and class_id in self.models:
                X_class = X_attack[class_indices, :self.num_classes]

                # Prédire
                if hasattr(self.models[class_id], 'predict') and hasattr(self.models[class_id], 'fit'):
                    # Modèle Keras
                    class_preds = self.models[class_id].predict(X_class)
                    if len(class_preds.shape) > 1:
                        class_preds = class_preds[:, 0]  # Première sortie pour modèle binaire
                    predictions[class_indices] = class_preds
                else:
                    # Modèle binaire simple
                    predictions[class_indices] = self.models[class_id].predict(X_class)

        return (predictions > 0.5).astype(int)


def perform_outsider_attack(target_model, target_data, external_data, shadow_data):
    """
    Effectue une attaque d'outsider (attaquant externe)

    Args:
        target_model: modèle cible
        target_data: données d'entraînement du modèle cible
        external_data: données externes
        shadow_data: données pour entraîner les modèles shadow

    Returns:
        attack_accuracy: précision de l'attaque
        class_precisions: précisions par classe
    """
    print("=== ATTAQUE D'OUTSIDER ===")

    # Entraîner un modèle shadow
    print("Entraînement du modèle shadow...")
    shadow_model = create_shadow_model()
    shadow_model.fit(
        shadow_data[0], shadow_data[1],
        validation_data=external_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_SHADOW,
        verbose=1
    )

    # Sauvegarder le modèle shadow
    shadow_path = os.path.join(MODEL_DIR, "shadow_model.h5")
    shadow_model.save(shadow_path)

    # Préparer les données d'attaque avec le modèle shadow
    print("Préparation des données d'attaque...")

    in_preds = shadow_model.predict(shadow_data[0])
    out_preds = shadow_model.predict(external_data[0])

    X_shadow = np.concatenate([
        np.concatenate([in_preds, out_preds], axis=0),
        np.concatenate([shadow_data[1], external_data[1]], axis=0)
    ], axis=1)

    y_shadow = np.concatenate([
        np.ones(SIZE),
        np.zeros(SIZE)
    ])

    # Entraîner les modèles d'attaque
    print("Entraînement des modèles d'attaque...")
    amb = AttackModelBundle(create_adaboost_attack_model, NUM_CLASSES)
    amb.fit(X_shadow, y_shadow)

    # Préparer les données de test d'attaque
    attack_test_data, real_membership_labels = prepare_attack_data(
        target_model, target_data, external_data
    )

    # Effectuer l'attaque
    attack_guesses = amb.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)

    # Calculer les précisions par classe
    class_precisions = calculate_class_precisions(
        attack_guesses, real_membership_labels, target_data, external_data
    )

    print(f"Précision de l'attaque: {attack_accuracy:.4f}")

    return attack_accuracy, class_precisions


def perform_insider_attack(target_model, target_data, insider_data, external_data):
    """
    Effectue une attaque d'insider (client participant)

    Args:
        target_model: modèle cible
        target_data: données d'entraînement du modèle cible
        insider_data: données du client attaquant
        external_data: données externes

    Returns:
        attack_accuracy: précision de l'attaque
        class_precisions: précisions par classe
    """
    print("=== ATTAQUE D'INSIDER ===")

    # Utiliser le modèle cible pour générer les données d'attaque
    print("Préparation des données d'attaque...")

    in_preds = target_model.predict(insider_data[0])
    out_preds = target_model.predict(external_data[0])

    X_shadow = np.concatenate([
        np.concatenate([in_preds, out_preds], axis=0),
        np.concatenate([insider_data[1], external_data[1]], axis=0)
    ], axis=1)

    y_shadow = np.concatenate([
        np.ones(SIZE),
        np.zeros(SIZE)
    ])

    # Entraîner les modèles d'attaque
    print("Entraînement des modèles d'attaque...")
    amb = AttackModelBundle(create_adaboost_attack_model, NUM_CLASSES)
    amb.fit(X_shadow, y_shadow)

    # Préparer les données de test d'attaque
    attack_test_data, real_membership_labels = prepare_attack_data(
        target_model, target_data, external_data
    )

    # Effectuer l'attaque
    attack_guesses = amb.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)

    # Calculer les précisions par classe
    class_precisions = calculate_class_precisions(
        attack_guesses, real_membership_labels, target_data, external_data
    )

    print(f"Précision de l'attaque: {attack_accuracy:.4f}")

    return attack_accuracy, class_precisions


def calculate_class_precisions(attack_guesses, real_labels, target_data, external_data):
    """
    Calcule les précisions d'attaque par classe

    Args:
        attack_guesses: prédictions d'attaque
        real_labels: vraies étiquettes de membership
        target_data: données cibles
        external_data: données externes

    Returns:
        class_precisions: précisions par classe
    """
    class_precisions = []

    for class_id in range(NUM_CLASSES):
        # Indices des échantillons de cette classe dans les données cibles
        target_indices = [
            i for i, label in enumerate(target_data[1].argmax(axis=1))
            if label == class_id
        ]

        # Indices des échantillons de cette classe dans les données externes
        external_indices = [
            i for i, label in enumerate(external_data[1].argmax(axis=1))
            if label == class_id
        ]

        if len(target_indices) > 0 and len(external_indices) > 0:
            # Calculer la précision pour cette classe
            target_correct = np.sum(attack_guesses[target_indices] == 1)
            external_correct = np.sum(attack_guesses[SIZE:][external_indices] == 0)

            total_class_samples = len(target_indices) + len(external_indices)
            class_precision = (target_correct + external_correct) / total_class_samples

            class_precisions.append(class_precision)
            print(f"Classe {class_id}: Précision = {class_precision:.4f}")
        else:
            class_precisions.append(0.0)
            print(f"Classe {class_id}: Pas assez d'échantillons")

    return class_precisions


def save_attack_results(attack_type, model_name, attack_accuracy, class_precisions):
    """
    Sauvegarde les résultats d'attaque

    Args:
        attack_type: type d'attaque ("outsider" ou "insider")
        model_name: nom du modèle attaqué
        attack_accuracy: précision globale de l'attaque
        class_precisions: précisions par classe
    """
    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    filename = f"MIA_{attack_type}_{model_name}_{timestamp}.txt"
    log_path = os.path.join(LOG_DIR, filename)

    try:
        with open(log_path, 'w') as log_file:
            log_file.write(f"Membership Inference Attack Results\n")
            log_file.write(f"Attack Type: {attack_type}\n")
            log_file.write(f"Target Model: {model_name}\n")
            log_file.write(f"Overall Accuracy: {attack_accuracy:.4f}\n")
            log_file.write(f"Class Precisions: {class_precisions}\n")

        print(f"Résultats d'attaque sauvegardés: {log_path}")
    except IOError:
        print("Erreur lors de la sauvegarde des résultats d'attaque")


def perform_original_mia_attack(target_model, target_data, test_data, shadow_dataset_size=1000, num_shadows=10):
    """
    Effectue l'attaque MIA originale avec plusieurs modèles shadow

    Args:
        target_model: modèle cible
        target_data: données d'entraînement du modèle cible
        test_data: données de test
        shadow_dataset_size: taille des datasets shadow
        num_shadows: nombre de modèles shadow

    Returns:
        attack_accuracy: précision de l'attaque
        class_precisions: précisions par classe
    """
    print("=== ATTAQUE MIA ORIGINALE ===")

    # Préparer les données pour l'attaquant
    attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
        test_data[0], test_data[1], test_size=0.5, random_state=42
    )

    print(f"Données attaquant: {attacker_X_train.shape}, {attacker_X_test.shape}")

    # Entraîner les modèles shadow
    print("Entraînement des modèles shadow...")
    smb = ShadowModelBundle(
        create_shadow_model,
        shadow_dataset_size=shadow_dataset_size,
        num_models=num_shadows
    )

    X_shadow, y_shadow = smb.fit_transform(
        attacker_X_train,
        attacker_y_train,
        fit_kwargs=dict(
            epochs=EPOCHS_SHADOW,
            verbose=1,
            validation_data=(attacker_X_test, attacker_y_test)
        )
    )

    # Entraîner les modèles d'attaque
    print("Entraînement des modèles d'attaque...")
    amb = AttackModelBundle(create_attack_model, NUM_CLASSES)
    amb.fit(
        X_shadow, y_shadow,
        fit_kwargs=dict(epochs=EPOCHS_ATTACK, verbose=1)
    )

    # Préparer les données de test d'attaque
    attack_test_data, real_membership_labels = prepare_attack_data(
        target_model, target_data, test_data
    )

    # Effectuer l'attaque
    attack_guesses = amb.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)

    # Calculer les précisions par classe
    class_precisions = calculate_class_precisions(
        attack_guesses, real_membership_labels, target_data, test_data
    )

    print(f"Précision de l'attaque: {attack_accuracy:.4f}")
    print(f"Rapport de classification:")
    print(classification_report(real_membership_labels, attack_guesses))

    return attack_accuracy, class_precisions