# Expériences d'Apprentissage Fédéré et Attaques d'Inférence de Membership

Ce projet implémente différentes approches d'apprentissage automatique et évalue leur vulnérabilité aux attaques d'inférence de membership (MIA) sur le dataset CIFAR-10.

## 🎯 Objectifs

1. **Comparer différentes approches d'apprentissage** :
   - Apprentissage centralisé
   - Modèles individuels par client
   - Ensemble de modèles
   - Transfert cyclique de poids
   - Apprentissage fédéré (FedAvg)

2. **Évaluer la confidentialité** :
   - Attaques d'inférence de membership
   - Attaques d'outsider vs insider
   - Métriques de confidentialité

## 📋 Prérequis

```bash
pip install tensorflow numpy scikit-learn matplotlib
```

## 🚀 Utilisation Rapide

### Option 1: Script d'exécution automatique
```bash
python run_experiments.py
```

### Option 2: Script principal avec options
```bash
# Expérience complète
python main.py

# Seulement les attaques (modèles existants)
python main.py --skip-training

# Seulement l'entraînement
python main.py --skip-attacks

# Avec visualisations
python main.py --visualize
```

## 📁 Structure du Projet

```
federated_learning_project/
├── config.py                    # Configuration et paramètres
├── data_preparation.py          # Préparation des données CIFAR-10
├── models.py                    # Définitions des modèles
├── centralized_training.py      # Entraînement centralisé
├── single_models.py             # Modèles individuels
├── ensemble_models.py           # Ensembles de modèles
├── cyclic_weight_transfer.py    # Transfert cyclique de poids
├── federated_learning.py        # Apprentissage fédéré (FedAvg)
├── membership_inference_attack.py # Attaques MIA
├── utils.py                     # Utilitaires et visualisations
├── main.py                      # Script principal
├── run_experiments.py           # Script d'exécution rapide
└── README.md                    # Documentation
```

## 🔬 Expériences Implémentées

### 1. Approches d'Apprentissage

#### Apprentissage Centralisé
- Un seul modèle entraîné sur toutes les données
- Référence pour les comparaisons de performance

#### Modèles Individuels
- Un modèle par client, entraîné seulement sur ses données
- Simule la non-collaboration entre clients

#### Ensemble de Modèles
- Combinaison des prédictions des modèles individuels
- Moyenne des probabilités de sortie

#### Transfert Cyclique de Poids
- Entraînement séquentiel sur chaque client
- Transfert des poids entre clients à chaque tour

#### Apprentissage Fédéré (FedAvg)
- Algorithme FedAvg standard
- Agrégation des gradients locaux

### 2. Attaques d'Inférence de Membership

#### Attaque d'Outsider
- Attaquant externe sans accès aux données d'entraînement
- Utilise des modèles shadow entraînés sur des données publiques

#### Attaque d'Insider
- Attaquant participant (client malveillant)
- Utilise ses propres données et l'accès au modèle

#### Attaque MIA Originale
- Implémentation de l'attaque de Shokri et al.
- Utilise plusieurs modèles shadow et modèles d'attaque par classe

## 📊 Métriques Évaluées

### Performance des Modèles
- Précision (Accuracy)
- Perte (Loss)
- Comparaison entre approches

### Confidentialité
- Précision des attaques MIA
- Précision par classe
- Avantage de l'attaque (Attack Advantage)
- Perte de confidentialité (Privacy Loss)

## 🔧 Configuration

### Paramètres par défaut (`config.py`)
```python
CLIENTS = 3              # Nombre de clients
SIZE = 5000             # Taille des données par client
EPOCHS_CENTRALIZED = 36  # Époques pour l'entraînement centralisé
EPOCHS_FEDERATED = 36   # Tours pour l'apprentissage fédéré
BATCH_SIZE = 32         # Taille des lots
```

### Personnalisation
Modifiez `config.py` pour ajuster :
- Nombre de clients
- Taille des datasets
- Paramètres d'entraînement
- Paramètres d'attaque

## 📈 Résultats et Visualisations

### Fichiers générés
- `models/`: Modèles entraînés sauvegardés
- `logs/`: Logs détaillés des expériences
- `visualizations/`: Graphiques et visualisations

### Types de visualisations
- Courbes d'entraînement
- Comparaison des performances
- Résultats des attaques par classe
- Métriques de confidentialité

## 🛡️ Analyse de Confidentialité

### Interprétation des Résultats
- **Précision d'attaque > 70%** : Risque élevé
- **Précision d'attaque 60-70%** : Risque modéré
- **Précision d'attaque 55-60%** : Risque faible
- **Précision d'attaque < 55%** : Risque minimal

### Recommandations
- Utiliser des techniques de préservation de la confidentialité
- Differential Privacy
- Agrégation sécurisée
- Bruit ajouté aux gradients

## 🔍 Détails Techniques

### Architecture des Modèles
```python
# Modèle CNN pour CIFAR-10
Conv2D(32, (5,5)) -> MaxPooling2D -> 
Conv2D(64, (5,5)) -> MaxPooling2D -> 
Flatten -> Dense(512) -> Dense(10)
```

### Algorithme FedAvg
1. Initialiser le modèle global
2. Pour chaque tour de communication :
   - Distribuer le modèle global aux clients
   - Entraînement local sur chaque client
   - Calculer les deltas de poids
   - Agréger les deltas (moyenne)
   - Mettre à jour le modèle global

### Attaques MIA
1. **Entraînement des modèles shadow**
   - Modèles similaires au modèle cible
   - Entraînés sur des données publiques
   
2. **Génération des données d'attaque**
   - Prédictions sur données "in" vs "out"
   - Étiquettes de membership
   
3. **Entraînement des modèles d'attaque**
   - Un modèle par classe
   - Classification binaire : membre/non-membre

## 🐛 Dépannage

### Problèmes courants
1. **Erreur de mémoire GPU** : Réduire `BATCH_SIZE` dans `config.py`
2. **Temps d'exécution long** : Utiliser l'expérience rapide
3. **Modèles manquants** : Vérifier le dossier `models/`

### Logs de débogage
Consultez les fichiers dans `logs/` pour des informations détaillées sur les erreurs.

## 📚 Références

1. Shokri, R., et al. "Membership inference attacks against machine learning models." 2017 IEEE Symposium on Security and Privacy.

2. McMahan, B., et al. "Communication-efficient learning of deep networks from decentralized data." AISTATS 2017.

3. Fredrikson, M., et al. "Privacy in pharmacogenetics: An end-to-end case study of personalized warfarin dosing." USENIX Security 2014.

## 📄 Licence

Ce projet est à des fins éducatives et de recherche. Veuillez citer les travaux originaux lors de l'utilisation.

## 🤝 Contribution

Les contributions sont les bienvenues ! Veuillez créer une issue ou une pull request pour :
- Corrections de bugs
- Nouvelles fonctionnalités
- Améliorations de la documentation
- Optimisations de performance

## 📧 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur le projet.