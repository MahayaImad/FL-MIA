"""
Configuration file for the federated learning experiments
"""

import datetime
import os

# Data configuration
CLIENTS = 3
SIZE = 5000
WIDTH = 32
HEIGHT = 32
CHANNELS = 3
NUM_CLASSES = 10

# Training configuration
EPOCHS_CENTRALIZED = 36
EPOCHS_FEDERATED = 36
EPOCHS_SHADOW = 12
EPOCHS_ATTACK = 12
BATCH_SIZE = 32

# Attack configuration
SHADOW_DATASET_SIZE = 1000
ATTACK_TEST_DATASET_SIZE = 5000
NUM_SHADOWS = 10

# Paths
MODEL_DIR = "models"
LOG_DIR = "logs"

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# CIFAR-10 class labels
CIFAR_CLASS_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]