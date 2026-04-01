"""
train_model.py - Train the NLP intent classification model using TensorFlow/Keras
"""

import json
import pickle
import numpy as np
import random
import nltk
import os
import logging
from datetime import datetime

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'intents.json')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

lemmatizer = WordNetLemmatizer()

def load_intents():
    """Load training data from intents.json"""
    with open(DATA_PATH, 'r') as f:
        return json.load(f)

def preprocess_data(intents_data):
    """Tokenize, lemmatize and prepare training data"""
    words = []
    classes = []
    documents = []
    ignore_letters = ['?', '!', '.', ',', "'", '"', '-', '(', ')']

    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    logger.info(f"Total unique words: {len(words)}")
    logger.info(f"Total intent classes: {len(classes)}")
    logger.info(f"Total training documents: {len(documents)}")
    logger.info(f"Intent classes: {classes}")

    return words, classes, documents

def create_training_data(words, classes, documents):
    """Create bag-of-words training data"""
    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]

        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))

    return train_x, train_y

def build_model(input_shape, output_shape):
    """Build a deep neural network for intent classification"""
    model = Sequential([
        Dense(256, input_shape=(input_shape,), activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(output_shape, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    model.summary()
    return model

def train_model():
    """Main training pipeline"""
    logger.info("=" * 60)
    logger.info("Starting Model Training Pipeline")
    logger.info("=" * 60)

    # Load and preprocess data
    intents_data = load_intents()
    words, classes, documents = preprocess_data(intents_data)
    train_x, train_y = create_training_data(words, classes, documents)

    logger.info(f"Training data shape: X={train_x.shape}, Y={train_y.shape}")

    # Build model
    model = build_model(len(train_x[0]), len(train_y[0]))

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'best_model.h5'),
            monitor='loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train
    logger.info("Training model...")
    history = model.fit(
        train_x,
        train_y,
        epochs=300,
        batch_size=8,
        verbose=1,
        callbacks=callbacks
    )

    # Save artifacts
    model_path = os.path.join(MODEL_DIR, 'chatbot_model.h5')
    model.save(model_path)

    words_path = os.path.join(MODEL_DIR, 'words.pkl')
    with open(words_path, 'wb') as f:
        pickle.dump(words, f)

    classes_path = os.path.join(MODEL_DIR, 'classes.pkl')
    with open(classes_path, 'wb') as f:
        pickle.dump(classes, f)

    # Save training metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'num_words': len(words),
        'num_classes': len(classes),
        'num_documents': len(documents),
        'classes': classes,
        'final_accuracy': float(history.history['accuracy'][-1]),
        'final_loss': float(history.history['loss'][-1]),
        'epochs_trained': len(history.history['loss'])
    }

    metadata_path = os.path.join(MODEL_DIR, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save training history
    history_data = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'loss': [float(x) for x in history.history['loss']]
    }
    history_path = os.path.join(MODEL_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)

    final_acc = history.history['accuracy'][-1]
    final_loss = history.history['loss'][-1]

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Final Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    logger.info(f"Final Loss: {final_loss:.4f}")
    logger.info(f"Model saved to: {model_path}")
    logger.info("=" * 60)

    return model, words, classes, history

if __name__ == '__main__':
    train_model()
