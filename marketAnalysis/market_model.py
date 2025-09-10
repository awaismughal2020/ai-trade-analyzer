"""
Market Analysis LSTM Model
Handles model architecture, training, evaluation, and prediction
"""

import numpy as np
import pandas as pd
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MarketAnalysisLSTM:
    def __init__(self, sequence_length=None, n_features=9, n_classes=3):
        """
        Initialize LSTM model for market analysis

        Args:
            sequence_length (int): Length of input sequences
            n_features (int): Number of features per timestep
            n_classes (int): Number of output classes (SELL, HOLD, BUY)
        """
        self.sequence_length = sequence_length or int(os.getenv('SEQUENCE_LENGTH', 20))
        self.n_features = n_features
        self.n_classes = n_classes
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None
        self.is_trained = False

        # Training parameters from environment
        self.batch_size = int(os.getenv('BATCH_SIZE', 32))
        self.epochs = int(os.getenv('EPOCHS', 30))
        self.validation_split = float(os.getenv('VALIDATION_SPLIT', 0.2))

        # Model save path
        self.model_save_path = os.getenv('MODEL_SAVE_PATH', './models/meme_coin_market_model')

        print(f"Initialized MarketAnalysisLSTM:")
        print(f"- Sequence length: {self.sequence_length}")
        print(f"- Features: {self.n_features}")
        print(f"- Classes: {self.n_classes}")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Max epochs: {self.epochs}")

    def build_model(self, lstm_units=[64, 32], dropout_rate=0.2, learning_rate=0.001):
        """
        Build LSTM model architecture

        Args:
            lstm_units (list): Number of units in each LSTM layer
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer

        Returns:
            tf.keras.Model: Compiled model
        """
        print("Building LSTM model architecture...")

        model = Sequential()

        # First LSTM layer
        model.add(LSTM(
            lstm_units[0],
            return_sequences=True,
            input_shape=(self.sequence_length, self.n_features),
            name='lstm_1'
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        # Second LSTM layer
        model.add(LSTM(
            lstm_units[1],
            return_sequences=False,
            name='lstm_2'
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        # Dense layers
        model.add(Dense(32, activation='relu', name='dense_1'))
        model.add(Dropout(dropout_rate / 2))

        model.add(Dense(16, activation='relu', name='dense_2'))
        model.add(Dropout(dropout_rate / 2))

        # Output layer
        model.add(Dense(self.n_classes, activation='softmax', name='output'))

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model

        # Print model summary
        print(f"Model built successfully:")
        print(f"- Total parameters: {model.count_params():,}")
        print(f"- LSTM units: {lstm_units}")
        print(f"- Dropout rate: {dropout_rate}")
        print(f"- Learning rate: {learning_rate}")

        return model

    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """
        Scale features and split data for training

        Args:
            X (np.array): Input sequences
            y (np.array): Target labels
            test_size (float): Proportion of test set
            random_state (int): Random seed for reproducibility

        Returns:
            tuple: (X_train, X_test, y_train, y_test) scaled and split
        """
        print("Preparing data for training...")

        # Validate input shapes
        if len(X.shape) != 3:
            raise ValueError(f"X must be 3D array, got shape {X.shape}")

        if X.shape[1] != self.sequence_length:
            raise ValueError(f"Sequence length mismatch: expected {self.sequence_length}, got {X.shape[1]}")

        if X.shape[2] != self.n_features:
            print(f"Warning: Feature count mismatch. Expected {self.n_features}, got {X.shape[2]}")
            self.n_features = X.shape[2]

        # Reshape for scaling (samples * timesteps, features)
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])

        # Fit scaler and transform
        print("Fitting scaler on training data...")
        X_scaled = self.scaler.fit_transform(X_reshaped)

        # Reshape back to original shape
        X_scaled = X_scaled.reshape(original_shape)

        # Time-based split (more realistic for time series)
        split_idx = int(len(X_scaled) * (1 - test_size))

        X_train = X_scaled[:split_idx]
        X_test = X_scaled[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        print(f"Data preparation complete:")
        print(f"- Training set: {len(X_train):,} samples")
        print(f"- Test set: {len(X_test):,} samples")
        print(f"- Feature scaling: Min-Max normalized")
        print(f"- Split method: Time-based ({test_size * 100:.0f}% test)")

        # Check class distribution
        train_unique, train_counts = np.unique(y_train, return_counts=True)
        test_unique, test_counts = np.unique(y_test, return_counts=True)

        print(f"\nClass distribution:")
        print("Training set:")
        for i, (cls, count) in enumerate(zip(train_unique, train_counts)):
            cls_name = ['SELL', 'HOLD', 'BUY'][cls]
            print(f"  {cls_name}: {count:,} ({count / len(y_train) * 100:.1f}%)")

        print("Test set:")
        for i, (cls, count) in enumerate(zip(test_unique, test_counts)):
            cls_name = ['SELL', 'HOLD', 'BUY'][cls]
            print(f"  {cls_name}: {count:,} ({count / len(y_test) * 100:.1f}%)")

        return X_train, X_test, y_train, y_test

    def create_callbacks(self, patience=10, min_lr=1e-7):
        """
        Create training callbacks for better training

        Args:
            patience (int): Early stopping patience
            min_lr (float): Minimum learning rate

        Returns:
            list: List of Keras callbacks
        """
        callbacks = []

        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=min_lr,
            verbose=1
        )
        callbacks.append(reduce_lr)

        # Model checkpoint
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        checkpoint = ModelCheckpoint(
            f"{self.model_save_path}_best.h5",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)

        return callbacks

    def train(self, X_train, y_train, X_test, y_test, verbose=1):
        """
        Train the LSTM model

        Args:
            X_train, y_train: Training data
            X_test, y_test: Validation data
            verbose (int): Verbosity level

        Returns:
            tf.keras.History: Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print("Starting model training...")
        print(f"Training configuration:")
        print(f"- Epochs: {self.epochs}")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Validation split: {self.validation_split}")

        # Create callbacks
        callbacks = self.create_callbacks()

        # Train model
        start_time = datetime.now()

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False  # Don't shuffle time series data
        )

        end_time = datetime.now()
        training_time = end_time - start_time

        self.is_trained = True

        print(f"\nTraining completed in {training_time}")
        print(f"Best validation loss: {min(self.history.history['val_loss']):.4f}")
        print(f"Best validation accuracy: {max(self.history.history['val_accuracy']):.4f}")

        return self.history

    def evaluate(self, X_test, y_test, plot_results=True):
        """
        Evaluate model performance

        Args:
            X_test, y_test: Test data
            plot_results (bool): Whether to plot confusion matrix

        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        print("Evaluating model performance...")

        # Get predictions
        predictions = self.model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predicted_classes)

        # Classification report
        class_names = ['SELL', 'HOLD', 'BUY']
        report = classification_report(
            y_test, predicted_classes,
            target_names=class_names,
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, predicted_classes)

        print(f"\n=== MODEL EVALUATION RESULTS ===")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"\nPer-class Performance:")
        for i, class_name in enumerate(class_names):
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            support = report[class_name]['support']
            print(f"{class_name:>4}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Support={support}")

        print(f"\nConfusion Matrix:")
        print("Predicted →")
        print("Actual ↓   ", end="")
        for name in class_names:
            print(f"{name:>6}", end="")
        print()

        for i, name in enumerate(class_names):
            print(f"{name:>8}: ", end="")
            for j in range(len(class_names)):
                print(f"{cm[i, j]:>6}", end="")
            print()

        # Plot confusion matrix if requested
        if plot_results:
            self.plot_confusion_matrix(cm, class_names)

        # Calculate business metrics
        business_metrics = self._calculate_business_metrics(y_test, predicted_classes, predictions)

        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'business_metrics': business_metrics
        }

        return results

    def _calculate_business_metrics(self, y_true, y_pred, y_prob):
        """Calculate trading-specific performance metrics"""

        # Simulate trading performance
        # Assume we take positions based on BUY signals
        buy_signals = (y_pred == 2)
        actual_returns = (y_true == 2).astype(float) - (y_true == 0).astype(float)

        if buy_signals.sum() > 0:
            # Precision for BUY signals (how often BUY signals are correct)
            buy_precision = (y_true[buy_signals] == 2).mean()

            # Average return when following BUY signals
            avg_return_on_buys = actual_returns[buy_signals].mean()

            # Hit rate (percentage of profitable BUY signals)
            hit_rate = (actual_returns[buy_signals] > 0).mean()
        else:
            buy_precision = 0
            avg_return_on_buys = 0
            hit_rate = 0

        # Confidence analysis
        high_confidence_threshold = 0.7
        high_conf_mask = y_prob.max(axis=1) > high_confidence_threshold

        if high_conf_mask.sum() > 0:
            high_conf_accuracy = (y_true[high_conf_mask] == y_pred[high_conf_mask]).mean()
        else:
            high_conf_accuracy = 0

        business_metrics = {
            'buy_signal_precision': buy_precision,
            'average_return_on_buys': avg_return_on_buys,
            'hit_rate': hit_rate,
            'high_confidence_accuracy': high_conf_accuracy,
            'high_confidence_percentage': high_conf_mask.mean()
        }

        print(f"\n=== BUSINESS METRICS ===")
        print(f"BUY Signal Precision: {buy_precision:.3f}")
        print(f"Hit Rate (Profitable BUYs): {hit_rate:.3f}")
        print(f"Avg Return on BUY Signals: {avg_return_on_buys:.3f}")
        print(f"High Confidence Accuracy: {high_conf_accuracy:.3f}")
        print(f"High Confidence Predictions: {high_conf_mask.mean():.1%}")

        return business_metrics

    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()

            # Save plot
            os.makedirs("plots", exist_ok=True)
            plt.savefig("plots/confusion_matrix.png", dpi=300, bbox_inches='tight')
            print("Confusion matrix plot saved to plots/confusion_matrix.png")
            plt.show()
        except ImportError:
            print("Matplotlib/Seaborn not available for plotting")

    def predict(self, sequence):
        """
        Make prediction on new sequence

        Args:
            sequence (np.array): Input sequence to predict

        Returns:
            dict: Prediction results with signal, confidence, and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Ensure correct shape
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])

        # Scale the sequence
        original_shape = sequence.shape
        sequence_reshaped = sequence.reshape(-1, sequence.shape[-1])
        sequence_scaled = self.scaler.transform(sequence_reshaped)
        sequence_scaled = sequence_scaled.reshape(original_shape)

        # Get prediction
        prediction = self.model.predict(sequence_scaled, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])

        class_names = ['SELL', 'HOLD', 'BUY']

        result = {
            'signal': class_names[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                'SELL': float(prediction[0][0]),
                'HOLD': float(prediction[0][1]),
                'BUY': float(prediction[0][2])
            },
            'timestamp': datetime.now().isoformat()
        }

        return result

    def save_model(self, filepath=None):
        """
        Save trained model and scaler

        Args:
            filepath (str): Custom filepath (optional)
        """
        if not self.is_trained:
            raise ValueError("No trained model to save.")

        save_path = filepath or self.model_save_path

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save model
        model_path = f"{save_path}_model.h5"
        self.model.save(model_path)

        # Save scaler
        scaler_path = f"{save_path}_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)

        # Save model metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'training_date': datetime.now().isoformat(),
            'model_version': '1.0'
        }

        metadata_path = f"{save_path}_metadata.pkl"
        joblib.dump(metadata, metadata_path)

        print(f"Model saved successfully:")
        print(f"- Model: {model_path}")
        print(f"- Scaler: {scaler_path}")
        print(f"- Metadata: {metadata_path}")

        return save_path

    def load_model(self, filepath=None):
        """
        Load trained model and scaler

        Args:
            filepath (str): Path to saved model (optional)
        """
        load_path = filepath or self.model_save_path

        model_path = f"{load_path}_model.h5"
        scaler_path = f"{load_path}_scaler.pkl"
        metadata_path = f"{load_path}_metadata.pkl"

        try:
            # Load model
            self.model = load_model(model_path)

            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Load scaler
            self.scaler = joblib.load(scaler_path)

            # Load metadata
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.sequence_length = metadata['sequence_length']
                self.n_features = metadata['n_features']
                self.n_classes = metadata['n_classes']
                print(f"Model loaded from {metadata['training_date']}")

            self.is_trained = True

            print(f"Model loaded successfully:")
            print(f"- Sequence length: {self.sequence_length}")
            print(f"- Features: {self.n_features}")
            print(f"- Classes: {self.n_classes}")

            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False
