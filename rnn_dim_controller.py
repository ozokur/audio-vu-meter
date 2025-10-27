#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN-based DMX Dimmer Controller
Uses LSTM to predict optimal dimming levels based on audio patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import time
from collections import deque
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger("rnn_dim_controller")


class AudioSequenceDataset:
    """Dataset for audio sequence data and dimmer values"""
    
    def __init__(self, sequence_length: int = 50, max_samples: int = 10000):
        self.sequence_length = sequence_length
        self.max_samples = max_samples
        self.audio_sequences = deque(maxlen=max_samples)
        self.dimmer_sequences = deque(maxlen=max_samples)
        self.timestamps = deque(maxlen=max_samples)
        
    def add_sample(self, audio_bands: List[float], dimmer_value: int, timestamp: float):
        """Add a new audio-dimmer sample"""
        # Normalize audio bands to 0-1 range
        normalized_bands = [max(0.0, min(1.0, band)) for band in audio_bands]
        
        self.audio_sequences.append(normalized_bands)
        self.dimmer_sequences.append(dimmer_value / 255.0)  # Normalize to 0-1
        self.timestamps.append(timestamp)
        
        # Debug: Her 10 sample'da bir log
        if len(self.audio_sequences) % 10 == 0:
            print(f"RNN Data: {len(self.audio_sequences)} samples collected")
        
    def get_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get training sequences"""
        if len(self.audio_sequences) < self.sequence_length:
            print(f"Not enough data: {len(self.audio_sequences)} samples, need {self.sequence_length}")
            return np.array([]), np.array([])
            
        # Convert to numpy arrays
        audio_data = np.array(list(self.audio_sequences))
        dimmer_data = np.array(list(self.dimmer_sequences))
        
        # Create sequences
        X, y = [], []
        for i in range(len(audio_data) - self.sequence_length + 1):
            X.append(audio_data[i:i + self.sequence_length])
            y.append(dimmer_data[i + self.sequence_length - 1])  # Predict next value
            
        return np.array(X), np.array(y)
    
    def save_to_file(self, filename: str):
        """Save dataset to file"""
        data = {
            'audio_sequences': list(self.audio_sequences),
            'dimmer_sequences': list(self.dimmer_sequences),
            'timestamps': list(self.timestamps),
            'sequence_length': self.sequence_length
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    def load_from_file(self, filename: str):
        """Load dataset from file"""
        if not os.path.exists(filename):
            return
            
        with open(filename, 'r') as f:
            data = json.load(f)
            
        self.audio_sequences = deque(data['audio_sequences'], maxlen=self.max_samples)
        self.dimmer_sequences = deque(data['dimmer_sequences'], maxlen=self.max_samples)
        self.timestamps = deque(data['timestamps'], maxlen=self.max_samples)
        self.sequence_length = data.get('sequence_length', self.sequence_length)


class RNN_DimController(nn.Module):
    """LSTM-based RNN for dimmer control prediction"""
    
    def __init__(self, input_size: int = 6, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2, output_size: int = 1):
        super(RNN_DimController, self).__init__()
        
        self.input_size = input_size  # 6 audio bands (Llow, Lmid, Lhigh, Rlow, Rmid, Rhigh)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.activation = nn.Sigmoid()  # Output 0-1 range
        
    def forward(self, x):
        """Forward pass through the network"""
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output from LSTM
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Fully connected layers
        x = torch.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        
        return x
    
    def predict_single(self, audio_sequence: np.ndarray) -> float:
        """Predict dimmer value for a single sequence"""
        self.eval()
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            x = torch.FloatTensor(audio_sequence).unsqueeze(0)  # (1, seq_len, input_size)
            
            # Predict
            output = self.forward(x)
            return output.item()


class RNNDimController:
    """Main RNN-based dimmer controller class"""
    
    def __init__(self, model_path: str = "rnn_dim_model.pth", 
                 data_path: str = "rnn_training_data.json",
                 sequence_length: int = 10):  # Reduced from 50 to 10 for faster training
        self.model_path = model_path
        self.data_path = data_path
        self.sequence_length = sequence_length
        
        # Initialize model
        self.model = RNN_DimController(input_size=6, hidden_size=64, num_layers=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Dataset
        self.dataset = AudioSequenceDataset(sequence_length=sequence_length)
        
        # Training state
        self.is_training = False
        self.training_losses = []
        
        # Load existing model and data if available
        self.load_model()
        self.dataset.load_from_file(data_path)
        
        logger.info(f"RNN Dim Controller initialized (sequence_length={sequence_length})")
    
    def add_audio_sample(self, audio_bands: List[float], current_dimmer: int):
        """Add audio sample for training data collection"""
        timestamp = time.time()
        self.dataset.add_sample(audio_bands, current_dimmer, timestamp)
    
    def predict_dimmer(self, audio_bands: List[float]) -> int:
        """Predict optimal dimmer value based on audio bands"""
        if len(self.dataset.audio_sequences) < self.sequence_length:
            # Not enough data, use simple heuristic
            return self._heuristic_dimmer(audio_bands)
        
        # Get recent audio sequence
        recent_audio = list(self.dataset.audio_sequences)[-self.sequence_length:]
        recent_audio = np.array(recent_audio)
        
        # Predict using RNN
        predicted_value = self.model.predict_single(recent_audio)
        
        # Convert back to 0-255 range
        dimmer_value = int(predicted_value * 255)
        return max(0, min(255, dimmer_value))
    
    def _heuristic_dimmer(self, audio_bands: List[float]) -> int:
        """Fallback heuristic when RNN is not trained"""
        # Simple heuristic: use Llow + Rlow average
        llow, lmid, lhigh, rlow, rmid, rhigh = audio_bands
        avg_low = (llow + rlow) / 2.0
        return int(avg_low * 255)
    
    def train_model(self, epochs: int = 100, batch_size: int = 32):
        """Train the RNN model"""
        X, y = self.dataset.get_sequences()
        
        if len(X) == 0:
            logger.warning("No training data available")
            return
        
        logger.info(f"Training RNN with {len(X)} sequences")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.is_training = True
        self.training_losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.training_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        self.is_training = False
        self.save_model()
        logger.info("RNN training completed")
    
    def train_model_with_progress(self, epochs: int = 100, batch_size: int = 32, progress_callback=None):
        """Train the RNN model with progress callback"""
        X, y = self.dataset.get_sequences()
        
        if len(X) == 0:
            logger.warning("No training data available")
            return False
        
        logger.info(f"Training RNN with {len(X)} sequences")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.is_training = True
        self.training_losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.training_losses.append(avg_loss)
            
            # Progress callback
            if progress_callback:
                try:
                    progress_callback(epoch + 1, epochs, avg_loss)
                except Exception as e:
                    logger.debug(f"Progress callback error: {e}")
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        self.is_training = False
        self.save_model()
        logger.info("RNN training completed")
        return True
    
    def save_model(self):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
            'sequence_length': self.sequence_length
        }, self.model_path)
        
        # Also save training data
        self.dataset.save_to_file(self.data_path)
        
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a trained model"""
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_losses = checkpoint.get('training_losses', [])
            logger.info(f"Model loaded from {self.model_path}")
        else:
            logger.info("No existing model found, using untrained model")
    
    def get_training_stats(self) -> dict:
        """Get training statistics"""
        return {
            'samples_collected': len(self.dataset.audio_sequences),
            'is_training': self.is_training,
            'last_loss': self.training_losses[-1] if self.training_losses else None,
            'total_epochs': len(self.training_losses)
        }


# Global instance
rnn_controller = None

def initialize_rnn_controller():
    """Initialize global RNN controller instance"""
    global rnn_controller
    if rnn_controller is None:
        rnn_controller = RNNDimController()
    return rnn_controller

def get_rnn_controller():
    """Get global RNN controller instance"""
    return rnn_controller
