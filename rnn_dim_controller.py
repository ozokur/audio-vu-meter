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
    """Dataset for audio sequence data and DMX values (Pan/Tilt/Color/Dimmer) - Simple System"""
    
    def __init__(self, sequence_length: int = 100, max_samples: int = 2000):
        self.sequence_length = sequence_length
        self.max_samples = max_samples
        
        # Simple deque system - unlimited growth
        self.audio_sequences = deque()
        self.dmx_sequences = deque()
        self.timestamps = deque()
        
    def add_sample(self, audio_bands: List[float], dmx_values: List[int], timestamp: float):
        """Add a new audio-DMX sample (pan, tilt, color, dimmer) - Simple System"""
        # Normalize audio bands to 0-1 range
        normalized_bands = [max(0.0, min(1.0, band)) for band in audio_bands]
        
        # Normalize DMX values to 0-1 range
        normalized_dmx = [val / 255.0 for val in dmx_values]  # [pan, tilt, color, dimmer]
        
        # Add to simple deque
        self.audio_sequences.append(normalized_bands)
        self.dmx_sequences.append(normalized_dmx)
        self.timestamps.append(timestamp)
        
    def get_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get training sequences from simple deque"""
        if len(self.audio_sequences) < self.sequence_length:
            print(f"Not enough data: {len(self.audio_sequences)} samples, need {self.sequence_length}")
            return np.array([]), np.array([])
        
        # Convert to numpy arrays
        audio_data = np.array(list(self.audio_sequences))
        dmx_data = np.array(list(self.dmx_sequences))
        
        # Create sequences
        X, y = [], []
        for i in range(len(audio_data) - self.sequence_length + 1):
            X.append(audio_data[i:i + self.sequence_length])
            y.append(dmx_data[i + self.sequence_length - 1])  # Predict next DMX values
            
        return np.array(X), np.array(y)
    
    def save_to_file(self, filename: str):
        """Save dataset to file - simple system"""
        try:
            data = {
                'audio_sequences': list(self.audio_sequences),
                'dmx_sequences': list(self.dmx_sequences),
                'timestamps': list(self.timestamps),
                'sequence_length': self.sequence_length
            }
            
            # Direkt dosyaya yaz (temp dosya yok)
            with open(filename, 'w') as f:
                json.dump(data, f)  # Compact format for speed
            
        except Exception as e:
            # Debug disabled
            pass
    
    def load_from_file(self, filename: str):
        """Load dataset from file with error handling - Simple System"""
        if not os.path.exists(filename):
            return
            
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Check for new simple format
            if 'audio_sequences' in data:
                # New simple format
                self.audio_sequences = deque(data['audio_sequences'])
                self.dmx_sequences = deque(data['dmx_sequences'])
                self.timestamps = deque(data['timestamps'])
                self.sequence_length = data.get('sequence_length', self.sequence_length)
                print(f"Loaded simple system: {len(self.audio_sequences)} samples")
            elif 'buffer1_audio' in data:
                # Old dual buffer format - convert to simple
                print("Converting from dual buffer to simple system...")
                buffer1_audio = data.get('buffer1_audio', [])
                buffer1_dmx = data.get('buffer1_dmx', [])
                buffer1_timestamps = data.get('buffer1_timestamps', [])
                buffer2_audio = data.get('buffer2_audio', [])
                buffer2_dmx = data.get('buffer2_dmx', [])
                buffer2_timestamps = data.get('buffer2_timestamps', [])
                
                # Combine both buffers
                self.audio_sequences = deque(buffer1_audio + buffer2_audio)
                self.dmx_sequences = deque(buffer1_dmx + buffer2_dmx)
                self.timestamps = deque(buffer1_timestamps + buffer2_timestamps)
                self.sequence_length = data.get('sequence_length', self.sequence_length)
                print(f"Converted to simple system: {len(self.audio_sequences)} samples")
            else:
                # Very old format - backward compatibility
                print("Loading very old format...")
                self.audio_sequences = deque(data.get('audio_sequences', []))
                self.dmx_sequences = deque(data.get('dmx_sequences', []))
                self.timestamps = deque(data.get('timestamps', []))
                self.sequence_length = data.get('sequence_length', self.sequence_length)
            
        except json.JSONDecodeError as e:
            # Debug disabled
            # Bozuk dosyayı sil ve yeniden başla
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            # Debug disabled
            pass


class RNN_DimController(nn.Module):
    """LSTM-based RNN for multi-channel DMX control prediction (Pan/Tilt/Dimmer)"""
    
    def __init__(self, input_size: int = 6, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2, output_size: int = 4):
        super(RNN_DimController, self).__init__()
        
        self.input_size = input_size  # 6 audio bands (Llow, Lmid, Lhigh, Rlow, Rmid, Rhigh)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size  # 4 outputs: Pan, Tilt, Color, Dimmer
        
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
    
    def predict_single(self, audio_sequence: np.ndarray, device) -> List[float]:
        """Predict DMX values for a single sequence (Pan, Tilt, Color, Dimmer)"""
        self.eval()
        with torch.no_grad():
            # Convert to tensor and add batch dimension, move to device
            x = torch.FloatTensor(audio_sequence).unsqueeze(0).to(device)  # (1, seq_len, input_size)
            
            # Predict
            output = self.forward(x)
            return output[0].cpu().tolist()  # [pan, tilt, color, dimmer] normalized 0-1


class RNNDimController:
    """Main RNN-based multi-channel DMX controller class"""
    
    def __init__(self, model_path: str = "rnn_dim_model.pth", 
                 data_path: str = "rnn_training_data.json",
                 sequence_length: int = 10,  # Reduced from 50 to 10 for faster training
                 auto_retrain_interval: int = 100,
                 pan_offset: int = 0,  # Pan offset adjustment
                 tilt_offset: int = 0):  # Tilt offset adjustment
        self.model_path = model_path
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.auto_retrain_interval = auto_retrain_interval
        self.samples_since_last_train = 0
        self.last_training_time = None
        self.pan_offset = pan_offset
        self.tilt_offset = tilt_offset
        
        # Check for CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = RNN_DimController(input_size=8, hidden_size=128, num_layers=3, output_size=4)
        self.model.to(self.device)  # Move model to device
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)  # Increased learning rate
        self.criterion = nn.MSELoss()
        
        # Dataset
        self.dataset = AudioSequenceDataset(sequence_length=sequence_length)
        
        # Training state
        self.is_training = False
        self.training_losses = []
        
        # Load existing model and data if available
        self.load_model()
        self.dataset.load_from_file(data_path)
        
        logger.info(f"RNN Multi-Channel Controller initialized (sequence_length={sequence_length})")
    
    def add_audio_sample(self, audio_bands: List[float], pan: int, tilt: int, color: int, dimmer: int):
        """Add audio sample for training data collection"""
        timestamp = time.time()
        dmx_values = [pan, tilt, color, dimmer]
        self.dataset.add_sample(audio_bands, dmx_values, timestamp)
        
        # Auto-retrain disabled - manual training only
        self.samples_since_last_train += 1
        # Sample progress mesajını kaldırdık
    
    def predict_dmx_channels(self, audio_bands: List[float]) -> dict:
        """Predict optimal DMX values based on audio bands"""
        # Throttle debug messages (2 seconds)
        current_time = time.time()
        if not hasattr(self, '_last_debug_time'):
            self._last_debug_time = 0
        
        # Check if we have enough data, if not create a sequence from current audio
        if len(self.dataset.audio_sequences) < self.sequence_length:
            # Create a sequence by repeating current audio data with variations
            if len(audio_bands) >= 8:
                # Create a more dynamic sequence with audio variations
                base_audio = np.array(audio_bands)
                # Add some variation to make it more responsive to music
                variations = []
                for i in range(self.sequence_length):
                    # Add small random variations to make it more dynamic
                    variation = base_audio + np.random.normal(0, 0.01, len(base_audio))
                    variation = np.clip(variation, 0, 1)  # Keep in 0-1 range
                    variations.append(variation)
                recent_audio = np.array(variations)
                
                if current_time - self._last_debug_time > 2.0:
                    print(f"✅ RNN Prediction: Model kullanılıyor (veri yok, mevcut audio ile)")
                    print(f"🔍 Audio bands: {[f'{v:.3f}' for v in audio_bands]}")
                    self._last_debug_time = current_time
            else:
                # Fallback to heuristic if audio data is incomplete
                if current_time - self._last_debug_time > 2.0:
                    print(f"⚠️ RNN Fallback: Eksik audio verisi")
                    self._last_debug_time = current_time
                return self._heuristic_dmx_channels(audio_bands)
        else:
            # Get recent audio sequence
            recent_audio = list(self.dataset.audio_sequences)[-self.sequence_length:]
            recent_audio = np.array(recent_audio)
            
            # Debug: RNN prediction being used
            if current_time - self._last_debug_time > 2.0:
                beat_intensity = audio_bands[6] if len(audio_bands) > 6 else 0
                color_mode = audio_bands[7] if len(audio_bands) > 7 else 0
                print(f"✅ RNN Prediction: {len(self.dataset.audio_sequences)} samples, beat_intensity={beat_intensity:.2f}, color_mode={int(color_mode)}")
                self._last_debug_time = current_time
        
        # Predict using RNN
        predicted_values = self.model.predict_single(recent_audio, self.device)  # [pan, tilt, color, dimmer]
        
        # Convert back to 0-255 range with music-responsive dynamics
        # Use current audio data to enhance RNN predictions
        if len(audio_bands) >= 8:
            llow, lmid, lhigh, rlow, rmid, rhigh, beat_intensity, color_mode = audio_bands[:8]
            
            # Base RNN predictions
            pan_raw = predicted_values[0] * 255
            tilt_raw = predicted_values[1] * 255
            color_raw = predicted_values[2] * 255
            dimmer_raw = predicted_values[3] * 255
            
            # Enhance with current audio data for music responsiveness - MORE AGGRESSIVE
            audio_pan_influence = (llow - rlow) * 150  # Left-right balance (increased more)
            audio_tilt_influence = (lmid - 0.5) * 120  # Mid frequency influence (increased)
            
            # Frequency-based color calculation (more aggressive)
            # Low frequencies (bass) -> Red/Orange
            # Mid frequencies -> Yellow/Green  
            # High frequencies (treble) -> Blue/Purple
            low_freq = (llow + rlow) / 2.0
            mid_freq = (lmid + rmid) / 2.0
            high_freq = (lhigh + rhigh) / 2.0
            
            # Determine dominant frequency and calculate color
            if high_freq > mid_freq and high_freq > low_freq:
                # High frequency dominant - Blue/Purple range (70-120)
                audio_color_influence = 70 + (high_freq * 50)  # 70-120
            elif mid_freq > low_freq:
                # Mid frequency dominant - Yellow/Green range (35-70)
                audio_color_influence = 35 + (mid_freq * 35)  # 35-70
            else:
                # Low frequency dominant - Red/Orange range (15-35)
                audio_color_influence = 15 + (low_freq * 20)  # 15-35
            
            audio_dimmer_influence = beat_intensity * 200  # Beat intensity for brightness
            
            # Additional movement boost based on beat intensity
            beat_pan_boost = beat_intensity * 80  # Beat makes pan more dynamic (increased)
            beat_tilt_boost = beat_intensity * 60  # Beat makes tilt more dynamic
            
            # Additional dimmer boost based on overall audio level
            total_audio = (llow + lmid + lhigh + rlow + rmid + rhigh) / 6.0
            audio_level_boost = total_audio * 100  # Additional brightness boost
            
            # Beat dimmer boost - much more aggressive
            beat_dimmer_boost = beat_intensity * 100  # Beat makes dimmer much brighter
            
            # Combine RNN with audio data + beat boost + offsets
            pan_value = int(pan_raw + audio_pan_influence + beat_pan_boost + self.pan_offset)
            tilt_value = int(tilt_raw + audio_tilt_influence + beat_tilt_boost + self.tilt_offset)
            
            # Use frequency-based color instead of RNN color prediction
            color_value = int(audio_color_influence)  # Pure frequency-based color
            
            # Beat anında dimmer 255 olsun - daha agresif threshold
            if beat_intensity > 0.1:  # Beat threshold düşürüldü (0.3 -> 0.1)
                dimmer_value = 255  # Maximum brightness on beat
                # Debug disabled - Beat detection
                # print(f"🔥 BEAT DETECTED! Dimmer=255, beat_intensity={beat_intensity:.3f}")
            else:
                dimmer_value = int(dimmer_raw + audio_dimmer_influence + audio_level_boost + beat_dimmer_boost)
                # Debug disabled - No beat
                # if current_time - self._last_debug_time > 2.0:
                #     print(f"🎵 No beat: Dimmer={dimmer_value}, beat_intensity={beat_intensity:.3f}")
                #     self._last_debug_time = current_time
        else:
            # Fallback to enhanced RNN only
            pan_value = int(64 + (predicted_values[0] * 255 - 127) * 1.5)
            tilt_value = int(64 + (predicted_values[1] * 255 - 127) * 1.5)
            color_value = int(predicted_values[2] * 255 * 1.2)
            dimmer_value = int(predicted_values[3] * 255 * 1.5)
        
        result = {
            'pan': max(0, min(255, pan_value)),
            'tilt': max(0, min(255, tilt_value)),
            'color': max(0, min(255, color_value)),
            'dimmer': max(0, min(255, dimmer_value))
        }
        
        # Debug enabled for troubleshooting
        if current_time - self._last_debug_time > 2.0:
            print(f"🔍 RNN Raw: {[f'{v:.3f}' for v in predicted_values]}")
            print(f"🎯 RNN Final: Pan={result['pan']}, Tilt={result['tilt']}, Color={result['color']}, Dimmer={result['dimmer']}")
            self._last_debug_time = current_time
        
        return result
    
    def _heuristic_dmx_channels(self, audio_bands: List[float]) -> dict:
        """Enhanced heuristic fallback with beat intensity and color mode focus"""
        # Extract enhanced audio features (8 features now)
        if len(audio_bands) >= 8:
            llow, lmid, lhigh, rlow, rmid, rhigh, beat_intensity, color_mode = audio_bands[:8]
        else:
            # Fallback for old format
            llow, lmid, lhigh, rlow, rmid, rhigh = audio_bands[:6]
            beat_intensity = (llow + lmid + rlow + rmid) / 4.0
            color_mode = 0  # Default: Frequency-based
        
        # Beat-driven pan movement (more dynamic)
        pan_value = int(128 + (llow - rlow) * 80 + beat_intensity * 40)
        pan_value = max(0, min(255, pan_value))
        
        # Beat-driven tilt movement (more responsive to intensity)
        tilt_value = int(128 + (lmid - 0.5) * 120 + beat_intensity * 60)
        tilt_value = max(0, min(255, tilt_value))
        
        # Color mode-aware color calculation
        if color_mode == 1:  # Smart Random mode
            # More dynamic color changes based on beat intensity
            color_value = int((beat_intensity * 200) + (lhigh * 55))
            color_value = max(0, min(255, color_value))
        else:  # Frequency-based mode
            # Traditional frequency-based color with beat enhancement
            color_value = int((lhigh * 150) + (beat_intensity * 105))
            color_value = max(0, min(255, color_value))
        
        # Beat intensity-driven dimmer (more responsive)
        dimmer_value = int(beat_intensity * 200 + (llow + lmid) * 55)
        dimmer_value = max(0, min(255, dimmer_value))
        
        return {
            'pan': pan_value,
            'tilt': tilt_value,
            'color': color_value,
            'dimmer': dimmer_value
        }
    
    def train_model(self, epochs: int = 100, batch_size: int = 32):
        """Train the RNN model"""
        X, y = self.dataset.get_sequences()
        
        if len(X) == 0:
            logger.warning("No training data available")
            return
        
        logger.info(f"Training RNN with {len(X)} sequences")
        
        # Convert to tensors and move to device
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
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
                
                # Gradient clipping to prevent CUDA gradient issues
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
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
            print("RNN Training: No training data available. Please collect more samples first.")
            return False
        
        # Check if we have enough data for meaningful training
        min_sequences = max(20, self.sequence_length)  # At least 1x sequence length for buffer system
        if len(X) < min_sequences:
            logger.warning(f"Insufficient training data: {len(X)} sequences, need at least {min_sequences}")
            print(f"RNN Training: Insufficient data! Have {len(X)} sequences, need at least {min_sequences}. Please collect more samples.")
            return False
        
        logger.info(f"Training RNN with {len(X)} sequences, {epochs} epochs")
        print(f"RNN Training: Starting with {len(X)} sequences, {epochs} epochs")
        
        # Debug: Veri çeşitliliğini kontrol et
        print(f"RNN Training: Audio data range - Min: {X.min():.4f}, Max: {X.max():.4f}, Mean: {X.mean():.4f}")
        print(f"RNN Training: DMX data range - Min: {y.min():.4f}, Max: {y.max():.4f}, Mean: {y.mean():.4f}")
        print(f"RNN Training: DMX data std - {y.std():.4f}")
        
        # İlk birkaç sample'ı göster
        print(f"RNN Training: First 3 audio samples: {X[:3].tolist()}")
        print(f"RNN Training: First 3 DMX samples: {y[:3].tolist()}")
        
        # Convert to tensors and move to device
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.is_training = True
        self.training_losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient kontrolü
                total_grad_norm = 0
                for param in self.model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                
                # Gradient clipping to prevent CUDA gradient issues
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # İlk epoch'ta gradient bilgilerini göster
                if epoch == 0 and batch_idx == 0:
                    print(f"RNN Training: Gradient norm: {total_grad_norm:.6f}")
                    print(f"RNN Training: Loss: {loss.item():.6f}")
                
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
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
                print(f"RNN Training: Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        self.is_training = False
        self.last_training_time = time.time()
        
        # Eğitim sonrası verileri temizle (Buffer sistemi kapalı)
        self.clear_training_data()
        
        self.save_model()
        logger.info("RNN training completed")
        print(f"RNN Training: Completed! Final loss: {self.training_losses[-1]:.6f}")
        return True
    
    def clear_training_data(self):
        """Eğitim tamamlandıktan sonra kullanılan samples'ları temizle - Simple System"""
        samples_before = len(self.dataset.audio_sequences)
        self.dataset.audio_sequences.clear()
        self.dataset.dmx_sequences.clear()
        self.dataset.timestamps.clear()
        self.samples_since_last_train = 0
        logger.info(f"Cleared {samples_before} training samples after training completion")
        print(f"RNN Training: Cleared {samples_before} samples after training")
    
    def train_model_silent(self, epochs: int = 50, batch_size: int = 32):
        """Silent training for auto-retrain (no progress callback)"""
        X, y = self.dataset.get_sequences()
        
        if len(X) == 0:
            logger.warning("No training data available for auto-retrain")
            return False
        
        logger.info(f"Silent RNN training with {len(X)} sequences")
        
        # Convert to tensors and move to device
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.is_training = True
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent CUDA gradient issues
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
        
        self.is_training = False
        self.save_model()
        logger.info("Silent RNN training completed")
        return True
    
    def save_model(self):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
            'sequence_length': self.sequence_length,
            'device': str(self.device),
            'last_training_time': self.last_training_time
        }, self.model_path)
        
        # Also save training data
        self.dataset.save_to_file(self.data_path)
        
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a trained model"""
        if os.path.exists(self.model_path):
            # Suppress PyTorch warning about weights_only
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                checkpoint = torch.load(self.model_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_losses = checkpoint.get('training_losses', [])
            self.last_training_time = checkpoint.get('last_training_time', None)
            # Move model back to current device
            self.model.to(self.device)
            logger.info(f"Model loaded from {self.model_path} on {self.device}")
            if self.last_training_time:
                import datetime
                training_time = datetime.datetime.fromtimestamp(self.last_training_time)
                logger.info(f"Last training: {training_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            logger.info("No existing model found, using untrained model")
    
    def is_model_trained(self) -> bool:
        """Check if model is trained and loaded"""
        return (hasattr(self, 'model') and 
                self.model is not None and 
                os.path.exists(self.model_path))
    
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
