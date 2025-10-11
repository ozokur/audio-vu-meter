#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio VU Meter GUI
Ses kartından gerçek zamanlı VU verisi okuyup görselleştiren uygulama
"""

import sys
import numpy as np
import pyaudio
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, QProgressBar)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QColor, QPalette
import threading
import queue

__version__ = "1.0.0"
__author__ = "Audio VU Meter Team"


class AudioMonitor(QObject):
    """Ses kartından veri okuma thread'i"""
    
    # Signal tanımlamaları
    level_updated = pyqtSignal(float, float)  # Sol, Sağ kanal
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.is_running = False
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.channels = 2
        self.device_index = None
        self.use_loopback = False  # Sistem ses çıkışını dinle
        
        self.pa = None
        self.stream = None
        
    def start(self):
        """Ses kartı okumayı başlat"""
        try:
            self.pa = pyaudio.PyAudio()
            
            # Varsayılan giriş cihazını al
            if self.device_index is None:
                if self.use_loopback:
                    # Loopback için varsayılan çıkış cihazını kullan
                    self.device_index = self.pa.get_default_output_device_info()['index']
                else:
                    self.device_index = self.pa.get_default_input_device_info()['index']
            
            # Stream parametreleri
            stream_params = {
                'format': pyaudio.paInt16,
                'channels': self.channels,
                'rate': self.sample_rate,
                'input': True,
                'input_device_index': self.device_index,
                'frames_per_buffer': self.chunk_size,
                'stream_callback': self._audio_callback
            }
            
            # WASAPI loopback için özel ayar (Windows)
            if self.use_loopback:
                try:
                    # WASAPI host API için loopback flag'i ekle
                    import platform
                    if platform.system() == 'Windows':
                        stream_params['as_loopback'] = True
                except:
                    pass
            
            # Stream aç
            self.stream = self.pa.open(**stream_params)
            
            self.is_running = True
            self.stream.start_stream()
            
        except Exception as e:
            self.error_occurred.emit(f"Ses kartı hatası: {e}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - her chunk'ta çağrılır"""
        try:
            # Byte'ları numpy array'e çevir
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            
            # Stereo ayrımı
            if self.channels == 2:
                left = audio_data[0::2]
                right = audio_data[1::2]
            else:
                left = right = audio_data
            
            # RMS seviyesi hesapla (0-1 arası normalize)
            left_rms = np.sqrt(np.mean(left**2)) / 32768.0
            right_rms = np.sqrt(np.mean(right**2)) / 32768.0
            
            # Signal emit et
            self.level_updated.emit(left_rms, right_rms)
            
        except Exception as e:
            print(f"Callback hatası: {e}")
        
        return (in_data, pyaudio.paContinue)
    
    def stop(self):
        """Ses kartı okumayı durdur"""
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.pa:
            self.pa.terminate()
    
    @staticmethod
    def get_audio_devices():
        """Mevcut ses kartlarını listele (input ve loopback)"""
        pa = pyaudio.PyAudio()
        devices = []
        
        # Önce sistem çıkışını loopback olarak ekle (Windows için)
        import platform
        if platform.system() == 'Windows':
            try:
                default_output = pa.get_default_output_device_info()
                devices.append({
                    'index': default_output['index'],
                    'name': '🔊 Sistem Ses Çıkışı (Loopback)',
                    'channels': default_output['maxOutputChannels'],
                    'is_loopback': True
                })
            except:
                pass
        
        # Sonra normal input cihazlarını ekle
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:  # Sadece giriş cihazları
                    devices.append({
                        'index': i,
                        'name': f"🎤 {info['name']}",
                        'channels': info['maxInputChannels'],
                        'is_loopback': False
                    })
            except:
                pass
        
        pa.terminate()
        return devices


class VUMeterWidget(QWidget):
    """VU Meter görselleştirme widget'ı"""
    
    def __init__(self):
        super().__init__()
        self.left_level = 0.0
        self.right_level = 0.0
        self.peak_left = 0.0
        self.peak_right = 0.0
        self.init_ui()
    
    def init_ui(self):
        """UI bileşenlerini oluştur"""
        layout = QVBoxLayout()
        
        # Sol kanal
        left_layout = QHBoxLayout()
        left_label = QLabel("Sol:")
        left_label.setFixedWidth(40)
        self.left_bar = QProgressBar()
        self.left_bar.setMaximum(100)
        self.left_bar.setTextVisible(True)
        self.left_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 green, stop:0.7 yellow, stop:0.9 orange, stop:1 red);
            }
        """)
        self.left_db_label = QLabel("0.0 dB")
        self.left_db_label.setFixedWidth(70)
        
        left_layout.addWidget(left_label)
        left_layout.addWidget(self.left_bar)
        left_layout.addWidget(self.left_db_label)
        
        # Sağ kanal
        right_layout = QHBoxLayout()
        right_label = QLabel("Sağ:")
        right_label.setFixedWidth(40)
        self.right_bar = QProgressBar()
        self.right_bar.setMaximum(100)
        self.right_bar.setTextVisible(True)
        self.right_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 green, stop:0.7 yellow, stop:0.9 orange, stop:1 red);
            }
        """)
        self.right_db_label = QLabel("0.0 dB")
        self.right_db_label.setFixedWidth(70)
        
        right_layout.addWidget(right_label)
        right_layout.addWidget(self.right_bar)
        right_layout.addWidget(self.right_db_label)
        
        # Peak göstergesi
        peak_layout = QHBoxLayout()
        peak_label = QLabel("Peak:")
        peak_label.setFixedWidth(40)
        self.peak_label = QLabel("-∞ dB")
        peak_layout.addWidget(peak_label)
        peak_layout.addWidget(self.peak_label)
        peak_layout.addStretch()
        
        layout.addLayout(left_layout)
        layout.addLayout(right_layout)
        layout.addLayout(peak_layout)
        
        self.setLayout(layout)
    
    def update_levels(self, left, right):
        """Ses seviyelerini güncelle"""
        self.left_level = left
        self.right_level = right
        
        # Peak tracking
        self.peak_left = max(self.peak_left * 0.95, left)
        self.peak_right = max(self.peak_right * 0.95, right)
        
        # Progress bar güncelle (0-100 arası)
        left_percent = int(left * 100)
        right_percent = int(right * 100)
        
        self.left_bar.setValue(left_percent)
        self.right_bar.setValue(right_percent)
        
        # dB hesapla
        left_db = self._linear_to_db(left)
        right_db = self._linear_to_db(right)
        peak_db = self._linear_to_db(max(self.peak_left, self.peak_right))
        
        self.left_db_label.setText(f"{left_db:.1f} dB")
        self.right_db_label.setText(f"{right_db:.1f} dB")
        self.peak_label.setText(f"{peak_db:.1f} dB")
    
    @staticmethod
    def _linear_to_db(linear):
        """Linear değeri dB'ye çevir"""
        if linear > 0:
            return 20 * np.log10(linear)
        else:
            return -float('inf')


class VUMeterApp(QMainWindow):
    """Ana uygulama penceresi"""
    
    def __init__(self):
        super().__init__()
        self.audio_monitor = AudioMonitor()
        self.init_ui()
        
        # Signal bağlantıları
        self.audio_monitor.level_updated.connect(self.on_level_updated)
        self.audio_monitor.error_occurred.connect(self.on_error)
    
    def init_ui(self):
        """UI oluştur"""
        self.setWindowTitle(f"🎵 Audio VU Meter v{__version__}")
        self.setGeometry(100, 100, 600, 250)
        
        # Ana widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Başlık
        title = QLabel("🎵 Gerçek Zamanlı VU Meter - Mikrofon & Sistem Sesi")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Cihaz seçimi
        device_layout = QHBoxLayout()
        device_label = QLabel("Ses Kartı:")
        self.device_combo = QComboBox()
        self.refresh_devices()
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
        layout.addLayout(device_layout)
        
        # VU Meter widget
        self.vu_widget = VUMeterWidget()
        layout.addWidget(self.vu_widget)
        
        # Kontrol butonları
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("▶ Başlat")
        self.start_button.clicked.connect(self.start_monitoring)
        self.start_button.setStyleSheet("padding: 10px; font-size: 14px;")
        
        self.stop_button = QPushButton("⏹ Durdur")
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("padding: 10px; font-size: 14px;")
        
        self.refresh_button = QPushButton("🔄 Yenile")
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.refresh_button.setStyleSheet("padding: 10px; font-size: 14px;")
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.refresh_button)
        
        layout.addLayout(button_layout)
        
        # Status bar
        self.status_label = QLabel("Hazır")
        self.status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        layout.addWidget(self.status_label)
        
        central_widget.setLayout(layout)
    
    def refresh_devices(self):
        """Ses kartlarını yenile"""
        self.device_combo.clear()
        devices = AudioMonitor.get_audio_devices()
        
        for device in devices:
            self.device_combo.addItem(
                f"{device['name']} ({device['channels']} kanal)",
                device  # Tüm device bilgisini data olarak sakla
            )
    
    def start_monitoring(self):
        """İzlemeyi başlat"""
        # Seçilen cihazı al
        device_info = self.device_combo.currentData()
        if device_info is not None:
            self.audio_monitor.device_index = device_info['index']
            self.audio_monitor.use_loopback = device_info.get('is_loopback', False)
        
        # Başlat
        self.audio_monitor.start()
        
        # UI güncelle
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.device_combo.setEnabled(False)
        
        # Loopback ise farklı mesaj göster
        if self.audio_monitor.use_loopback:
            self.status_label.setText("🔊 Sistem sesi dinleniyor (Edge, YouTube vb.)...")
        else:
            self.status_label.setText("🎤 Mikrofon dinleniyor...")
        self.status_label.setStyleSheet("padding: 5px; background-color: #90EE90;")
    
    def stop_monitoring(self):
        """İzlemeyi durdur"""
        self.audio_monitor.stop()
        
        # UI güncelle
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.device_combo.setEnabled(True)
        self.status_label.setText("⏹ Durduruldu")
        self.status_label.setStyleSheet("padding: 5px; background-color: #FFB6C1;")
        
        # VU meter'ı sıfırla
        self.vu_widget.update_levels(0, 0)
    
    def on_level_updated(self, left, right):
        """Ses seviyesi güncellendiğinde"""
        self.vu_widget.update_levels(left, right)
    
    def on_error(self, error_msg):
        """Hata oluştuğunda"""
        self.status_label.setText(f"❌ Hata: {error_msg}")
        self.status_label.setStyleSheet("padding: 5px; background-color: #FFB6C1;")
        self.stop_monitoring()
    
    def closeEvent(self, event):
        """Pencere kapatıldığında"""
        self.audio_monitor.stop()
        event.accept()


def main():
    """Ana program"""
    app = QApplication(sys.argv)
    
    # Dark theme (opsiyonel)
    app.setStyle("Fusion")
    
    window = VUMeterApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

