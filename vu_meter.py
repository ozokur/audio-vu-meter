#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio VU Meter GUI
Ses kartından gerçek zamanlı VU verisi okuyup görselleştiren uygulama
"""

import sys
import os
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
try:
    import pyaudiowpatch as pyaudio
except ImportError:
    import pyaudio
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QComboBox, QProgressBar
)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject

__version__ = "1.4.0"
__author__ = "Audio VU Meter Team"

# Audio Constants
AUDIO_FORMAT = pyaudio.paInt16
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHANNELS = 2
EPSILON = 1e-12
INT16_MAX = 32768.0
GUI_UPDATE_INTERVAL_MS = 8  # ~120 FPS
PEAK_DECAY_FACTOR = 0.95

# Logging setup
LOG_FILE = os.path.join(os.path.dirname(__file__), "audio_vu_meter.log")
_logger = logging.getLogger("audio_vu_meter")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    try:
        _fh = RotatingFileHandler(LOG_FILE, maxBytes=524288, backupCount=2, encoding="utf-8")
        _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        _logger.addHandler(_fh)
    except Exception:
        # Fallback to console if file handler fails
        _ch = logging.StreamHandler()
        _ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        _logger.addHandler(_ch)


class AudioMonitor(QObject):
    """Ses kartından veri okuma thread'i"""

    # Sinyaller
    level_updated = pyqtSignal(float, float)  # Sol, Sağ kanal
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.chunk_size = DEFAULT_CHUNK_SIZE
        self.channels = DEFAULT_CHANNELS
        self.device_index = None
        self.use_loopback = False  # Sistem ses çıkışını dinle

        self.pa = None
        self.stream = None

    def start(self) -> None:
        """Ses kartı okumayı başlat"""
        try:
            self.pa = pyaudio.PyAudio()

            # Varsayılan giriş cihazını al
            if self.device_index is None:
                self.device_index = self.pa.get_default_input_device_info()['index']

            # Seçilen cihaz bilgisini al
            device_info = self.pa.get_device_info_by_index(self.device_index)

            # Stream parametreleri
            stream_params = {
                'format': AUDIO_FORMAT,
                'channels': self.channels,
                'rate': int(device_info['defaultSampleRate']),
                'input': True,
                'input_device_index': self.device_index,
                'frames_per_buffer': self.chunk_size,
                'stream_callback': self._audio_callback
            }

            # Stream aç
            self.stream = self.pa.open(**stream_params)

            self.is_running = True
            self.stream.start_stream()

        except Exception as e:
            _logger.exception("Ses kartı başlatma hatası")
            self.error_occurred.emit(f"Ses kartı hatası: {e}")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - her chunk'ta çağrılır"""
        try:
            # Boş veri kontrolü
            if in_data is None or len(in_data) == 0:
                left_rms = 0.0
                right_rms = 0.0
            else:
                # Byte'ları numpy array'e çevir (float32'ye terfi ettir, overflow önle)
                audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32, copy=False)

                if audio_data.size == 0:
                    left_rms = 0.0
                    right_rms = 0.0
                else:
                    # Stereo ayrımı (eşit bölünemiyorsa mono varsay)
                    if self.channels == 2 and audio_data.size >= 2:
                        left = audio_data[0::2]
                        right = audio_data[1::2]
                    else:
                        left = right = audio_data

                    # RMS seviyesi (0-1 arası normalize). Epsilon ile negatif kök hatalarını engelle.
                    left_rms = float(np.sqrt(max(EPSILON, float(np.mean(left * left))))) / INT16_MAX
                    right_rms = float(np.sqrt(max(EPSILON, float(np.mean(right * right))))) / INT16_MAX

            # NaN/Inf temizle ve 0..1 aralığına kırp
            def _sanitize(v: float) -> float:
                if not np.isfinite(v):
                    return 0.0
                if v < 0.0:
                    return 0.0
                if v > 1.0:
                    return 1.0
                return v

            left_rms = _sanitize(left_rms)
            right_rms = _sanitize(right_rms)

            # Sinyal yayınla
            self.level_updated.emit(left_rms, right_rms)

        except Exception as e:
            _logger.exception("Audio callback hatası")
        
        return (in_data, pyaudio.paContinue)

    def stop(self) -> None:
        """Ses kartı okumayı durdur"""
        self.is_running = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.pa:
            self.pa.terminate()

    @staticmethod
    def get_audio_devices():
        """Mevcut ses kartlarını listele (giriş ve loopback)"""
        pa = pyaudio.PyAudio()
        devices = []

        # Windows için WASAPI loopback cihazlarını bul
        import platform
        if platform.system() == 'Windows':
            try:
                # pyaudiowpatch ile WASAPI loopback'i bul
                wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)

                # Varsayılan WASAPI loopback cihazını bul
                default_speakers = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

                # Loopback versiyonunu kontrol et
                if not default_speakers.get("isLoopbackDevice", False):
                    for loopback in pa.get_loopback_device_info_generator():
                        if default_speakers["name"] in loopback["name"]:
                            devices.append({
                                'index': loopback['index'],
                                'name': f"[Sistem] {loopback['name']}",
                                'channels': loopback['maxInputChannels'],
                                'is_loopback': True
                            })
                            break
                else:
                    devices.append({
                        'index': default_speakers['index'],
                        'name': f"[Sistem] {default_speakers['name']}",
                        'channels': default_speakers['maxInputChannels'],
                        'is_loopback': True
                    })
            except Exception as e:
                _logger.warning(f"WASAPI loopback bulunamadı: {e}")

        # Sonra normal giriş cihazlarını ekle
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0 and not info.get('isLoopbackDevice', False):
                    devices.append({
                        'index': i,
                        'name': f"[Mikrofon] {info['name']}",
                        'channels': info['maxInputChannels'],
                        'is_loopback': False
                    })
            except Exception as e:
                _logger.debug(f"Cihaz okunamadı (index {i}): {e}")
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
        self.min_db = -60.0
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
        self.peak_label = QLabel("-inf dB")
        peak_layout.addWidget(peak_label)
        peak_layout.addWidget(self.peak_label)
        peak_layout.addStretch()

        layout.addLayout(left_layout)
        layout.addLayout(right_layout)
        layout.addLayout(peak_layout)

        self.setLayout(layout)

    def update_levels(self, left: float, right: float) -> None:
        """Ses seviyelerini güncelle"""
        # NaN/Inf temizle, 0..1'e kırp
        def _s(v: float) -> float:
            try:
                v = float(v)
            except Exception:
                return 0.0
            if not np.isfinite(v):
                return 0.0
            if v < 0.0:
                return 0.0
            if v > 1.0:
                return 1.0
            return v

        self.left_level = _s(left)
        self.right_level = _s(right)

        # Peak tracking
        self.peak_left = max(self.peak_left * PEAK_DECAY_FACTOR, self.left_level)
        self.peak_right = max(self.peak_right * PEAK_DECAY_FACTOR, self.right_level)

        # Progress bar güncelle (0-100 arası)
        left_percent = int(round(self.left_level * 100))
        right_percent = int(round(self.right_level * 100))

        self.left_bar.setValue(left_percent)
        self.right_bar.setValue(right_percent)

        # dB hesapla
        left_db = self._linear_to_db(self.left_level)
        right_db = self._linear_to_db(self.right_level)
        peak_db = self._linear_to_db(max(self.peak_left, self.peak_right))

        # dB → % (min_db..0 dB) ve progress bar güncelle
        def _db_to_percent(dbv: float) -> int:
            import numpy as _np
            if not _np.isfinite(dbv):
                return 0
            top = 0.0
            bottom = float(self.min_db)
            if dbv <= bottom:
                return 0
            if dbv >= top:
                return 100
            return int(round((dbv - bottom) / (top - bottom) * 100))

        self.left_bar.setValue(_db_to_percent(left_db))
        self.right_bar.setValue(_db_to_percent(right_db))

        self.left_db_label.setText(f"{left_db:.1f} dB")
        self.right_db_label.setText(f"{right_db:.1f} dB")
        self.peak_label.setText(f"{peak_db:.1f} dB")

    @staticmethod
    def _linear_to_db(linear: float) -> float:
        """Linear değeri dB'ye çevir"""
        if linear > EPSILON:
            return 20 * np.log10(linear)
        else:
            return -float('inf')


    def set_min_db(self, min_db: float):
        """Meter alt dB aralığını ayarla (örn. -90, -60)."""
        try:
            val = float(min_db)
        except Exception:
            return
        if val > -10:
            val = -10.0
        if val < -120:
            val = -120.0
        self.min_db = val

class VUMeterApp(QMainWindow):
    """Ana uygulama penceresi"""

    def __init__(self):
        super().__init__()
        self.audio_monitor = AudioMonitor()
        self._last_left = 0.0
        self._last_right = 0.0
        self.init_ui()

        # Sinyal bağlantıları
        self.audio_monitor.level_updated.connect(self.on_level_updated)
        self.audio_monitor.error_occurred.connect(self.on_error)
        
        # 120 FPS GUI güncelleme zamanlayıcısı (~8 ms)
        self.gui_timer = QTimer(self)
        self.gui_timer.setInterval(GUI_UPDATE_INTERVAL_MS)
        self.gui_timer.timeout.connect(self._on_gui_tick)

    def init_ui(self):
        """UI oluştur"""
        self.setWindowTitle(f"Audio VU Meter + DMX Bass Reaktif Işıklandırma v{__version__}")
        self.setGeometry(100, 100, 600, 250)

        # Ana widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # Başlık
        title = QLabel("Audio VU Meter + DMX Bass Reaktif Işıklandırma")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Cihaz seçimi
        device_layout = QHBoxLayout()
        device_label = QLabel("Ses Kartı:")
        self.device_combo = QComboBox()
        self.refresh_devices()
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)

        # Aralık (dB) seçimi
        range_label = QLabel("Aralık:")
        self.range_combo = QComboBox()
        for val in (-90, -60, -48, -40, -30):
            self.range_combo.addItem(f"{val} dB", float(val))
        self.range_combo.setCurrentIndex(1)  # -60 dB
        self.range_combo.currentIndexChanged.connect(self.on_range_changed)

        device_layout.addWidget(range_label)
        device_layout.addWidget(self.range_combo)
        device_layout.addStretch()
        layout.addLayout(device_layout)

        # VU Meter widget
        self.vu_widget = VUMeterWidget()
        layout.addWidget(self.vu_widget)

        # Kontrol butonları
        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Başlat")
        self.start_button.clicked.connect(self.start_monitoring)
        self.start_button.setStyleSheet("padding: 10px; font-size: 14px;")

        self.stop_button = QPushButton("Durdur")
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("padding: 10px; font-size: 14px;")

        self.refresh_button = QPushButton("Yenile")
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
        self.gui_timer.start()

        # UI güncelle
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.device_combo.setEnabled(False)

        # Loopback ise farklı mesaj göster
        if self.audio_monitor.use_loopback:
            self.status_label.setText("Sistem sesi dinleniyor (Edge, YouTube vb.)...")
        else:
            self.status_label.setText("Mikrofon dinleniyor...")
        self.status_label.setStyleSheet("padding: 5px; background-color: #90EE90;")

    def stop_monitoring(self):
        """İzlemeyi durdur"""
        self.audio_monitor.stop()

        # UI güncelle
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.device_combo.setEnabled(True)
        self.status_label.setText("Durduruldu")
        self.status_label.setStyleSheet("padding: 5px; background-color: #FFB6C1;")

        # VU meter'ı sıfırla
        self.vu_widget.update_levels(0, 0)

    def on_level_updated(self, left, right):
        """Ses seviyesi güncellendiğinde"""
        try:
            self.vu_widget.update_levels(left, right)
        except Exception:
            _logger.exception("UI update_levels hatası")


    def _on_gui_tick(self):
        """120 Hz GUI güncellemesi"""
        try:
            self.vu_widget.update_levels(self._last_left, self._last_right)
        except Exception:
            _logger.exception("GUI tick update_levels hatası")

    def on_range_changed(self):
        """dB alt aralığı seçim değişti"""
        try:
            data = self.range_combo.currentData()
            if data is not None:
                self.vu_widget.set_min_db(float(data))
        except Exception:
            _logger.exception("on_range_changed hatası")
    def on_error(self, error_msg):
        """Hata oluştuğunda"""
        self.status_label.setText(f"Hata: {error_msg}")
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
