#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio VU Meter GUI (120 FPS + Ayarlanabilir dB AralÄ±ÄŸÄ±)
"""

import sys
import os
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import time
import json
try:
    import pyaudiowpatch as pyaudio
except ImportError:
    import pyaudio
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QComboBox, QProgressBar, QDoubleSpinBox, QCheckBox, QGridLayout, QSpinBox
)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject

# DMX USB kontrolü için
try:
    import usb.core
    import usb.util
    DMX_AVAILABLE = True
except ImportError:
    DMX_AVAILABLE = False

__version__ = "1.6.6"


# Logging
LOG_FILE = os.path.join(os.path.dirname(__file__), "audio_vu_meter.log")
logger = logging.getLogger("audio_vu_meter")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    try:
        fh = RotatingFileHandler(LOG_FILE, maxBytes=524288, backupCount=2, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    except Exception:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(ch)


class AudioMonitor(QObject):
    level_updated = pyqtSignal(float, float)
    bands_updated = pyqtSignal(object)  # (Llow, Lmid, Lhigh, Rlow, Rmid, Rhigh)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.sample_rate = 44100
        self.chunk_size = 256  # 120 Hz GUI iÃ§in daha sÄ±k callback
        self.channels = 2
        self.device_index = None
        self.use_loopback = False
        self.pa = None
        self.stream = None

    def start(self):
        try:
            self.pa = pyaudio.PyAudio()
            if self.device_index is None:
                self.device_index = self.pa.get_default_input_device_info()['index']
            device_info = self.pa.get_device_info_by_index(self.device_index)
            params = dict(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=int(device_info['defaultSampleRate']),
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback,
            )
            self.stream = self.pa.open(**params)
            self.stream.start_stream()
        except Exception as e:
            logger.exception("Ses kartÄ± baÅŸlatma hatasÄ±")
            self.error_occurred.emit(f"Ses kartÄ± hatasÄ±: {e}")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        try:
            if not in_data:
                l_rms = r_rms = 0.0
            else:
                data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32, copy=False)
                if data.size == 0:
                    l_rms = r_rms = 0.0
                else:
                    if self.channels == 2 and data.size >= 2:
                        left = data[0::2]
                        right = data[1::2]
                    else:
                        left = right = data
                    eps = 1e-12
                    l_rms = float(np.sqrt(max(eps, float(np.mean(left * left))))) / 32768.0
                    r_rms = float(np.sqrt(max(eps, float(np.mean(right * right))))) / 32768.0

            def _s(v: float) -> float:
                if not np.isfinite(v):
                    return 0.0
                if v < 0.0:
                    return 0.0
                if v > 1.0:
                    return 1.0
                return v

            l_rms = _s(l_rms)
            r_rms = _s(r_rms)
            self.level_updated.emit(l_rms, r_rms)

            # 3 bant (Low/Mid/High) iÃ§in spektral gÃ¼Ã§ten RMS tahmini
            try:
                if 'data' in locals() and data is not None and data.size > 0:
                    # Kanal ayrÄ±mÄ±
                    if self.channels == 2 and data.size >= 2:
                        lch = data[0::2]
                        rch = data[1::2]
                    else:
                        lch = rch = data

                    def band_rms(x: np.ndarray, sr: int, bands):
                        N = x.size
                        if N <= 0:
                            return [0.0] * len(bands)
                        w = np.hanning(N).astype(np.float32)
                        X = np.fft.rfft(x * w)
                        freqs = np.fft.rfftfreq(N, 1.0 / sr)
                        pow_spec = (np.abs(X) ** 2)
                        out = []
                        denom = float(N * N)
                        for lo, hi in bands:
                            idx = np.where((freqs >= lo) & (freqs < hi))[0]
                            if idx.size == 0:
                                out.append(0.0)
                                continue
                            p = float(np.sum(pow_spec[idx]))
                            rms = np.sqrt(max(0.0, p) / denom) / 32768.0
                            out.append(_s(rms))
                        return out

                    bands = [(20.0, 250.0), (250.0, 4000.0), (4000.0, 20000.0)]
                    sr = int(self.sample_rate)
                    l_bands = band_rms(lch, sr, bands)
                    r_bands = band_rms(rch, sr, bands)
                    self.bands_updated.emit(tuple(l_bands + r_bands))
            except Exception:
                # Spektral hesap hatasÄ± UI'Ä± durdurmasÄ±n
                pass
        except Exception:
            logger.exception("Audio callback hatasÄ±")
        return (in_data, pyaudio.paContinue)

    def stop(self):
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.pa:
                self.pa.terminate()
        except Exception:
            pass

    @staticmethod
    def get_audio_devices():
        pa = pyaudio.PyAudio()
        devices = []
        import platform
        if platform.system() == 'Windows':
            try:
                wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
                if not default_out.get("isLoopbackDevice", False):
                    for loopback in pa.get_loopback_device_info_generator():
                        if default_out["name"] in loopback["name"]:
                            devices.append({
                                'index': loopback['index'],
                                'name': f"[Sistem] {loopback['name']}",
                                'channels': loopback['maxInputChannels'],
                                'is_loopback': True
                            })
                            break
                else:
                    devices.append({
                        'index': default_out['index'],
                        'name': f"[Sistem] {default_out['name']}",
                        'channels': default_out['maxInputChannels'],
                        'is_loopback': True
                    })
            except Exception as e:
                logger.warning(f"WASAPI loopback bulunamadÄ±: {e}")
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0 and not info.get('isLoopbackDevice', False):
                    devices.append({'index': i, 'name': f"[Mikrofon] {info['name']}", 'channels': info['maxInputChannels'], 'is_loopback': False})
            except Exception:
                pass
        pa.terminate()
        return devices


# UDMX Device IDs
UDMX_DEVICES = [
    {'vendor': 0x16C0, 'product': 0x05DC, 'name': 'Anyma uDMX'},
    {'vendor': 0x03EB, 'product': 0x8888, 'name': 'DMXControl uDMX'},
]


class DMXController:
    """DMX UDMX Controller for Audio-reactive lighting"""
    def __init__(self, logger_instance=None):
        self.dmx_data = [0] * 512
        self.usb_device = None
        self.running = False
        self.logger = logger_instance or logger
        self.enabled = False
        
        # Color mapping for audio bands (Channel 3 values)
        # 0-9: White, 10+: Colors (Red, Yellow, Green, Cyan, Blue, Magenta, etc.)
        self.color_map = {
            'white': 5,      # 0-9 range
            'red': 15,       # Kırmızı
            'orange': 25,    # Turuncu
            'yellow': 35,    # Sarı
            'green': 50,     # Yeşil
            'cyan': 70,      # Camgöbeği
            'blue': 90,      # Mavi
            'purple': 110,   # Mor
            'magenta': 120,  # Magenta
        }
        
        if DMX_AVAILABLE:
            self.logger.info("DMX Controller initialized (USB support available)")
        else:
            self.logger.warning("DMX Controller initialized (pyusb not available - DMX disabled)")
    
    def find_udmx_devices(self):
        """Find connected UDMX devices"""
        if not DMX_AVAILABLE:
            return []
        
        devices = []
        try:
            for device_info in UDMX_DEVICES:
                dev = usb.core.find(idVendor=device_info['vendor'], idProduct=device_info['product'])
                if dev is not None:
                    devices.append({
                        'device': dev,
                        'name': device_info['name'],
                        'vendor': device_info['vendor'],
                        'product': device_info['product'],
                    })
                    self.logger.info(f"Found UDMX device: {device_info['name']}")
        except Exception as e:
            self.logger.error(f"Error finding UDMX devices: {e}")
        return devices
    
    def connect(self, device_index=0):
        """Connect to UDMX device"""
        if not DMX_AVAILABLE:
            self.logger.warning("Cannot connect: pyusb not available")
            return False
        
        try:
            devices = self.find_udmx_devices()
            if not devices:
                self.logger.warning("No UDMX devices found")
                return False
            
            if device_index >= len(devices):
                device_index = 0
            
            device_info = devices[device_index]
            self.usb_device = device_info['device']
            
            try:
                if self.usb_device.is_kernel_driver_active(0):
                    self.usb_device.detach_kernel_driver(0)
            except:
                pass
            
            try:
                self.usb_device.set_configuration()
            except:
                pass
            
            self.running = True
            self.enabled = True
            self.logger.info(f"Connected to {device_info['name']}")
            return True
            
        except Exception as e:
            self.logger.error(f"DMX connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from UDMX device"""
        self.running = False
        self.enabled = False
        if self.usb_device:
            try:
                usb.util.dispose_resources(self.usb_device)
                self.logger.info("DMX device disconnected")
            except:
                pass
            self.usb_device = None
    
    def set_channel(self, channel, value):
        """Set DMX channel value (1-512, 0-255)"""
        if 1 <= channel <= 512 and 0 <= value <= 255:
            self.dmx_data[channel - 1] = int(value)
    
    def send_frame(self):
        """Send DMX frame to UDMX device"""
        if not self.enabled or not self.usb_device:
            return
        
        try:
            # Send first 9 channels (enough for our use)
            for i in range(9):
                self.usb_device.ctrl_transfer(0x40, 0x01, self.dmx_data[i], i, [])
        except Exception as e:
            # Silently fail to avoid log spam
            pass
    
    def set_audio_reactive(self, llow, lmid, lhigh, rlow, rmid, rhigh, light_states=None, beat_flash=False, range_config=None):
        """Set DMX values based on audio band levels - Llow mode with beat flash and range scaling"""
        if not self.enabled:
            return
        
        # 🎵 Llow seviyesini hesapla (L ve R kanallarının ortalaması)
        llow_level = (llow + rlow) / 2.0
        
        # 🎚️ RANGE SCALING: Range light min/max dB'ye göre scale et
        if range_config and 'min_db' in range_config and 'max_db' in range_config:
            min_db = float(range_config['min_db'])
            max_db = float(range_config['max_db'])
            
            # Llow'u dB'ye çevir
            if llow_level > 1e-12:
                llow_db = 20.0 * np.log10(llow_level)
            else:
                llow_db = -120.0
            
            # Min/Max dB aralığına göre 0-1 arası scale et
            if llow_db <= min_db:
                scaled_level = 0.0
            elif llow_db >= max_db:
                scaled_level = 1.0
            else:
                scaled_level = (llow_db - min_db) / (max_db - min_db)
            
            # Scaled level'i kullan
            llow_level = max(0.0, min(1.0, scaled_level))
        
        # Base parlaklık: Scaled Llow seviyesine göre (0-255)
        base_brightness = int(llow_level * 255)
        base_brightness = max(0, min(255, base_brightness))
        
        # 🔥 BEAT FLASH: Beat anında maksimum parlaklık!
        if beat_flash:
            brightness = 255  # Full brightness on beat!
        else:
            brightness = base_brightness
        
        # Minimum eşik: Çok düşük seslerde kapansın
        if base_brightness < 10 and not beat_flash:
            self.set_channel(5, 0)      # Dimmer off
            self.set_channel(3, self.color_map['white'])
            self.set_channel(6, 0)      # Master off
            self.send_frame()
            return
        
        # Channel 5: Dimmer (Parlaklık - beat'te maksimum!)
        self.set_channel(5, brightness)
        
        # Channel 3: Renk - MANUEL KONTROL (GUI'den ayarlanır)
        # Bu kanal artık otomatik değişmez, sadece GUI'den set edilir
        
        # Channel 6: Master Dimmer - AYNI MAPPING (Llow + Range Scaling)
        # Ch5 ile aynı mantık ama ayrı kanal
        self.set_channel(6, brightness)
        
        # Send the frame
        self.send_frame()


class VUMeterWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.left_level = 0.0
        self.right_level = 0.0
        self.peak_left = 0.0
        self.peak_right = 0.0
        self.min_db = -60.0
        self.peak_hold_ms = 1000
        self.peak_hold_enabled = False  # disable peak-hold by default
        self.peak_decay_dbps = 10.0  # dB per second decay after hold
        self.bands = [0.0] * 6  # Llow, Lmid, Lhigh, Rlow, Rmid, Rhigh
        # Tempo parametreleri
        self.tempo_alpha = 0.20
        self.tempo_delta = 0.08
        self.tempo_min_interval = 0.25  # s
        self.tempo_hold = 0.12          # s
        # Tempo/beat takibi durumu
        self._tempo = {
            'L': {'env': 0.0, 'on_until': 0.0, 'last': 0.0, 'last_flash': 0.0, 'beats': []},
            'R': {'env': 0.0, 'on_until': 0.0, 'last': 0.0, 'last_flash': 0.0, 'beats': []},
            'Llow': {'env': 0.0, 'on_until': 0.0, 'last': 0.0, 'last_flash': 0.0, 'beats': []},
            'Lmid': {'env': 0.0, 'on_until': 0.0, 'last': 0.0, 'last_flash': 0.0, 'beats': []},
            'Lhigh': {'env': 0.0, 'on_until': 0.0, 'last': 0.0, 'last_flash': 0.0, 'beats': []},
            'Rlow': {'env': 0.0, 'on_until': 0.0, 'last': 0.0, 'last_flash': 0.0, 'beats': []},
            'Rmid': {'env': 0.0, 'on_until': 0.0, 'last': 0.0, 'last_flash': 0.0, 'beats': []},
            'Rhigh': {'env': 0.0, 'on_until': 0.0, 'last': 0.0, 'last_flash': 0.0, 'beats': []},
        }
        # Per-key tarihÃ§e ve varsayÄ±lan parametreler
        for k in list(self._tempo.keys()):
            self._tempo[k]['hist'] = []
            self._tempo[k]['hist_maxlen'] = 120
        self._tempo_params = {kk: {'alpha': 0.20, 'delta': 0.08, 'min_interval': 0.25, 'hold': 0.12, 'auto': False, 'k': 0.6}
                              for kk in self._tempo.keys()}
        # TÃ¼m kanallar/bantlar iÃ§in (L, R, Llow/Lmid/Lhigh, Rlow/Rmid/Rhigh)
        # otomatik eÅŸik ve hÄ±zlÄ± tepki parametrelerini varsayÄ±lan yap
        for key in list(self._tempo_params.keys()):
            self._tempo_params[key].update({
                'auto': True,   # otomatik delta kullan
                'alpha': 0.15,  # biraz daha hÄ±zlÄ± envelope
                'k': 0.8,       # std Ã§arpanÄ± (daha hassas)
                'min_interval': 0.20,
                'hold': 0.10,
            })
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        left_layout = QHBoxLayout()
        self.l_light = self._make_light()
        left_label = QLabel("Sol:")
        left_label.setFixedWidth(40)
        self.left_bar = QProgressBar()
        self.left_bar.setMaximum(100)
        self.left_bar.setTextVisible(True)
        self.left_bar.setStyleSheet("""
            QProgressBar { border: 2px solid grey; border-radius: 5px; text-align: center; }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 green, stop:0.7 yellow, stop:0.9 orange, stop:1 red);
            }
        """)
        self.left_db_label = QLabel("0.0 dB")
        self.left_db_label.setFixedWidth(70)
        self.left_byte_label = QLabel("0x00")
        self.left_byte_label.setFixedWidth(48)
        self.left_bpm_label = QLabel("-- BPM")
        self.left_bpm_label.setFixedWidth(70)
        self.left_peak_hold_label = QLabel("Pk -- dB")
        self.left_peak_hold_label.setFixedWidth(80)
        left_layout.addWidget(self.l_light)
        left_layout.addWidget(left_label)
        left_layout.addWidget(self.left_bar)
        left_layout.addWidget(self.left_db_label)
        left_layout.addWidget(self.left_byte_label)
        left_layout.addWidget(self.left_bpm_label)
        left_layout.addWidget(self.left_peak_hold_label)

        right_layout = QHBoxLayout()
        self.r_light = self._make_light()
        right_label = QLabel("SaÄŸ:")
        right_label.setFixedWidth(40)
        self.right_bar = QProgressBar()
        self.right_bar.setMaximum(100)
        self.right_bar.setTextVisible(True)
        self.right_bar.setStyleSheet(self.left_bar.styleSheet())
        self.right_db_label = QLabel("0.0 dB")
        self.right_db_label.setFixedWidth(70)
        self.right_byte_label = QLabel("0x00")
        self.right_byte_label.setFixedWidth(48)
        self.right_bpm_label = QLabel("-- BPM")
        self.right_bpm_label.setFixedWidth(70)
        self.right_peak_hold_label = QLabel("Pk -- dB")
        self.right_peak_hold_label.setFixedWidth(80)
        right_layout.addWidget(self.r_light)
        right_layout.addWidget(right_label)
        right_layout.addWidget(self.right_bar)
        right_layout.addWidget(self.right_db_label)
        right_layout.addWidget(self.right_byte_label)
        right_layout.addWidget(self.right_bpm_label)
        right_layout.addWidget(self.right_peak_hold_label)

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

        # 6 bant (Low/Mid/High x L/R)
        bands_layout = QVBoxLayout()

        def make_band_row(label_text: str):
            row = QHBoxLayout()
            light = self._make_light()
            lab = QLabel(label_text)
            lab.setFixedWidth(60)
            bar = QProgressBar()
            bar.setMaximum(100)
            bar.setTextVisible(True)
            bar.setStyleSheet(self.left_bar.styleSheet())
            db_lab = QLabel("0.0 dB")
            db_lab.setFixedWidth(70)
            bpm_lab = QLabel("-- BPM")
            bpm_lab.setFixedWidth(70)
            row.addWidget(light)
            row.addWidget(lab)
            row.addWidget(bar)
            row.addWidget(db_lab)
            row.addWidget(bpm_lab)
            return row, light, bar, db_lab, bpm_lab

        self.l_low_row, self.l_low_light, self.l_low_bar, self.l_low_db, self.l_low_bpm = make_band_row("L Low:")
        self.l_mid_row, self.l_mid_light, self.l_mid_bar, self.l_mid_db, self.l_mid_bpm = make_band_row("L Mid:")
        self.l_high_row, self.l_high_light, self.l_high_bar, self.l_high_db, self.l_high_bpm = make_band_row("L High:")
        self.r_low_row, self.r_low_light, self.r_low_bar, self.r_low_db, self.r_low_bpm = make_band_row("R Low:")
        self.r_mid_row, self.r_mid_light, self.r_mid_bar, self.r_mid_db, self.r_mid_bpm = make_band_row("R Mid:")
        self.r_high_row, self.r_high_light, self.r_high_bar, self.r_high_db, self.r_high_bpm = make_band_row("R High:")

        for row in (self.l_low_row, self.l_mid_row, self.l_high_row,
                    self.r_low_row, self.r_mid_row, self.r_high_row):
            bands_layout.addLayout(row)

        layout.addLayout(bands_layout)
        
        # Range-based light controls
        range_ctl = QHBoxLayout()
        range_ctl.addWidget(QLabel("Range Target:"))
        self.range_target = QComboBox()
        for key in ("Llow","Lmid","Lhigh","Rlow","Rmid","Rhigh"):
            self.range_target.addItem(key, key)
        range_ctl.addWidget(self.range_target)
        range_ctl.addWidget(QLabel("Min dB:"))
        self.range_min_db = QDoubleSpinBox()
        self.range_min_db.setRange(-120.0, 0.0)
        self.range_min_db.setSingleStep(1.0)
        self.range_min_db.setValue(-60.0)
        range_ctl.addWidget(self.range_min_db)
        range_ctl.addWidget(QLabel("Max dB:"))
        self.range_max_db = QDoubleSpinBox()
        self.range_max_db.setRange(-120.0, 0.0)
        self.range_max_db.setSingleStep(1.0)
        self.range_max_db.setValue(0.0)
        range_ctl.addWidget(self.range_max_db)
        self.range_one_shot = QCheckBox("One Shot")
        range_ctl.addWidget(self.range_one_shot)
        range_ctl.addWidget(QLabel("Shot ms:"))
        self.range_shot_ms = QSpinBox()
        self.range_shot_ms.setRange(1, 10000)
        self.range_shot_ms.setSingleStep(10)
        self.range_shot_ms.setValue(120)
        range_ctl.addWidget(self.range_shot_ms)
        self.range_enable = QCheckBox("Range Lights Enable")
        self.range_enable.setChecked(False)
        range_ctl.addWidget(self.range_enable)
        range_ctl.addStretch()
        layout.addLayout(range_ctl)

        # Backend store for range configs and state
        self._range_cfg = {k: {"min_db": -60.0, "max_db": 0.0, "one_shot": False, "shot_ms": 120}
                           for k in ("Llow","Lmid","Lhigh","Rlow","Rmid","Rhigh")}
        self._range_state = {k: {"prev_in": False} for k in self._range_cfg.keys()}

        def _apply_range_ui_to_cfg():
            try:
                key = self.range_target.currentData() or "Llow"
                cfg = self._range_cfg[str(key)]
                cfg["min_db"] = float(self.range_min_db.value())
                cfg["max_db"] = float(self.range_max_db.value())
                if cfg["min_db"] > cfg["max_db"]:
                    cfg["min_db"], cfg["max_db"] = cfg["max_db"], cfg["min_db"]
                cfg["one_shot"] = bool(self.range_one_shot.isChecked())
                cfg["shot_ms"] = int(self.range_shot_ms.value())
                # persist
                self.save_range_cfg()
            except Exception:
                pass

        def _refresh_range_ui_from_cfg():
            try:
                key = self.range_target.currentData() or "Llow"
                cfg = self._range_cfg[str(key)]
                self.range_min_db.setValue(float(cfg.get("min_db", -60.0)))
                self.range_max_db.setValue(float(cfg.get("max_db", 0.0)))
                self.range_one_shot.setChecked(bool(cfg.get("one_shot", False)))
                self.range_shot_ms.setValue(int(cfg.get("shot_ms", 120)))
            except Exception:
                pass

        self.range_target.currentIndexChanged.connect(_refresh_range_ui_from_cfg)
        self.range_min_db.valueChanged.connect(_apply_range_ui_to_cfg)
        self.range_max_db.valueChanged.connect(_apply_range_ui_to_cfg)
        self.range_one_shot.stateChanged.connect(_apply_range_ui_to_cfg)
        self.range_shot_ms.valueChanged.connect(_apply_range_ui_to_cfg)
        # Persist on enable toggle
        self.range_enable.stateChanged.connect(lambda *_: self.save_range_cfg())
        # Try load persisted cfg and update UI
        try:
            loaded = self.load_range_cfg()
        except Exception:
            loaded = None
        _refresh_range_ui_from_cfg()
        if isinstance(loaded, dict) and 'range_enable' in loaded:
            try:
                self.range_enable.setChecked(bool(loaded.get('range_enable')))
            except Exception:
                pass
        
        # Peak hold control (milliseconds, 1 .. 10000 ms)
        ctl_row = QHBoxLayout()
        ctl_row.addWidget(QLabel("Peak Hold ms:"))
        self.peak_hold_spin = QSpinBox()
        self.peak_hold_spin.setRange(1, 10000)
        self.peak_hold_spin.setSingleStep(1)
        self.peak_hold_spin.setValue(1000)
        self.peak_hold_spin.valueChanged.connect(self.on_peak_hold_changed)
        # Disable control when feature is off
        self.peak_hold_spin.setEnabled(False)
        self.peak_hold_spin.setToolTip("Peak-hold devre dışı (ms kullanılmıyor)")
        ctl_row.addWidget(self.peak_hold_spin)
        ctl_row.addStretch()
        layout.addLayout(ctl_row)
        self.setLayout(layout)

    # --- Range config persistence ---
    def _range_cfg_path(self) -> str:
        try:
            base = os.path.dirname(__file__)
        except Exception:
            base = os.getcwd()
        return os.path.join(base, "range_cfg.json")

    def load_range_cfg(self):
        path = self._range_cfg_path()
        try:
            if not os.path.exists(path):
                return None
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cfg = data.get('range_cfg')
            if isinstance(cfg, dict):
                for k in ("Llow","Lmid","Lhigh","Rlow","Rmid","Rhigh"):
                    v = cfg.get(k)
                    if isinstance(v, dict):
                        self._range_cfg[k].update({
                            'min_db': float(v.get('min_db', self._range_cfg[k]['min_db'])),
                            'max_db': float(v.get('max_db', self._range_cfg[k]['max_db'])),
                            'one_shot': bool(v.get('one_shot', self._range_cfg[k]['one_shot'])),
                            'shot_ms': int(v.get('shot_ms', self._range_cfg[k]['shot_ms'])),
                        })
            ren = data.get('range_enable')
            if isinstance(ren, bool) and hasattr(self, 'range_enable'):
                try:
                    self.range_enable.setChecked(ren)
                except Exception:
                    pass
            return data
        except Exception:
            return None

    def save_range_cfg(self):
        path = self._range_cfg_path()
        try:
            data = {
                'range_enable': bool(self.range_enable.isChecked()) if hasattr(self, 'range_enable') else False,
                'range_cfg': self._range_cfg,
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def on_peak_hold_changed(self):
        try:
            # peak_hold_spin is in milliseconds with 1 ms resolution
            ms = int(self.peak_hold_spin.value())
            self.set_peak_hold_ms(max(0, ms))
        except Exception:
            pass

    @staticmethod
    def _linear_to_db(linear: float) -> float:
        if linear > 1e-12:
            return 20.0 * np.log10(linear)
        return -float('inf')

    def set_min_db(self, min_db: float):
        try:
            val = float(min_db)
        except Exception:
            return
        self.min_db = max(-120.0, min(-10.0, val))

    def set_peak_hold_ms(self, ms: int):
        try:
            self.peak_hold_ms = max(0, int(ms))
        except Exception:
            self.peak_hold_ms = 1000

    def update_levels(self, left: float, right: float):
        def _s(v: float) -> float:
            try:
                v = float(v)
            except Exception:
                return 0.0
            if not np.isfinite(v):
                return 0.0
            return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

        self.left_level = _s(left)
        self.right_level = _s(right)

        # Peak-and-hold per channel with slow decay after hold
        now = time.monotonic()
        # Respect disable flag: when off, behave instantaneous (no hold)
        hold_s = 0.0 if not getattr(self, 'peak_hold_enabled', False) else (float(getattr(self, 'peak_hold_ms', 1000)) / 1000.0)
        if not hasattr(self, '_last_peak_t'):
            self._last_peak_t = now
        dt = max(0.0, now - self._last_peak_t)
        if not hasattr(self, '_peak_l_val'):
            self._peak_l_val = 0.0
            self._peak_r_val = 0.0
            self._peak_l_until = 0.0
            self._peak_r_until = 0.0
        # If hold is zero, behave instantaneous (no hold/decay)
        if hold_s <= 0.0:
            self._peak_l_val = self.left_level
            self._peak_r_val = self.right_level
            self._peak_l_until = now
            self._peak_r_until = now
        else:
            decay_a = 10.0 ** (-(getattr(self, 'peak_decay_dbps', 10.0)) * dt / 20.0)
            # Left
            if self.left_level > self._peak_l_val:
                self._peak_l_val = self.left_level
                self._peak_l_until = now + hold_s
            elif now > self._peak_l_until and self.left_level < self._peak_l_val:
                self._peak_l_val = max(self.left_level, self._peak_l_val * decay_a)
            # Right
            if self.right_level > self._peak_r_val:
                self._peak_r_val = self.right_level
                self._peak_r_until = now + hold_s
            elif now > self._peak_r_until and self.right_level < self._peak_r_val:
                self._peak_r_val = max(self.right_level, self._peak_r_val * decay_a)
        self._last_peak_t = now

        self.peak_left = max(self.peak_left * 0.95, self.left_level)
        self.peak_right = max(self.peak_right * 0.95, self.right_level)

        left_db = self._linear_to_db(self.left_level)
        right_db = self._linear_to_db(self.right_level)
        peak_db = self._linear_to_db(max(self.peak_left, self.peak_right))

        def _db_to_percent(dbv: float) -> int:
            if not np.isfinite(dbv):
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
        # Peak-hold labels per channel
        try:
            lpk = self._linear_to_db(self._peak_l_val)
            rpk = self._linear_to_db(self._peak_r_val)
        except Exception:
            lpk = -float('inf'); rpk = -float('inf')
        self.left_peak_hold_label.setText(f"Pk {lpk:.1f} dB")
        self.right_peak_hold_label.setText(f"Pk {rpk:.1f} dB")
        # Channel bytes (each channel’s LEDs as a byte)
        try:
            def _level_to_byte_db(v: float, min_db: float) -> int:
                v = 0.0 if not np.isfinite(v) else max(0.0, min(1.0, float(v)))
                dbv = -float('inf') if v <= 1e-12 else 20.0 * np.log10(v)
                if not np.isfinite(dbv) or dbv <= self.min_db:
                    n = 0
                elif dbv >= 0.0:
                    n = 8
                else:
                    pct = (dbv - self.min_db) / (0.0 - self.min_db)
                    n = int(round(pct * 8))
                n = max(0, min(8, n))
                return (1 << n) - 1 if n > 0 else 0

            l_byte = _level_to_byte_db(self.left_level, self.min_db)
            r_byte = _level_to_byte_db(self.right_level, self.min_db)
            if hasattr(self, 'left_byte_label') and hasattr(self, 'right_byte_label'):
                self.left_byte_label.setText(f"0x{l_byte:02X}")
                self.right_byte_label.setText(f"0x{r_byte:02X}")
        except Exception:
            pass
        # Tempo Ä±ÅŸÄ±klarÄ± (Sol/SaÄŸ)
        self._update_tempo('L', self.left_level)
        self._update_tempo('R', self.right_level)
        self._apply_light('L', self.l_light)
        self._apply_light('R', self.r_light)
        # BPM etiketleri
        try:
            self.left_bpm_label.setText(self._bpm_text('L'))
            self.right_bpm_label.setText(self._bpm_text('R'))
        except Exception:
            pass

    def update_bands(self, bands_tuple):
        # bands_tuple: (Llow, Lmid, Lhigh, Rlow, Rmid, Rhigh)
        if not isinstance(bands_tuple, (list, tuple)) or len(bands_tuple) != 6:
            return
        self.bands = [float(x) if np.isfinite(x) and x >= 0 else 0.0 for x in bands_tuple]

        # Peak-and-hold for bands with slow decay
        now = time.monotonic()
        hold_s = 0.0 if not getattr(self, 'peak_hold_enabled', False) else (float(getattr(self, 'peak_hold_ms', 1000)) / 1000.0)
        if not hasattr(self, '_band_peak_vals'):
            self._band_peak_vals = [0.0]*6
            self._band_peak_until = [0.0]*6
            self._band_last_t = now
        dt = max(0.0, now - getattr(self, '_band_last_t', now))
        if hold_s <= 0.0:
            # Instantaneous behavior for bands
            for i in range(6):
                self._band_peak_vals[i] = self.bands[i]
                self._band_peak_until[i] = now
        else:
            decay_a = 10.0 ** (-(getattr(self, 'peak_decay_dbps', 10.0)) * dt / 20.0)
            for i in range(6):
                cur = self.bands[i]
                if cur > self._band_peak_vals[i]:
                    self._band_peak_vals[i] = cur
                    self._band_peak_until[i] = now + hold_s
                elif now > self._band_peak_until[i] and cur < self._band_peak_vals[i]:
                    self._band_peak_vals[i] = max(cur, self._band_peak_vals[i] * decay_a)
        self._band_last_t = now

        def _db_to_percent(dbv: float) -> int:
            if not np.isfinite(dbv):
                return 0
            top = 0.0
            bottom = float(self.min_db)
            if dbv <= bottom:
                return 0
            if dbv >= top:
                return 100
            return int(round((dbv - bottom) / (top - bottom) * 100))

        Llow, Lmid, Lhigh, Rlow, Rmid, Rhigh = self.bands
        # Use peak-hold values for display
        vals = list(self._band_peak_vals)
        dbs = [self._linear_to_db(v) for v in vals]
        perc = [_db_to_percent(d) for d in dbs]

        self.l_low_bar.setValue(perc[0]); self.l_low_db.setText(f"{dbs[0]:.1f} dB")
        self.l_mid_bar.setValue(perc[1]); self.l_mid_db.setText(f"{dbs[1]:.1f} dB")
        self.l_high_bar.setValue(perc[2]); self.l_high_db.setText(f"{dbs[2]:.1f} dB")
        self.r_low_bar.setValue(perc[3]); self.r_low_db.setText(f"{dbs[3]:.1f} dB")
        self.r_mid_bar.setValue(perc[4]); self.r_mid_db.setText(f"{dbs[4]:.1f} dB")
        self.r_high_bar.setValue(perc[5]); self.r_high_db.setText(f"{dbs[5]:.1f} dB")
        # Tempo Ä±ÅŸÄ±klarÄ± (bantlar)
        keys = ['Llow','Lmid','Lhigh','Rlow','Rmid','Rhigh']
        lights = [self.l_low_light, self.l_mid_light, self.l_high_light,
                  self.r_low_light, self.r_mid_light, self.r_high_light]
        for k, v, w in zip(keys, vals, lights):
            self._update_tempo(k, v)
            self._apply_light(k, w)
        # If range-controlled lights enabled, override band lights based on range
        try:
            if hasattr(self, 'range_enable') and bool(self.range_enable.isChecked()):
                now = time.monotonic()
                if not hasattr(self, '_range_cfg'):
                    self._range_cfg = {kk: {"min_db": -60.0, "max_db": 0.0, "one_shot": False, "shot_ms": 120}
                                       for kk in keys}
                if not hasattr(self, '_range_state'):
                    self._range_state = {kk: {"armed": True} for kk in keys}
                for idx, (k, w) in enumerate(zip(keys, lights)):
                    dbv = dbs[idx]
                    cfg = self._range_cfg.get(k, {"min_db": -60.0, "max_db": 0.0, "one_shot": False, "shot_ms": 120})
                    lo = float(cfg.get("min_db", -60.0)); hi = float(cfg.get("max_db", 0.0))
                    if lo > hi:
                        lo, hi = hi, lo
                    is_finite = np.isfinite(dbv)
                    in_range = is_finite and (dbv >= lo) and (dbv <= hi)
                    below = is_finite and (dbv < lo)
                    st = self._range_state.setdefault(k, {"armed": True})
                    s = self._tempo.get(k)
                    if s is None:
                        continue
                    if bool(cfg.get("one_shot", False)):
                        # Re-arm only after dropping below min
                        if below:
                            st['armed'] = True
                        if in_range and st.get('armed', True):
                            dur = max(0, int(cfg.get("shot_ms", 120))) / 1000.0
                            s['on_until'] = now + dur
                            st['armed'] = False
                    else:
                        # While in range keep on; out of range turns off
                        if in_range:
                            s['on_until'] = now + 0.08
                        else:
                            s['on_until'] = now - 1.0
                    self._apply_light(k, w)
        except Exception:
            pass
        # BPM etiketleri (bantlar)
        try:
            self.l_low_bpm.setText(self._bpm_text('Llow'))
            self.l_mid_bpm.setText(self._bpm_text('Lmid'))
            self.l_high_bpm.setText(self._bpm_text('Lhigh'))
            self.r_low_bpm.setText(self._bpm_text('Rlow'))
            self.r_mid_bpm.setText(self._bpm_text('Rmid'))
            self.r_high_bpm.setText(self._bpm_text('Rhigh'))
        except Exception:
            pass

    def _bpm_text(self, key: str) -> str:
        s = self._tempo.get(key)
        if not s:
            return "-- BPM"
        beats = s.get('beats') or []
        if len(beats) < 2:
            return "-- BPM"
        intervals = np.diff(beats)
        intervals = intervals[intervals > 0.1]
        if intervals.size == 0:
            return "-- BPM"
        bpm = 60.0 / float(np.median(intervals))
        if not np.isfinite(bpm) or bpm <= 0:
            return "-- BPM"
        return f"{bpm:.0f} BPM"

    def set_tempo_params_for(self, key: str, *, alpha: float = None, delta: float = None,
                             min_interval_s: float = None, hold_s: float = None,
                             auto: bool = None, k: float = None):
        if not hasattr(self, '_tempo_params') or key not in self._tempo_params:
            return
        p = self._tempo_params[key]
        if alpha is not None:
            p['alpha'] = max(0.0, min(1.0, float(alpha)))
        if delta is not None:
            p['delta'] = max(0.0, min(1.0, float(delta)))
        if min_interval_s is not None:
            p['min_interval'] = max(0.0, float(min_interval_s))
        if hold_s is not None:
            p['hold'] = max(0.0, float(hold_s))
        if auto is not None:
            p['auto'] = bool(auto)
        if k is not None:
            p['k'] = max(0.0, float(k))

    def get_tempo_params_for(self, key: str):
        if not hasattr(self, '_tempo_params'):
            return {}
        return dict(self._tempo_params.get(key, {}))

    def _make_light(self) -> QLabel:
        lab = QLabel()
        lab.setFixedSize(14, 14)
        lab.setStyleSheet("border-radius:7px; background-color:#555; border:1px solid #333;")
        return lab

    def _set_light(self, lab: QLabel, on: bool):
        # Backward-compatible helper kept; replaced by _apply_light with color mapping
        color = "#39FF14" if on else "#555"
        lab.setStyleSheet(f"border-radius:7px; background-color:{color}; border:1px solid #333;")

    def _update_tempo(self, key: str, value: float):
        now = time.monotonic()
        s = self._tempo.get(key)
        if s is None:
            return
        # Per-key params and history for optional auto threshold
        p = getattr(self, '_tempo_params', {}).get(key, {'alpha':0.2,'delta':0.08,'min_interval':0.25,'hold':0.12,'auto':False,'k':0.6})
        alpha = float(p.get('alpha', 0.2))
        s['env'] = (1.0 - alpha) * s['env'] + alpha * float(value)
        hist = s.setdefault('hist', [])
        hist.append(s['env'])
        maxlen = s.setdefault('hist_maxlen', 120)
        if len(hist) > maxlen:
            del hist[0:len(hist)-maxlen]
        if bool(p.get('auto')) and len(hist) >= 10:
            std = float(np.std(hist))
            delta = max(0.01, float(p.get('k', 0.6)) * std)
        else:
            delta = float(p.get('delta', 0.08))
        min_interval = float(p.get('min_interval', 0.25))  # s
        hold = float(p.get('hold', 0.12))  # s
        if float(value) > s['env'] + delta and (now - s['last_flash'] > min_interval):
            s['on_until'] = now + hold
            s['last_flash'] = now
            # BPM tahmini iÃ§in zaman damgasÄ± kaydÄ± (son 10 sn)
            beats = s.setdefault('beats', [])
            beats.append(now)
            cutoff = now - 10.0
            while beats and beats[0] < cutoff:
                beats.pop(0)
        s['last'] = float(value)

    def _apply_light(self, key: str, lab: QLabel):
        now = time.monotonic()
        s = self._tempo.get(key)
        if s is None:
            return
        on = now < s.get('on_until', 0.0)
        # Renk seÃ§imi: genel L/R yeÅŸil; low=mavi, mid=sarÄ±, high=kÄ±rmÄ±zÄ±
        if key in ('L', 'R'):
            color = "#39FF14"
        elif 'low' in key.lower():
            color = "#1E90FF"
        elif 'mid' in key.lower():
            color = "#FFD700"
        elif 'high' in key.lower():
            color = "#FF4500"
        else:
            color = "#39FF14"
        col = color if on else "#555"
        lab.setStyleSheet(f"border-radius:7px; background-color:{col}; border:1px solid #333;")


class VUMeterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.audio_monitor = AudioMonitor()
        self._last_left = 0.0
        self._last_right = 0.0
        
        # DMX Controller entegrasyonu
        self.dmx_controller = DMXController(logger_instance=logger)
        
        self._init_ui()

        self.audio_monitor.level_updated.connect(self.on_level_updated)
        self.audio_monitor.error_occurred.connect(self.on_error)
        self.audio_monitor.bands_updated.connect(self.on_bands_updated)

        self.gui_timer = QTimer(self)
        self.gui_timer.setInterval(8)  # ~120 Hz
        self.gui_timer.timeout.connect(self._on_gui_tick)
        self._last_bands = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # Peak blink state per target (LH, LM, LL, L, R, RL, RM, RH)
        self._peak_blink = {}

        # LED bits window and serial sender
        self.led_window = LedBitsWindow()
        try:
            self.led_window.show()
        except Exception:
            pass
        # Big lights window (large indicators for L/R and 6 bands)
        try:
            self.big_lights_window = BigLightsWindow()
            self.big_lights_window.show()
        except Exception:
            self.big_lights_window = None
        self.ser = None
        self._serial_port_name = None
        self._serial_baud = 9600
        self._init_serial_auto()

    def _init_ui(self):
        self.setWindowTitle(f"Audio VU Meter v{__version__}")
        self.setGeometry(100, 100, 620, 260)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()

        title = QLabel("GerÃ§ek ZamanlÄ± VU Meter - Mikrofon & Sistem Sesi")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        device_layout = QHBoxLayout()
        device_label = QLabel("Ses KartÄ±:")
        self.device_combo = QComboBox()
        self.refresh_devices()
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)

        range_label = QLabel("AralÄ±k:")
        self.range_combo = QComboBox()
        for val in (-90, -60, -48, -40, -30):
            self.range_combo.addItem(f"{val} dB", float(val))
        self.range_combo.setCurrentIndex(1)
        self.range_combo.currentIndexChanged.connect(self.on_range_changed)
        device_layout.addWidget(range_label)
        device_layout.addWidget(self.range_combo)

        fps_label = QLabel("FPS:")
        # Allow fractional FPS down to 0.1
        self.fps_dspin = QDoubleSpinBox()
        self.fps_dspin.setDecimals(1)
        self.fps_dspin.setRange(0.1, 1000.0)
        self.fps_dspin.setSingleStep(0.1)
        self.fps_dspin.setValue(120.0)
        self.fps_dspin.valueChanged.connect(self.on_fps_changed)
        device_layout.addWidget(fps_label)
        device_layout.addWidget(self.fps_dspin)
        
        # Tempo hedefi ve otomatik eÅŸik
        target_label = QLabel("Tempo Hedefi:")
        self.tempo_target = QComboBox()
        for key in ("L","R","Llow","Lmid","Lhigh","Rlow","Rmid","Rhigh"):
            self.tempo_target.addItem(key, key)
        self.tempo_target.currentIndexChanged.connect(self.on_tempo_target_changed)
        self.tempo_auto = QCheckBox("Oto")
        self.tempo_auto.stateChanged.connect(self.on_tempo_params_changed)
        device_layout.addWidget(target_label)
        device_layout.addWidget(self.tempo_target)
        device_layout.addWidget(self.tempo_auto)

        # Tempo parametreleri (EÅŸik, Hold, Min, Alfa)
        thr_label = QLabel("EÅŸik:")
        self.tempo_thr = QDoubleSpinBox()
        self.tempo_thr.setRange(0.0, 1.0)
        self.tempo_thr.setSingleStep(0.01)
        self.tempo_thr.setValue(0.08)

        hold_label = QLabel("Hold ms:")
        self.tempo_hold = QDoubleSpinBox()
        self.tempo_hold.setRange(0.0, 1000.0)
        self.tempo_hold.setSingleStep(10.0)
        self.tempo_hold.setValue(120.0)

        min_label = QLabel("Min ms:")
        self.tempo_min = QDoubleSpinBox()
        self.tempo_min.setRange(0.0, 2000.0)
        self.tempo_min.setSingleStep(10.0)
        self.tempo_min.setValue(250.0)

        alpha_label = QLabel("Alfa:")
        self.tempo_alpha = QDoubleSpinBox()
        self.tempo_alpha.setRange(0.0, 1.0)
        self.tempo_alpha.setSingleStep(0.01)
        self.tempo_alpha.setValue(0.20)

        for w in (thr_label, self.tempo_thr, hold_label, self.tempo_hold,
                  min_label, self.tempo_min, alpha_label, self.tempo_alpha):
            device_layout.addWidget(w)

        # DeÄŸer deÄŸiÅŸince uygula
        self.tempo_thr.valueChanged.connect(self.on_tempo_params_changed)
        self.tempo_hold.valueChanged.connect(self.on_tempo_params_changed)
        self.tempo_min.valueChanged.connect(self.on_tempo_params_changed)
        self.tempo_alpha.valueChanged.connect(self.on_tempo_params_changed)

        device_layout.addStretch()
        layout.addLayout(device_layout)
        
        # DMX Kontrol Paneli
        dmx_frame = QHBoxLayout()
        dmx_label = QLabel("🎭 DMX Kontrol:")
        dmx_label.setStyleSheet("font-weight:bold; color:#FF6B35;")
        dmx_frame.addWidget(dmx_label)
        
        self.dmx_device_combo = QComboBox()
        self.dmx_device_combo.setMinimumWidth(200)
        dmx_frame.addWidget(self.dmx_device_combo)
        
        self.dmx_refresh_btn = QPushButton("Yenile")
        self.dmx_refresh_btn.clicked.connect(self.refresh_dmx_devices)
        dmx_frame.addWidget(self.dmx_refresh_btn)
        
        self.dmx_connect_btn = QPushButton("DMX Bağlan")
        self.dmx_connect_btn.clicked.connect(self.toggle_dmx_connection)
        self.dmx_connect_btn.setStyleSheet("background-color:#4CAF50; color:white; padding:5px;")
        dmx_frame.addWidget(self.dmx_connect_btn)
        
        self.dmx_status_label = QLabel("DMX: Kapalı")
        self.dmx_status_label.setStyleSheet("color:red; font-weight:bold;")
        dmx_frame.addWidget(self.dmx_status_label)
        
        dmx_frame.addStretch()
        layout.addLayout(dmx_frame)
        
        # DMX bilgi satırı
        dmx_info = QLabel("🎨 DMX: Ch3=MANUEL Renk | Ch5+Ch6=Llow Range Scaled (Min/Max dB) + Beat Flash | Ch1+Ch2=Manuel Pan/Tilt")
        dmx_info.setStyleSheet("font-size:9px; color:#673AB7; font-weight:bold; padding:3px; background-color:#EDE7F6; border-radius:3px;")
        layout.addWidget(dmx_info)
        
        # DMX Manuel Kontroller (Ch1 Pan, Ch2 Tilt)
        dmx_manual_frame = QHBoxLayout()
        
        # Channel 1: Pan (Horizontal)
        ch1_label = QLabel("Ch1 Pan:")
        ch1_label.setFixedWidth(60)
        dmx_manual_frame.addWidget(ch1_label)
        
        self.dmx_ch1_slider = QSpinBox()
        self.dmx_ch1_slider.setRange(0, 255)
        self.dmx_ch1_slider.setValue(127)
        self.dmx_ch1_slider.setToolTip("Channel 1: Horizontal Rotation (0-255)")
        self.dmx_ch1_slider.valueChanged.connect(self.on_dmx_manual_changed)
        dmx_manual_frame.addWidget(self.dmx_ch1_slider)
        
        # Channel 2: Tilt (Vertical)
        ch2_label = QLabel("Ch2 Tilt:")
        ch2_label.setFixedWidth(60)
        dmx_manual_frame.addWidget(ch2_label)
        
        self.dmx_ch2_slider = QSpinBox()
        self.dmx_ch2_slider.setRange(0, 255)
        self.dmx_ch2_slider.setValue(127)
        self.dmx_ch2_slider.setToolTip("Channel 2: Vertical Rotation (0-255)")
        self.dmx_ch2_slider.valueChanged.connect(self.on_dmx_manual_changed)
        dmx_manual_frame.addWidget(self.dmx_ch2_slider)
        
        # Reset butonu
        dmx_reset_btn = QPushButton("Merkez (127)")
        dmx_reset_btn.setToolTip("Pan ve Tilt'i merkeze (127, 127) getir")
        dmx_reset_btn.clicked.connect(self.dmx_reset_position)
        dmx_manual_frame.addWidget(dmx_reset_btn)
        
        dmx_manual_frame.addStretch()
        layout.addLayout(dmx_manual_frame)
        
        # DMX Renk Kontrolü (Channel 3)
        dmx_color_frame = QHBoxLayout()
        
        ch3_label = QLabel("Ch3 Renk:")
        ch3_label.setFixedWidth(70)
        dmx_color_frame.addWidget(ch3_label)
        
        self.dmx_ch3_slider = QSpinBox()
        self.dmx_ch3_slider.setRange(0, 255)
        self.dmx_ch3_slider.setValue(5)  # Default: White
        self.dmx_ch3_slider.setToolTip("Channel 3: Color (0-9=Beyaz, 10+=Renkler)")
        self.dmx_ch3_slider.valueChanged.connect(self.on_dmx_manual_changed)
        dmx_color_frame.addWidget(self.dmx_ch3_slider)
        
        # Renk preset butonları
        preset_white = QPushButton("⚪ Beyaz (5)")
        preset_white.setToolTip("Beyaz renk")
        preset_white.clicked.connect(lambda: self.dmx_ch3_slider.setValue(5))
        dmx_color_frame.addWidget(preset_white)
        
        preset_red = QPushButton("🔴 Kırmızı (15)")
        preset_red.setToolTip("Kırmızı renk")
        preset_red.clicked.connect(lambda: self.dmx_ch3_slider.setValue(15))
        dmx_color_frame.addWidget(preset_red)
        
        preset_orange = QPushButton("🟠 Turuncu (25)")
        preset_orange.setToolTip("Turuncu renk")
        preset_orange.clicked.connect(lambda: self.dmx_ch3_slider.setValue(25))
        dmx_color_frame.addWidget(preset_orange)
        
        preset_yellow = QPushButton("🟡 Sarı (35)")
        preset_yellow.setToolTip("Sarı renk")
        preset_yellow.clicked.connect(lambda: self.dmx_ch3_slider.setValue(35))
        dmx_color_frame.addWidget(preset_yellow)
        
        preset_green = QPushButton("🟢 Yeşil (50)")
        preset_green.setToolTip("Yeşil renk")
        preset_green.clicked.connect(lambda: self.dmx_ch3_slider.setValue(50))
        dmx_color_frame.addWidget(preset_green)
        
        preset_blue = QPushButton("🔵 Mavi (90)")
        preset_blue.setToolTip("Mavi renk")
        preset_blue.clicked.connect(lambda: self.dmx_ch3_slider.setValue(90))
        dmx_color_frame.addWidget(preset_blue)
        
        dmx_color_frame.addStretch()
        layout.addLayout(dmx_color_frame)
        
        # DMX cihazlarını listele
        self.refresh_dmx_devices()

        # Inline LED bit panel (8 bytes) with labels
        inline_box = QVBoxLayout()
        inline_header = QLabel("LED Columns (Inline, bottom-up)")
        inline_header.setStyleSheet("font-weight:bold; padding:4px;")
        inline_box.addWidget(inline_header)
        inline_cols_row = QHBoxLayout()
        # Column order left-to-right: LH LM LL L R RL RM RH
        self._inline_col_labels = [
            ("LH", "L High", 'Lhigh'), ("LM", "L Mid", 'Lmid'), ("LL", "L Low", 'Llow'),
            ("L", "L", 'L'), ("R", "R", 'R'), ("RL", "R Low", 'Rlow'),
            ("RM", "R Mid", 'Rmid'), ("RH", "R High", 'Rhigh')
        ]
        self.inline_led_cols = []  # list of (bit_labels_bottom_up[8], hex_label)
        for code, label, _key in self._inline_col_labels:
            col = QVBoxLayout()
            col.setSpacing(2)
            title = QLabel(f"{code}")
            title.setStyleSheet("font-size:12px; font-weight:bold;")
            col.addWidget(title)
            bits = []
            # Build bottom-up: add spacer, then bits from top to bottom for layout,
            # but store list in bottom-up order for easy addressing
            for _ in range(8):
                lab = QLabel()
                lab.setFixedSize(12, 12)
                lab.setStyleSheet("border:1px solid #333; background:#222;")
                col.addWidget(lab)
                bits.append(lab)
            # bits currently top-down; reverse to get bottom-up indexing
            bits_bottom_up = list(reversed(bits))
            hx = QLabel("0x00")
            hx.setFixedWidth(44)
            hx.setStyleSheet("padding:2px;")
            col.addWidget(hx)
            inline_cols_row.addLayout(col)
            self.inline_led_cols.append((bits_bottom_up, hx))
        inline_box.addLayout(inline_cols_row)
        # Separate row to display per-column light bytes (on/off)
        lights_header = QLabel("Light Bytes (per column)")
        lights_header.setStyleSheet("font-weight:bold; padding:4px;")
        inline_box.addWidget(lights_header)
        lights_row = QHBoxLayout()
        self._light_hex_labels = []
        for _ in range(8):
            hx = QLabel("0x00")
            hx.setFixedWidth(44)
            hx.setStyleSheet("padding:2px;")
            lights_row.addWidget(hx)
            self._light_hex_labels.append(hx)
        lights_row.addStretch()
        inline_box.addLayout(lights_row)
        layout.addLayout(inline_box)

        self.vu_widget = VUMeterWidget()
        layout.addWidget(self.vu_widget)
        # VarsayÄ±lan tempo parametrelerini uygula
        try:
            self.on_tempo_params_changed()
        except Exception:
            pass

        buttons = QHBoxLayout()
        self.start_button = QPushButton("BaÅŸlat")
        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button = QPushButton("Durdur")
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.stop_button.setEnabled(False)
        self.refresh_button = QPushButton("Yenile")
        self.refresh_button.clicked.connect(self.refresh_devices)
        for b in (self.start_button, self.stop_button, self.refresh_button):
            b.setStyleSheet("padding: 10px; font-size: 14px;")
            buttons.addWidget(b)
        layout.addLayout(buttons)

        self.status_label = QLabel("HazÄ±r")
        self.status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        layout.addWidget(self.status_label)

        central.setLayout(layout)

    def refresh_devices(self):
        self.device_combo.clear()
        for device in AudioMonitor.get_audio_devices():
            self.device_combo.addItem(f"{device['name']} ({device['channels']} kanal)", device)

    def start_monitoring(self):
        info = self.device_combo.currentData()
        if info is not None:
            self.audio_monitor.device_index = info['index']
            self.audio_monitor.use_loopback = info.get('is_loopback', False)
        self.audio_monitor.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.device_combo.setEnabled(False)
        self.gui_timer.start()
        if self.audio_monitor.use_loopback:
            self.status_label.setText("Sistem sesi dinleniyor (Edge, YouTube vb.)...")
        else:
            self.status_label.setText("Mikrofon dinleniyor...")
        self.status_label.setStyleSheet("padding: 5px; background-color: #90EE90;")

    def stop_monitoring(self):
        self.audio_monitor.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.device_combo.setEnabled(True)
        self.status_label.setText("Durduruldu")
        self.status_label.setStyleSheet("padding: 5px; background-color: #FFB6C1;")
        self.vu_widget.update_levels(0, 0)
        self.gui_timer.stop()

    def on_level_updated(self, left, right):
        try:
            self._last_left = float(left)
            self._last_right = float(right)
        except Exception:
            self._last_left = 0.0
            self._last_right = 0.0

    def _on_gui_tick(self):
        self.vu_widget.update_levels(self._last_left, self._last_right)
        self.vu_widget.update_bands(self._last_bands)
        
        # DMX güncelleme - BASIT MOD: Llow seviyesi + Beat Flash
        if hasattr(self, 'dmx_controller') and self.dmx_controller.enabled:
            try:
                b = list(self._last_bands) if isinstance(self._last_bands, (list, tuple)) else [0]*6
                while len(b) < 6:
                    b.append(0.0)
                llow, lmid, lhigh, rlow, rmid, rhigh = b[:6]
                
                # Beat flash kontrolü: Llow tempo state'inden beat tespiti
                beat_flash = False
                try:
                    now = time.monotonic()
                    tempo = getattr(self.vu_widget, '_tempo', {})
                    llow_state = tempo.get('Llow')
                    if llow_state:
                        # Beat tespit edildi mi? (on_until aktif ve çok kısa süre geçti)
                        on_until = float(llow_state.get('on_until', 0.0))
                        last_flash = float(llow_state.get('last_flash', 0.0))
                        
                        # Beat'ten sonraki ilk 50ms'de flash aktif
                        if now < on_until and (now - last_flash) < 0.05:
                            beat_flash = True
                except Exception:
                    pass
                
                # Range config'i al (Llow için)
                range_config = None
                try:
                    if hasattr(self.vu_widget, '_range_cfg'):
                        llow_cfg = self.vu_widget._range_cfg.get('Llow', {})
                        if llow_cfg:
                            range_config = {
                                'min_db': float(llow_cfg.get('min_db', -60.0)),
                                'max_db': float(llow_cfg.get('max_db', 0.0))
                            }
                except Exception:
                    pass
                
                # DMX gönder (beat_flash ve range_config ile)
                self.dmx_controller.set_audio_reactive(
                    llow, lmid, lhigh, rlow, rmid, rhigh, None, beat_flash, range_config
                )
            except Exception as e:
                pass  # Sessizce devam et
        # Build 8 bytes for LED output and display
        try:
            frame = self._build_led_bytes()
            if hasattr(self, 'led_window') and self.led_window:
                self.led_window.update_bits(frame)
            # Inline vertical columns update (bottom-up)
            try:
                for i in range(8):
                    val = int(frame[i]) & 0xFF
                    bits_bottom_up, hx = self.inline_led_cols[i]
                    for j in range(8):
                        on = (val & (1 << j)) != 0  # j=0 bottom LED (LSB)
                        bits_bottom_up[j].setStyleSheet(
                            "border:1px solid #333; background:" + ("#39FF14" if on else "#222") + ";"
                        )
                    hx.setText(f"0x{val:02X}")
            except Exception:
                pass
            self._send_led_bytes(frame)
            # Build and show/send light-only bytes (tempo lights)
            try:
                light_frame = self._build_light_bytes()
            except Exception:
                light_frame = [0x00]*8
            try:
                # Update hex labels for lights
                if hasattr(self, '_light_hex_labels'):
                    for i in range(8):
                        self._light_hex_labels[i].setText(f"0x{(int(light_frame[i]) & 0xFF):02X}")
            except Exception:
                pass
            try:
                self._send_light_bytes(light_frame)
            except Exception:
                pass
            # Update big lights window colors
            try:
                if hasattr(self, 'big_lights_window') and self.big_lights_window:
                    now = time.monotonic()
                    tempo = getattr(self.vu_widget, '_tempo', {})
                    # Order: L, R, Llow, Lmid, Lhigh, Rlow, Rmid, Rhigh
                    keys = ['L','R','Llow','Lmid','Lhigh','Rlow','Rmid','Rhigh']
                    states = {}
                    for k in keys:
                        s = tempo.get(k)
                        on = (s is not None) and (now < float(s.get('on_until', 0.0)))
                        states[k] = on
                    self.big_lights_window.update_lights(states)
            except Exception:
                pass
        except Exception:
            pass

    def on_range_changed(self):
        data = self.range_combo.currentData()
        if data is not None:
            self.vu_widget.set_min_db(float(data))

    def on_fps_changed(self):
        # Prefer double spin (0.1..1000), fallback to int spin or combo
        fps = 120.0
        try:
            if hasattr(self, 'fps_dspin') and self.fps_dspin is not None:
                fps = float(self.fps_dspin.value())
            elif hasattr(self, 'fps_spin') and self.fps_spin is not None:
                fps = float(int(self.fps_spin.value()))
            elif hasattr(self, 'fps_combo') and self.fps_combo is not None:
                data = self.fps_combo.currentData()
                fps = float(int(data) if data is not None else 120)
        except Exception:
            fps = 120.0
        if fps < 0.1:
            fps = 0.1
        if fps > 1000.0:
            fps = 1000.0
        interval_ms = max(1, int(round(1000.0 / fps)))
        self.gui_timer.setInterval(interval_ms)

    def on_bands_updated(self, bands_tuple):
        try:
            if isinstance(bands_tuple, (list, tuple)) and len(bands_tuple) == 6:
                self._last_bands = tuple(float(x) for x in bands_tuple)
        except Exception:
            self._last_bands = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _build_led_bytes(self):
        """Map current levels to 8 bytes (vertical bars) using dB range."""
        def level_to_byte_db(v: float, min_db: float) -> int:
            # Convert linear level to dB and map to 0..8 LEDs based on min_db..0dB
            v = 0.0 if not np.isfinite(v) else max(0.0, min(1.0, float(v)))
            db = -float('inf')
            if v > 1e-12:
                db = 20.0 * np.log10(v)
            if not np.isfinite(db) or db <= min_db:
                n = 0
            elif db >= 0.0:
                n = 8
            else:
                pct = (db - min_db) / (0.0 - min_db)
                n = int(round(pct * 8))
            n = max(0, min(8, n))
            return (1 << n) - 1 if n > 0 else 0

        min_db = float(getattr(self.vu_widget, 'min_db', -60.0))
        L = level_to_byte_db(self._last_left, min_db)
        R = level_to_byte_db(self._last_right, min_db)
        b = list(self._last_bands) if isinstance(self._last_bands, (list, tuple)) else [0]*6
        while len(b) < 6:
            b.append(0.0)
        Llow, Lmid, Lhigh, Rlow, Rmid, Rhigh = b[:6]
        # Order: LH LM LL L R RL RM RH
        bytes_list = [
            level_to_byte_db(Lhigh, min_db),
            level_to_byte_db(Lmid, min_db),
            level_to_byte_db(Llow, min_db),
            L,
            R,
            level_to_byte_db(Rlow, min_db),
            level_to_byte_db(Rmid, min_db),
            level_to_byte_db(Rhigh, min_db),
        ]
        # Overlay beat: set LSB (0x01) when beat is active for that target
        try:
            # Beat key order must match bytes_list order above
            keys = ['Lhigh','Lmid','Llow','L','R','Rlow','Rmid','Rhigh']
            now = time.monotonic()
            tempo = getattr(self.vu_widget, '_tempo', {})
            for i, k in enumerate(keys):
                s = tempo.get(k)
                if s and now < float(s.get('on_until', 0.0)):
                    # Ensure the lowest bit is set to indicate beat, do not touch MSB
                    bytes_list[i] = int(bytes_list[i]) | 0x01
        except Exception:
            pass
        # Overlay peak-return blink: set MSB (0x80) briefly when peak falls
        try:
            # Prepare linear levels in the same order as keys
            linear_levels = [Lhigh, Lmid, Llow, self._last_left, self._last_right, Rlow, Rmid, Rhigh]
            now = time.monotonic()
            peak_delta = 0.05  # trigger when drop exceeds 5% linear
            hold_s = 0.12
            for i, k in enumerate(['Lhigh','Lmid','Llow','L','R','Rlow','Rmid','Rhigh']):
                cur = float(linear_levels[i]) if np.isfinite(linear_levels[i]) else 0.0
                st = self._peak_blink.get(k)
                if st is None:
                    st = {'max': 0.0, 'on_until': 0.0}
                # rising peak
                if cur > st['max']:
                    st['max'] = cur
                # falling enough -> blink
                elif st['max'] - cur >= peak_delta:
                    st['on_until'] = now + hold_s
                    st['max'] = cur
                # apply overlay if active
                if now < st.get('on_until', 0.0):
                    bytes_list[i] = int(bytes_list[i]) | 0x80
                self._peak_blink[k] = st
        except Exception:
            pass
        return bytes_list

    def _build_light_bytes(self):
        """Build 8 bytes representing only the on/off state of lights for
        [Lhigh, Lmid, Llow, L, R, Rlow, Rmid, Rhigh]. Each byte is 0x01 if the
        corresponding light is ON, else 0x00."""
        try:
            keys = ['Lhigh','Lmid','Llow','L','R','Rlow','Rmid','Rhigh']
            now = time.monotonic()
            tempo = getattr(self.vu_widget, '_tempo', {})
            out = []
            for k in keys:
                s = tempo.get(k)
                on = (s is not None) and (now < float(s.get('on_until', 0.0)))
                out.append(0x01 if on else 0x00)
            return out
        except Exception:
            return [0x00]*8

    def _init_serial_auto(self):
        """Try to auto-pick first available serial port at 9600."""
        try:
            if serial is None or list_ports is None:
                return
            ports = list(list_ports.comports())
            if ports:
                self._serial_port_name = ports[0].device
                self._serial_baud = 9600
                self._ensure_serial()
        except Exception:
            pass

    def _ensure_serial(self):
        try:
            if serial is None:
                return
            if self.ser and self.ser.is_open:
                return
            if not self._serial_port_name:
                return
            self.ser = serial.Serial(self._serial_port_name, self._serial_baud, timeout=0)
        except Exception:
            try:
                if self.ser:
                    self.ser.close()
            except Exception:
                pass
            self.ser = None

    def _send_led_bytes(self, bytes_list):
        try:
            if not isinstance(bytes_list, (list, tuple)) or len(bytes_list) != 8:
                return
            self._ensure_serial()
            if not self.ser or not self.ser.is_open:
                return
            payload = bytes([0xA0]) + bytes(int(x) & 0xFF for x in bytes_list)
            self.ser.write(payload)
        except Exception:
            pass

    def _send_light_bytes(self, bytes_list):
        """Send an additional 8-byte frame for light states."""
        try:
            if not isinstance(bytes_list, (list, tuple)) or len(bytes_list) != 8:
                return
            self._ensure_serial()
            if not self.ser or not self.ser.is_open:
                return
            payload = bytes([0xB0]) + bytes(int(x) & 0xFF for x in bytes_list)
            self.ser.write(payload)
        except Exception:
            pass

    def on_peak_hold_changed(self):
        try:
            val = int(self.vu_widget.peak_hold_ms) if hasattr(self.vu_widget, 'peak_hold_ms') else 1000
        except Exception:
            val = 1000
        try:
            new_val = int(self.peak_hold_spin.value())
        except Exception:
            new_val = val
        try:
            self.vu_widget.set_peak_hold_ms(new_val)
        except Exception:
            pass

    # ---- Tempo UI handlers (per-target) ----
    def on_tempo_params_changed(self):
        pass

    def on_tempo_target_changed(self):
        pass

    def on_error(self, error_msg):
        pass
    
    def refresh_dmx_devices(self):
        """DMX cihazlarını yenile"""
        try:
            if not DMX_AVAILABLE:
                self.dmx_device_combo.clear()
                self.dmx_device_combo.addItem("pyusb yüklü değil")
                return
            
            devices = self.dmx_controller.find_udmx_devices()
            self.dmx_device_combo.clear()
            
            if devices:
                for dev in devices:
                    self.dmx_device_combo.addItem(f"{dev['name']} (VID:{dev['vendor']:04X} PID:{dev['product']:04X})")
                logger.info(f"{len(devices)} DMX cihazı bulundu")
            else:
                self.dmx_device_combo.addItem("UDMX cihazı bulunamadı")
                logger.warning("DMX cihazı bulunamadı")
        except Exception as e:
            logger.error(f"DMX cihaz listesi hatası: {e}")
            self.dmx_device_combo.clear()
            self.dmx_device_combo.addItem("Hata: Cihaz bulunamadı")
    
    def toggle_dmx_connection(self):
        """DMX bağlantısını aç/kapat"""
        try:
            if not self.dmx_controller.enabled:
                # Bağlan
                device_index = self.dmx_device_combo.currentIndex()
                if device_index < 0:
                    return
                
                if self.dmx_controller.connect(device_index):
                    self.dmx_connect_btn.setText("DMX Kes")
                    self.dmx_connect_btn.setStyleSheet("background-color:#f44336; color:white; padding:5px;")
                    self.dmx_status_label.setText("DMX: Aktif ✓")
                    self.dmx_status_label.setStyleSheet("color:#4CAF50; font-weight:bold;")
                    logger.info("DMX bağlantısı kuruldu")
                    # İlk bağlantıda manuel değerleri gönder
                    self.on_dmx_manual_changed()
                else:
                    self.dmx_status_label.setText("DMX: Bağlantı Hatası")
                    self.dmx_status_label.setStyleSheet("color:orange; font-weight:bold;")
                    logger.error("DMX bağlantısı kurulamadı")
            else:
                # Kes
                self.dmx_controller.disconnect()
                self.dmx_connect_btn.setText("DMX Bağlan")
                self.dmx_connect_btn.setStyleSheet("background-color:#4CAF50; color:white; padding:5px;")
                self.dmx_status_label.setText("DMX: Kapalı")
                self.dmx_status_label.setStyleSheet("color:red; font-weight:bold;")
                logger.info("DMX bağlantısı kesildi")
        except Exception as e:
            logger.error(f"DMX bağlantı hatası: {e}")
    
    def on_dmx_manual_changed(self):
        """DMX manuel kontrol değiştiğinde (Ch1 Pan, Ch2 Tilt, Ch3 Renk)"""
        try:
            if hasattr(self, 'dmx_controller') and self.dmx_controller.enabled:
                ch1_value = int(self.dmx_ch1_slider.value())
                ch2_value = int(self.dmx_ch2_slider.value())
                ch3_value = int(self.dmx_ch3_slider.value()) if hasattr(self, 'dmx_ch3_slider') else 5
                
                self.dmx_controller.set_channel(1, ch1_value)
                self.dmx_controller.set_channel(2, ch2_value)
                self.dmx_controller.set_channel(3, ch3_value)
                self.dmx_controller.send_frame()
                logger.debug(f"DMX Manuel: Ch1={ch1_value}, Ch2={ch2_value}, Ch3={ch3_value}")
        except Exception as e:
            logger.error(f"DMX manuel kontrol hatası: {e}")
    
    def dmx_reset_position(self):
        """DMX Pan/Tilt pozisyonunu merkeze getir"""
        try:
            self.dmx_ch1_slider.setValue(127)
            self.dmx_ch2_slider.setValue(127)
            logger.info("DMX pozisyonu merkeze getirildi (127, 127)")
        except Exception as e:
            logger.error(f"DMX reset hatası: {e}")

    def closeEvent(self, event):
        try:
            if hasattr(self, 'vu_widget'):
                self.vu_widget.save_range_cfg()
        except Exception:
            pass
        try:
            self.audio_monitor.stop()
        except Exception:
            pass
        try:
            if hasattr(self, 'dmx_controller'):
                self.dmx_controller.disconnect()
        except Exception:
            pass
        event.accept()


class LedBitsWindow(QWidget):
    """Simple window to display 8 bytes as 8 bits."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LED Bytes (8x8)")
        self.setGeometry(150, 150, 240, 220)
        layout = QVBoxLayout()
        mapping = [
            ("B0", "L Low"), ("B1", "L Mid"), ("B2", "L High"),
            ("B3", "R Low"), ("B4", "R Mid"), ("B5", "R High"),
            ("B6", "L"),     ("B7", "R")
        ]
        self.rows = []
        for i in range(8):
            row = QHBoxLayout()
            label_text = f"{mapping[i][0]} ({mapping[i][1]}):"
            row.addWidget(QLabel(label_text))
            bits = []
            for _ in range(8):
                lab = QLabel()
                lab.setFixedSize(14, 14)
                lab.setStyleSheet("border:1px solid #333; background:#222;")
                row.addWidget(lab)
                bits.append(lab)
            hex_lbl = QLabel("0x00")
            hex_lbl.setFixedWidth(40)
            row.addWidget(hex_lbl)
            row.addStretch()
            layout.addLayout(row)
            self.rows.append((bits, hex_lbl))
        self.setLayout(layout)

    def update_bits(self, byte_list):
        if not isinstance(byte_list, (list, tuple)) or len(byte_list) != 8:
            return
        # Fallback: if vertical columns not initialized, keep legacy rows
        if hasattr(self, '_cols'):
            for i in range(8):
                val = int(byte_list[i]) & 0xFF
                bits_bottom_up, hex_lbl = self._cols[i]
                for j in range(8):
                    on = (val & (1 << j)) != 0  # j=0 bottom LED
                    bits_bottom_up[j].setStyleSheet(
                        "border:1px solid #333; background:" + ("#39FF14" if on else "#222") + ";"
                    )
                hex_lbl.setText(f"0x{val:02X}")
        else:
            for i in range(8):
                val = int(byte_list[i]) & 0xFF
                bits, hex_lbl = self.rows[i]
                for j in range(8):
                    mask = 1 << (7 - j)
                    on = (val & mask) != 0
                    bits[j].setStyleSheet(
                        "border:1px solid #333; background:" + ("#39FF14" if on else "#222") + ";"
                    )
                hex_lbl.setText(f"0x{val:02X}")

    # No tempo or status handling here; this window only renders bytes.

    def closeEvent(self, event):
        event.accept()


class BigLightsWindow(QWidget):
    """Separate window to display large lights for L/R and 6 bands."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Big Lights")
        self.setGeometry(180, 180, 520, 180)
        layout = QVBoxLayout()
        title = QLabel("Large Channel/Band Lights")
        title.setStyleSheet("font-weight:bold; padding:6px;")
        layout.addWidget(title)
        row1 = QHBoxLayout()
        row2 = QHBoxLayout()
        # Create lights and labels in two rows
        self._keys = ['L','R','Llow','Lmid','Lhigh','Rlow','Rmid','Rhigh']
        self._labs = {}
        def make_light(text):
            box = QVBoxLayout()
            lamp = QLabel()
            lamp.setFixedSize(42, 42)
            lamp.setStyleSheet("border-radius:21px; background-color:#555; border:2px solid #333;")
            cap = QLabel(text)
            cap.setStyleSheet("padding-top:4px; text-align:center;")
            cap.setFixedWidth(70)
            cap.setAlignment(Qt.AlignCenter) if 'Qt' in globals() else None
            box.addWidget(lamp)
            box.addWidget(cap)
            return box, lamp
        # First row: L, R, Llow, Lmid
        for txt in ('L','R','L Low','L Mid'):
            box, lamp = make_light(txt)
            row1.addLayout(box)
            key = {'L':'L','R':'R','L Low':'Llow','L Mid':'Lmid'}[txt]
            self._labs[key] = lamp
        # Second row: Lhigh, Rlow, Rmid, Rhigh
        for txt in ('L High','R Low','R Mid','R High'):
            box, lamp = make_light(txt)
            row2.addLayout(box)
            key = {'L High':'Lhigh','R Low':'Rlow','R Mid':'Rmid','R High':'Rhigh'}[txt]
            self._labs[key] = lamp
        layout.addLayout(row1)
        layout.addLayout(row2)
        self.setLayout(layout)

    def _color_for_key(self, key: str, on: bool) -> str:
        if not on:
            return "#555"
        if key in ('L','R'):
            return "#39FF14"
        lk = key.lower()
        if 'low' in lk:
            return "#1E90FF"
        if 'mid' in lk:
            return "#FFD700"
        if 'high' in lk:
            return "#FF4500"
        return "#39FF14"

    def update_lights(self, states: dict):
        try:
            for k, lamp in self._labs.items():
                on = bool(states.get(k, False)) if isinstance(states, dict) else False
                col = self._color_for_key(k, on)
                lamp.setStyleSheet(f"border-radius:21px; background-color:{col}; border:2px solid #333;")
        except Exception:
            pass


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VUMeterApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


