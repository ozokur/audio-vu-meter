#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio VU Meter GUI (120 FPS + Ayarlanabilir dB Aralığı)
"""

import sys
import os
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import time
try:
    import pyaudiowpatch as pyaudio
except ImportError:
    import pyaudio
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QComboBox, QProgressBar, QDoubleSpinBox, QCheckBox
)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject

__version__ = "1.3.0"


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
        self.chunk_size = 256  # 120 Hz GUI için daha sık callback
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
            logger.exception("Ses kartı başlatma hatası")
            self.error_occurred.emit(f"Ses kartı hatası: {e}")

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

            # 3 bant (Low/Mid/High) için spektral güçten RMS tahmini
            try:
                if 'data' in locals() and data is not None and data.size > 0:
                    # Kanal ayrımı
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
                # Spektral hesap hatası UI'ı durdurmasın
                pass
        except Exception:
            logger.exception("Audio callback hatası")
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
                logger.warning(f"WASAPI loopback bulunamadı: {e}")
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0 and not info.get('isLoopbackDevice', False):
                    devices.append({'index': i, 'name': f"[Mikrofon] {info['name']}", 'channels': info['maxInputChannels'], 'is_loopback': False})
            except Exception:
                pass
        pa.terminate()
        return devices


class VUMeterWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.left_level = 0.0
        self.right_level = 0.0
        self.peak_left = 0.0
        self.peak_right = 0.0
        self.min_db = -60.0
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
        # Per-key tarihçe ve varsayılan parametreler
        for k in list(self._tempo.keys()):
            self._tempo[k]['hist'] = []
            self._tempo[k]['hist_maxlen'] = 120
        self._tempo_params = {kk: {'alpha': 0.20, 'delta': 0.08, 'min_interval': 0.25, 'hold': 0.12, 'auto': False, 'k': 0.6}
                              for kk in self._tempo.keys()}
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
        self.left_bpm_label = QLabel("-- BPM")
        self.left_bpm_label.setFixedWidth(70)
        left_layout.addWidget(self.l_light)
        left_layout.addWidget(left_label)
        left_layout.addWidget(self.left_bar)
        left_layout.addWidget(self.left_db_label)
        left_layout.addWidget(self.left_bpm_label)

        right_layout = QHBoxLayout()
        self.r_light = self._make_light()
        right_label = QLabel("Sağ:")
        right_label.setFixedWidth(40)
        self.right_bar = QProgressBar()
        self.right_bar.setMaximum(100)
        self.right_bar.setTextVisible(True)
        self.right_bar.setStyleSheet(self.left_bar.styleSheet())
        self.right_db_label = QLabel("0.0 dB")
        self.right_db_label.setFixedWidth(70)
        self.right_bpm_label = QLabel("-- BPM")
        self.right_bpm_label.setFixedWidth(70)
        right_layout.addWidget(self.r_light)
        right_layout.addWidget(right_label)
        right_layout.addWidget(self.right_bar)
        right_layout.addWidget(self.right_db_label)
        right_layout.addWidget(self.right_bpm_label)

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
        self.setLayout(layout)

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
        # Tempo ışıkları (Sol/Sağ)
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
        vals = [Llow, Lmid, Lhigh, Rlow, Rmid, Rhigh]
        dbs = [self._linear_to_db(v) for v in vals]
        perc = [_db_to_percent(d) for d in dbs]

        self.l_low_bar.setValue(perc[0]); self.l_low_db.setText(f"{dbs[0]:.1f} dB")
        self.l_mid_bar.setValue(perc[1]); self.l_mid_db.setText(f"{dbs[1]:.1f} dB")
        self.l_high_bar.setValue(perc[2]); self.l_high_db.setText(f"{dbs[2]:.1f} dB")
        self.r_low_bar.setValue(perc[3]); self.r_low_db.setText(f"{dbs[3]:.1f} dB")
        self.r_mid_bar.setValue(perc[4]); self.r_mid_db.setText(f"{dbs[4]:.1f} dB")
        self.r_high_bar.setValue(perc[5]); self.r_high_db.setText(f"{dbs[5]:.1f} dB")
        # Tempo ışıkları (bantlar)
        keys = ['Llow','Lmid','Lhigh','Rlow','Rmid','Rhigh']
        lights = [self.l_low_light, self.l_mid_light, self.l_high_light,
                  self.r_low_light, self.r_mid_light, self.r_high_light]
        for k, v, w in zip(keys, vals, lights):
            self._update_tempo(k, v)
            self._apply_light(k, w)
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
            # BPM tahmini için zaman damgası kaydı (son 10 sn)
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
        # Renk seçimi: genel L/R yeşil; low=mavi, mid=sarı, high=kırmızı
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
        self._init_ui()

        self.audio_monitor.level_updated.connect(self.on_level_updated)
        self.audio_monitor.error_occurred.connect(self.on_error)
        self.audio_monitor.bands_updated.connect(self.on_bands_updated)

        self.gui_timer = QTimer(self)
        self.gui_timer.setInterval(8)  # ~120 Hz
        self.gui_timer.timeout.connect(self._on_gui_tick)
        self._last_bands = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _init_ui(self):
        self.setWindowTitle(f"Audio VU Meter v{__version__}")
        self.setGeometry(100, 100, 620, 260)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()

        title = QLabel("Gerçek Zamanlı VU Meter - Mikrofon & Sistem Sesi")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        device_layout = QHBoxLayout()
        device_label = QLabel("Ses Kartı:")
        self.device_combo = QComboBox()
        self.refresh_devices()
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)

        range_label = QLabel("Aralık:")
        self.range_combo = QComboBox()
        for val in (-90, -60, -48, -40, -30):
            self.range_combo.addItem(f"{val} dB", float(val))
        self.range_combo.setCurrentIndex(1)
        self.range_combo.currentIndexChanged.connect(self.on_range_changed)
        device_layout.addWidget(range_label)
        device_layout.addWidget(self.range_combo)

        fps_label = QLabel("FPS:")
        self.fps_combo = QComboBox()
        for fps in (30, 60, 120, 240):
            self.fps_combo.addItem(f"{fps}", int(fps))
        self.fps_combo.setCurrentIndex(2)  # 120 FPS varsayılan
        self.fps_combo.currentIndexChanged.connect(self.on_fps_changed)
        device_layout.addWidget(fps_label)
        device_layout.addWidget(self.fps_combo)

        # Tempo hedefi ve otomatik eşik
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

        # Tempo parametreleri (Eşik, Hold, Min, Alfa)
        thr_label = QLabel("Eşik:")
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

        # Değer değişince uygula
        self.tempo_thr.valueChanged.connect(self.on_tempo_params_changed)
        self.tempo_hold.valueChanged.connect(self.on_tempo_params_changed)
        self.tempo_min.valueChanged.connect(self.on_tempo_params_changed)
        self.tempo_alpha.valueChanged.connect(self.on_tempo_params_changed)

        device_layout.addStretch()
        layout.addLayout(device_layout)

        self.vu_widget = VUMeterWidget()
        layout.addWidget(self.vu_widget)
        # Varsayılan tempo parametrelerini uygula
        try:
            self.on_tempo_params_changed()
        except Exception:
            pass

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Başlat")
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

        self.status_label = QLabel("Hazır")
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

    def on_range_changed(self):
        data = self.range_combo.currentData()
        if data is not None:
            self.vu_widget.set_min_db(float(data))

    def on_fps_changed(self):
        data = self.fps_combo.currentData()
        try:
            fps = int(data) if data is not None else 120
        except Exception:
            fps = 120
        if fps < 1:
            fps = 1
        if fps > 1000:
            fps = 1000
        interval_ms = max(1, int(round(1000.0 / float(fps))))
        self.gui_timer.setInterval(interval_ms)

    def on_bands_updated(self, bands_tuple):
        try:
            if isinstance(bands_tuple, (list, tuple)) and len(bands_tuple) == 6:
                self._last_bands = tuple(float(x) for x in bands_tuple)
        except Exception:
            self._last_bands = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def on_tempo_params_changed(self):
        try:
            target = self.tempo_target.currentData()
            alpha = float(self.tempo_alpha.value())
            delta = float(self.tempo_thr.value())
            hold_s = float(self.tempo_hold.value()) / 1000.0
            min_s = float(self.tempo_min.value()) / 1000.0
            auto = bool(self.tempo_auto.isChecked())
        except Exception:
            return
        try:
            if target is None:
                return
            self.vu_widget.set_tempo_params_for(str(target), alpha=alpha, delta=delta,
                                                min_interval_s=min_s, hold_s=hold_s,
                                                auto=auto)
        except Exception:
            pass

    def on_tempo_target_changed(self):
        try:
            target = self.tempo_target.currentData()
            if target is None:
                return
            params = self.vu_widget.get_tempo_params_for(str(target))
            self.tempo_alpha.setValue(float(params.get('alpha', 0.2)))
            self.tempo_thr.setValue(float(params.get('delta', 0.08)))
            self.tempo_min.setValue(float(params.get('min_interval', 0.25)) * 1000.0)
            self.tempo_hold.setValue(float(params.get('hold', 0.12)) * 1000.0)
            self.tempo_auto.setChecked(bool(params.get('auto', False)))
        except Exception:
            pass

    def on_error(self, error_msg):
        self.status_label.setText(f"Hata: {error_msg}")
        self.status_label.setStyleSheet("padding: 5px; background-color: #FFB6C1;")
        self.stop_monitoring()

    def closeEvent(self, event):
        self.audio_monitor.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VUMeterApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
