# Changelog

All notable changes to this project will be documented in this file.
This project uses Semantic Versioning (MAJOR.MINOR.PATCH) and a Keep a Changelog‑style format.

## [1.7.0] - 2025-10-19
### Added
- **🎵 Improved Low Frequency Detection (20-100Hz)**
  - ✅ Optimized frequency bands for better bass detection
  - ✅ Low band: 20-100 Hz (sub-bass, kick drum fundamentals)
  - ✅ Mid band: 100-4000 Hz (vocals, instruments)
  - ✅ High band: 4000-20000 Hz (treble)
  
### Changed
- **⚡ Enhanced FFT Resolution**
  - ✅ Chunk size: 256 → 2048 samples
  - ✅ Frequency resolution: 172 Hz → ~21.5 Hz
  - ✅ Better low-frequency bin coverage (~4 bins in 20-100Hz range)
  - ✅ More accurate bass detection for DMX dimmer control

### Improved
- **📊 UI Labels**
  - ✅ Frequency ranges now visible in band labels
  - ✅ Label width: 60 → 140 pixels
  - ✅ Shows "L Low (20-100Hz)", "L Mid (100-4k)", etc.

### Technical Details
- Sample rate: 44100 Hz
- Chunk size: 2048 samples
- FFT bins in low band: ~4 bins (20-100Hz)
- DMX Ch6 (Dimmer) now responds to 20-100Hz bass frequencies

## [1.6.10] - 2025-10-19
### Changed
- **Ch6 Range Scaling Geri Eklendi**: Sadece Range Scaling, Beat Flash YOK!
  - ✅ Ch6 → Llow Range Scaling (Min/Max dB ayarlarına göre)
  - ❌ Beat Flash efekti YOK (kaldırıldı)
  - ✅ Düzgün, smooth parlaklık değişimi
  - ✅ `set_audio_reactive()` tekrar aktif (beat_flash=False)
  - ✅ GUI'de Ch6 otomatik gösterim geri eklendi

### Kanal Özeti (v1.6.10)
```
Ch1: Pan      → MANUEL (SpinBox) 🎮
Ch2: Tilt     → MANUEL (SpinBox) 🎮
Ch3: Renk     → MANUEL (SpinBox + Presets) 🎨
Ch5: Master   → MANUEL (SpinBox) 🎛️
Ch6: Dimmer   → AUTO (Range Scaling ONLY) 📊
```

### Benefits
- 🎚️ Sadece Range Scaling - smooth ve doğal
- ❌ Beat flash yok - ani sıçramalar yok
- 🎵 Müziğe yumuşak tepki
- 🎯 Kontrollü otomatik parlaklık

## [1.6.9] - 2025-10-19
### Removed
- **Ch6 Otomatik Kontrol Kaldırıldı**: Ch6'daki tüm otomatik özellikler kaldırıldı (geçici)

## [1.6.8] - 2025-10-19
### Changed
- **Kanal swap: Ch5 ↔ Ch6**: Kullanıcı talebiyle kanal rolleri değiştirildi
  - **Ch5**: Artık MANUEL (Master Dimmer kontrolü, SpinBox ile 0-255)
  - **Ch6**: Artık OTOMATİK (Llow + Range scaling + Beat flash)
  - Eski Ch5 işlevleri → Ch6'ya taşındı
  - Eski Ch6 işlevleri → Ch5'e taşındı

### Kanal Özeti (v1.6.8)
```
Ch1: Pan      → MANUEL (SpinBox)
Ch2: Tilt     → MANUEL (SpinBox)
Ch3: Renk     → MANUEL (SpinBox + Preset butonlar)
Ch5: Master   → MANUEL (SpinBox) ✨ DEĞİŞTİ (eski Ch6)
Ch6: Dimmer   → OTOMATİK (Llow + Range + Beat) ✨ DEĞİŞTİ (eski Ch5)
```

### Technical Details
- `set_audio_reactive()`: Ch5 → Ch6 mapping değiştirildi
- GUI: Ch5 SpinBox, Ch6 read-only display
- `on_dmx_manual_changed()`: Ch6 → Ch5 kontrolü eklendi
- `_on_gui_tick()`: Ch5 → Ch6 otomatik gösterim
- DMX bilgi satırı güncellendi

### Benefits
- 🎛️ Ch5 manuel kontrol altında
- ⚡ Ch6 müziğe otomatik tepki veriyor
- 🎯 Kullanıcı tercihine göre özelleştirilebilir

## [1.6.7] - 2025-10-19
### Changed
- **Channel 6 (Master) artık MANUEL**: Otomatik kontrolden çıkarıldı
  - SpinBox ile manuel kontrol (0-255 arası)
  - Artık Llow seviyesine göre otomatik değişmiyor
  - Kullanıcı tam kontrol sahibi
  - `set_audio_reactive()` fonksiyonundan Ch6 set işlemi kaldırıldı
  - GUI'de Ch6 read-only display yerine manuel SpinBox eklendi
  - `on_dmx_manual_changed()` fonksiyonuna Ch6 kontrolü eklendi

### Kanal Özeti (v1.6.7)
```
Ch1: Pan      → MANUEL (SpinBox)
Ch2: Tilt     → MANUEL (SpinBox)
Ch3: Renk     → MANUEL (SpinBox + Preset butonlar)
Ch5: Dimmer   → OTOMATİK (Llow + Range scaling + Beat flash)
Ch6: Master   → MANUEL (SpinBox) ✨ DEĞİŞTİ
```

### Benefits
- 🎛️ Ch6 üzerinde tam kontrol
- 🎚️ Manuel master dimmer ayarı
- 🎯 Daha esnek setup
- ⚡ Ch5 hala otomatik, Ch6 sabit kalabilir

## [1.6.6] - 2025-10-18
### Changed
- **Channel 3 (Renk) artık MANUEL**: GUI'den SpinBox ve renk preset butonları ile kontrol
  - SpinBox: 0-255 arası manuel değer
  - Preset butonlar: ⚪ Beyaz (5), 🔴 Kırmızı (15), 🟠 Turuncu (25), 🟡 Sarı (35), 🟢 Yeşil (50), 🔵 Mavi (90)
  - Artık otomatik değişmiyor, kullanıcı seçiyor
- **Channel 6 artık Llow Mapping kullanıyor**: Ch5 ile aynı mantık (Range scaled + Beat flash)
  - Ch5: Dimmer (Llow + scaling)
  - Ch6: Master Dimmer (Llow + scaling) 
  - İki kanal da aynı parlaklık değerini alır
  - Range min/max dB ayarları her ikisini de etkiler

### Added
- Ch3 için renk preset butonları (hızlı renk değişimi)
- Ch3 SpinBox kontrolü (hassas ayar)

### Kanal Özeti (v1.6.6)
```
Ch1: Pan      → MANUEL (SpinBox)
Ch2: Tilt     → MANUEL (SpinBox)
Ch3: Renk     → MANUEL (SpinBox + Preset butonlar) ✨ YENİ
Ch5: Dimmer   → OTOMATİK (Llow + Range scaling + Beat flash)
Ch6: Master   → OTOMATİK (Llow + Range scaling + Beat flash) ✨ DEĞİŞTİ
```

### Benefits
- 🎨 Renk kontrolü kullanıcıda
- 🎚️ İki dimmer kanalı (bazı cihazlar için gerekli)
- ⚡ Beat flash her iki kanalı da etkiler
- 🎯 Daha esnek kontrol

## [1.6.5] - 2025-10-18
### Added
- **Range Scaling for DMX**: DMX parlaklık artık Range Target (Llow) min/max dB ayarlarına göre scale ediliyor!
  - Range Target → Llow seçildiğinde min/max dB değerleri kullanılır
  - Ses seviyesi dB'ye çevrilir ve range aralığına map edilir
  - Min dB altında: 0 (kapalı)
  - Max dB üstünde: 255 (maksimum)
  - Aradaki değerler: Linear interpolasyon
- Kullanıcı kontrollü hassasiyet ayarı
- Range light config entegrasyonu

### How Range Scaling Works
```
Örnek: Min dB = -40, Max dB = -10

Llow Seviyesi → dB Hesapla → Range Scale → DMX Parlaklık
   0.001      →    -60 dB   →    0.0     →      0
   0.01       →    -40 dB   →    0.0     →      0  (Min eşik)
   0.0316     →    -30 dB   →    0.33    →     84
   0.1        →    -20 dB   →    0.67    →    170
   0.316      →    -10 dB   →    1.0     →    255  (Max eşik)
   0.5        →    -6 dB    →    1.0     →    255
```

### Benefits
- 🎚️ **Kullanıcı Kontrolü**: Range min/max dB ile hassasiyeti ayarlayabilir
- 🎯 **Hassas Ayar**: Gürültülü ortamda veya sessiz müzikte optimize edilebilir
- 📊 **Görsel Feedback**: Range light ayarları DMX'i de etkiler
- 🔧 **Kolay Test**: Range ayarlarını değiştirerek hemen test edilebilir

## [1.6.4] - 2025-10-18
### Added
- **Beat Flash Efekti**: Tam beat vuruşunda kısa süre maksimum parlaklık!
  - Beat tespiti: Llow kanalındaki ani ses artışları
  - Flash süresi: 50ms (beat sonrası ilk 50 milisaniye)
  - Flash parlaklık: 255 (maksimum)
  - Flash renk: Kırmızı (beat intensity)
  - Normal zamanlarda: Llow seviyesine göre normal parlaklık
- Real-time beat detection entegrasyonu
- Strobe benzeri efekt ama müziğin ritmine senkronize

### Technical Details
- `beat_flash` parametresi `set_audio_reactive()` fonksiyonuna eklendi
- Beat tespiti: `_tempo['Llow']` state'inden `last_flash` ve `on_until` kontrolü
- 50ms pencere: `(now - last_flash) < 0.05`
- Beat anında: `brightness = 255`, `color = red`
- Normal zamanlarda: Base brightness ve renk korunur

### How It Works
```
Beat Tespit → last_flash set edilir
    ↓
İlk 50ms: beat_flash = True
    ↓
DMX: Brightness = 255 (MAX!)
    ↓
50ms sonra: beat_flash = False
    ↓
DMX: Normal seviyeye dönülür
```

## [1.6.3] - 2025-10-18
### Changed
- **DMX Basitleştirilmiş Mod**: Sadece Llow (Bass) kanalına göre otomatik parlaklık ve renk
  - Llow seviyesi → Parlaklık (0-255, linear mapping)
  - Llow > 0.7 → Kırmızı (güçlü bass)
  - Llow > 0.4 → Turuncu (orta bass)
  - Llow > 0.15 → Sarı (hafif bass)
  - Llow < 0.15 → Beyaz (minimal)
  - Eşik altında (< 10/255) → DMX kapanır
- **Range Light Sistemi Artık Opsiyonel**: DMX çalışması için "Range Lights Enable" gerekmez
- Daha direkt ve anlaşılır kontrol
- Real-time bass tepkisi

### Removed
- Range light bağımlılığı kaldırıldı (artık opsiyonel)
- Karmaşık multi-band renk mantığı basitleştirildi

## [1.6.2] - 2025-10-18
### Added
- **DMX Manuel Kontroller**: Channel 1 (Pan) ve Channel 2 (Tilt) için GUI kontrolleri
  - Ch1 Pan: Horizontal rotation (0-255) manuel kontrol
  - Ch2 Tilt: Vertical rotation (0-255) manuel kontrol
  - "Merkez (127)" butonu: Pan ve Tilt'i otomatik merkeze getirir
  - SpinBox ile hassas değer ayarlama
  - Bağlantı kurulduğunda otomatik değer gönderimi

### Technical Details
- Ch1 ve Ch2 otomatik ses kontrolüne tabi değil, sadece manuel
- Ch3 (Renk), Ch5 (Dimmer), Ch6 (Master) otomatik ses kontrollü
- Manuel değerler DMX frame'inde sürekli korunur

## [1.6.1] - 2025-10-18
### Changed
- **DMX Kontrol Mantığı Güncellendi**: DMX ışıkları artık sadece "Range Lights Enable" aktif olduğunda çalışır
  - Llow range light aktifken → Kırmızı/Turuncu
  - Lmid range light aktifken → Sarı/Yeşil
  - Lhigh range light aktifken → Mavi/Camgöbeği
  - Range lights kapalıysa DMX tamamen kapanır
- Parlaklık sadece aktif olan bantların seviyesine göre hesaplanır
- Renk önceliği: High > Mid > Low (birden fazla aktifse treble öncelikli)
- DMX bilgi satırı vurgulandı ve güncellendi

### Fixed
- DMX'in sürekli çalışması yerine sadece range light tetiklendiğinde aktif olması

## [1.6.0] - 2025-10-18
### Added
- **DMX UDMX Controller Entegrasyonu**: Ses reaktif DMX ışık kontrolü
  - UDMX USB cihazları için otomatik tanıma (Anyma uDMX, DMXControl uDMX)
  - Gerçek zamanlı ses bantlarına göre otomatik renk değişimi
  - Channel 5 (Dimmer): Genel ses seviyesine göre parlaklık kontrolü
  - Channel 3 (Renk): Frekans bantlarına göre otomatik renk seçimi
    - Düşük frekanslar (Llow/Rlow) → Kırmızı/Turuncu
    - Orta frekanslar (Lmid/Rmid) → Sarı/Yeşil
    - Yüksek frekanslar (Lhigh/Rhigh) → Mavi/Camgöbeği
  - DMX kontrol paneli GUI'ye eklendi
  - DMX cihaz yenileme ve bağlantı yönetimi
  - USB direkt bağlantı desteği (PyUSB)
- `pyusb>=1.2.1` bağımlılığı eklendi

### Technical Details
- DMXController sınıfı: 512 kanallı DMX evreni
- USB kontrol transferi (bmRequestType 0x40)
- Ses bantları ile senkronize DMX frame gönderimi
- Otomatik cihaz yönetimi ve kaynak temizleme
- Hata yönetimi ve loglama

### Notes
- DMX özelliği opsiyoneldir, pyusb yüklü değilse uygulama normal çalışır
- UDMX cihazı gerektir (USB-DMX interface)
- Windows için libusb driver kurulumu gerekebilir

## [1.5.0] - 2025-10-12
- Peak-hold control now in milliseconds with 1 ms steps (1–10000 ms, default 1000 ms) in `vu_meter_120fps.py`.
- Added per-channel byte display next to L/R bars (level bytes).
- Added separate 8-byte frame for light states (L, R, Llow, Lmid, Lhigh, Rlow, Rmid, Rhigh) and inline hex display.
- Minor UI polish on inline LED/Light byte sections.

## [1.3.0] - 2025-10-11
- Add per-band BPM labels (L/R low/mid/high) next to dB values.
- Add per-target tempo parameters (alpha, delta, min, hold) with optional auto-threshold.
- UI: target selector and auto toggle for tempo.

## [1.2.0] - 2025-10-11
- Per-target tempo params backend and UI wiring (alpha/delta/min/hold) with optional auto-threshold computation.

## [1.1.1] - 2025-10-11
- Fix: import `QCheckBox` for tempo auto toggle in advanced UI.

## [1.1.0] - 2025-10-11
- Add `vu_meter_120fps.py` advanced UI:
  - Adjustable dB range and FPS selector (30/60/120/240)
  - 6-band L/R meters (low/mid/high) with per-band dB display
  - Tempo lights per channel/band, L/R BPM estimate
  - Rotating file logger; NaN/Inf‑safe audio processing
  - README updated

## [1.0.0] - 2025-10-11
- Initial GUI with basic stereo VU, peak display, device selection.
## [1.4.0] - 2025-10-11
- Enable auto beat detection by default on all channels/bands (L, R, Llow/Lmid/Lhigh, Rlow/Rmid/Rhigh) with tuned parameters (alpha=0.15, k=0.8, min=0.20s, hold=0.10s).
