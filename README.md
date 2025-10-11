# Audio VU Meter GUI

Ses kartından gerçek zamanlı ses seviyesi (VU - Volume Unit) verisini okuyup hızlı ve görsel şekilde gösteren bir Python uygulaması. Hem mikrofon girişini hem de sistem ses çıkışını (Edge, YouTube, müzik çalar vb.) izleyebilirsiniz.

## Özellikler
- Gerçek zamanlı ses girişi (PyAudio)
- Sistem ses çıkışı (WASAPI loopback) dinleme
- Düşük gecikmeli, modern PyQt5 arayüz
- Görsel göstergeler: VU çubuğu, peak, dB, stereo (Sol/Sağ)
- Ayarlanabilir cihaz/loopback seçimi

## Gereksinimler
- Python 3.7+
- Windows/macOS/Linux
- Mikrofon veya line-in girişi (loopback için Windows önerilir)

## Kurulum
1) Gerekli paketleri yükleyin:

```
pip install -r requirements.txt
```

Windows’ta PyAudio kurulumu hata verirse:

```
pip install pipwin
pipwin install pyaudio
```

2) Uygulamayı çalıştırın:

```
python vu_meter.py
```

## Kullanım
1. “Ses Kartı” menüsünden kaynağı seçin:
   - Sistem Ses Çıkışı (Loopback): Edge, YouTube, Spotify vb. tüm sistem sesleri
   - Mikrofon: Fiziksel mikrofon girişi
2. “Başlat”a basın
3. Müzik çalın veya mikrofona konuşun; VU meter anlık seviyeyi gösterir

## İpuçları
- Tarayıcı sesi için “Sistem Ses Çıkışı (Loopback)” seçeneğini kullanın
- dB referansı: ~-60 dB sessiz, 0 dB maksimuma yakın
- Peak göstergesi en yüksek seviyeyi gösterir

## Teknik Notlar
- Örnekleme hızı: 44.1 kHz (ayarlanabilir)
- Chunk boyutu: 1024 samples
- RMS ve peak hesaplama, dB dönüşümü
- Ayrı ses okuma thread’i, ~60 FPS GUI güncelleme

## Sorun Giderme
### Sistem sesi gelmiyor (Loopback)
- Windows’ta WASAPI loopback gerekir; ses çıkışı aktifken deneyin
- “Stereo Mix/What U Hear” türü seçenekler açık olmalı (bazı cihazlarda)

### PyAudio kurulumu (Windows)
Yukarıdaki `pipwin` adımlarını uygulayın.

### Cihaz görünmüyor
- Menüyü “Yenile” ile güncelleyin
- [Sistem] etiketli olanlar loopback, [Mikrofon] etiketli olanlar giriş cihazlarıdır

## Sürüm ve Değişiklikler
- Sürüm: 1.3.0 — bkz. `CHANGELOG.md`

## Lisans
MIT License — detaylar için `LICENSE` dosyasına bakın.
# Audio VU Meter GUI

Gerçek zamanlı ses seviyesi (VU) ölçümü ve görselleştirmesi yapan bir Python uygulaması. Hem mikrofon girişini hem de Windows’ta WASAPI loopback ile sistem ses çıkışını (Edge, YouTube, müzik çalar vb.) izleyebilirsiniz.

## Öne Çıkanlar
- Gerçek zamanlı ölçüm (PyAudio/pyaudiowpatch)
- Sistem ses çıkışı (WASAPI loopback) dinleme
- Modern PyQt5 arayüz, düşük gecikme
- dB aralığı seçimi (örn. −90/−60/−48/−40/−30 dB)
- Ayarlanabilir FPS (30/60/120/240)
- 6 bant (L/R Low–Mid–High) barları ve per‑bant dB
- Tempo ışıkları: her kanal/bantta seviye darbelerine göre yanıp sönme
- BPM tahmini (L/R)
- Döner dosya log’u: `audio_vu_meter.log`

## Hızlı Başlangıç
1) Bağımlılıkları yükleyin

```
pip install -r requirements.txt
```

Windows’ta PyAudio kurulumu hata verirse:

```
pip install pipwin
pipwin install pyaudio
```

2) Çalıştırın (Gelişmiş, 120 FPS + 6 bant + tempo/BPM):

```
python vu_meter_120fps.py
```

Alternatif (Temel sürüm):

```
python vu_meter.py
```

## Kullanım
- “Ses Kartı” menüsünden kaynak seçin:
  - [Sistem] … (Loopback): Tüm sistem seslerini dinler
  - [Mikrofon] …: Fiziksel mikrofon girişini dinler
- Aralık: VU çubuğunun dB alt sınırını seçer (0 dB tepe referansı)
- FPS: Arayüz güncelleme hızını ayarlar (CPU/GPU yüküyle dengeli seçin)
- Tempo parametreleri (vu_meter_120fps.py):
  - Eşik (delta): Darbe tespiti hassasiyeti (0–1)
  - Hold ms: Işığın açık kalma süresi
  - Min ms: Darbeler arası minimum süre
  - Alfa: Zarf (envelope) yumuşatma katsayısı

6 bant görünümü (vu_meter_120fps.py):
- L/R için Low (20–250 Hz), Mid (250–4000 Hz), High (4–20 kHz)
- Her bant için dB değeri ve tempo ışığı
- L/R genel satırlarında BPM tahmini etiketi

## Teknik Notlar
- Örnekleme hızı: 44.1 kHz (cihaz varsayılanına uyarlanır)
- Chunk boyutu: 256 (120 FPS akıcılık için)
- RMS ve peak hesaplama; VU çubuğu dB → yüzde haritalaması (min_dB..0 dB)
- FFT (RFFT + Hann) ile bant güçlerinden RMS tahmini

## Sorun Giderme
Sistem sesi gelmiyor (Loopback):
- Windows’ta WASAPI loopback gerekir; ses çıkışı aktifken deneyin
- Bazı cihazlarda “Stereo Mix/What U Hear” benzeri seçenekleri etkinleştirin

PyAudio kurulumu (Windows):
- `pipwin install pyaudio` kullanın (öncesinde `pip install pipwin`)

Cihaz görünmüyor:
- “Yenile” ile listeyi güncelleyin
- [Sistem] etiketi loopback, [Mikrofon] etiketi giriş cihazıdır

Loglar:
- Hatalar ve önemli bilgiler `audio_vu_meter.log` dosyasına yazılır

## Lisans
MIT License — ayrıntılar için `LICENSE` dosyasına bakın.
