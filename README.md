# 🎵 Audio VU Meter GUI

Ses kartından gerçek zamanlı ses seviyesi (VU - Volume Unit) verisi okuyup hızlı ve görsel bir şekilde gösteren Python uygulaması. Hem mikrofon girişini hem de **sistem ses çıkışını** (Edge browser, YouTube, müzik çalar vb.) izleyebilir!

## 🎯 Özellikler

- 🎤 **Gerçek Zamanlı Ses Girişi**: PyAudio ile ses kartından canlı veri okuma
- 🔊 **Sistem Ses Çıkışı Dinleme**: Windows WASAPI loopback ile browser, YouTube, Spotify gibi uygulamaların sesini izleyin
- 🌐 **Edge Browser Desteği**: Tarayıcınızda çalan müzik/video seslerini gerçek zamanlı görselleştirin
- 📊 **Hızlı VU Meter Gösterimi**: Düşük latency ile anlık ses seviyesi
- 🎨 **Modern GUI**: PyQt5 ile kullanıcı dostu arayüz
- 📈 **Görsel Göstergeler**: 
  - VU meter çubuğu
  - Peak level göstergesi
  - dB seviyesi
  - Stereo (Sol/Sağ) kanal ayrı gösterimi
- ⚡ **Yüksek Performans**: Threading ile optimize edilmiş
- 🎛️ **Ayarlanabilir**: Ses kartı seçimi, loopback/mikrofon seçimi

## 📋 Gereksinimler

- **Python**: 3.7 veya üstü
- **İşletim Sistemi**: Windows, macOS, Linux
- **Ses Kartı**: Mikrofon veya line-in girişi

## 🚀 Kurulum

### 1. Repository'yi klonlayın:
```bash
git clone https://github.com/ozokur/audio-vu-meter.git
cd audio-vu-meter
```

### 2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

### 3. Programı çalıştırın:
```bash
python vu_meter.py
```

## 📦 Bağımlılıklar

- `pyaudio` - Ses kartı I/O
- `PyQt5` - GUI framework
- `numpy` - Ses verisi işleme
- `matplotlib` (opsiyonel) - Gelişmiş görselleştirme

## 🎛️ Kullanım

### Temel Kullanım
```bash
python vu_meter.py
```

Uygulama açıldığında:
1. **Ses Kartı** açılır menüsünden kaynağı seçin:
   - 🔊 **Sistem Ses Çıkışı (Loopback)**: Edge browser, YouTube, Spotify vb. tüm sistem seslerini dinler
   - 🎤 **Mikrofon**: Fiziksel mikrofon girişini dinler
2. **▶ Başlat** butonuna basın
3. Edge'de YouTube/müzik açın veya mikrofona konuşun
4. VU meter gerçek zamanlı ses seviyesini gösterecektir!

### İpuçları
- 🌐 **Browser Sesi İçin**: "🔊 Sistem Ses Çıkışı (Loopback)" seçeneğini seçin
- 🎤 **Mikrofon İçin**: Listeden mikrofonunuzu seçin
- 📊 **dB Değerleri**: -60 dB sessizlik, 0 dB maksimum
- ⚡ **Peak Göstergesi**: En yüksek seviyeyi gösterir

## 🖼️ Ekran Görüntüleri

```
┌─────────────────────────────────────┐
│  🎵 Audio VU Meter                  │
├─────────────────────────────────────┤
│                                     │
│  Sol  ████████████░░░░░ -12 dB     │
│  Sağ  █████████████████ -6 dB      │
│                                     │
│  Peak: -3 dB                        │
│                                     │
│  [Start] [Stop] [Settings]          │
└─────────────────────────────────────┘
```

## 🔧 Özelleştirme

### Renk Temaları
Ayarlar menüsünden veya `config.json` dosyasından:
- Dark theme (varsayılan)
- Light theme
- Custom colors

### VU Meter Davranışı
```python
# config.json
{
  "sample_rate": 44100,
  "chunk_size": 1024,
  "channels": 2,
  "update_interval": 50,  # ms
  "peak_hold_time": 1000  # ms
}
```

## 📊 Teknik Detayler

### Ses İşleme
- Örnekleme hızı: 44.1 kHz (ayarlanabilir)
- Chunk boyutu: 1024 samples
- RMS (Root Mean Square) hesaplama
- Peak detection
- dB dönüşümü

### Performans
- Threading ile ayrı ses okuma thread'i
- Queue sistemi ile thread-safe veri aktarımı
- 60 FPS GUI güncelleme
- <50ms latency

## 🛠️ Geliştirme

### Proje Yapısı
```
audio-vu-meter/
├── vu_meter.py          # Ana uygulama
├── audio_input.py       # Ses kartı okuma modülü
├── vu_widget.py         # VU meter GUI widget
├── config.py            # Yapılandırma yönetimi
├── requirements.txt     # Python bağımlılıklar
├── README.md           # Dokümantasyon
├── LICENSE             # MIT License
└── examples/           # Örnek kodlar
    ├── simple_meter.py
    └── spectrum_analyzer.py
```

### Katkıda Bulunma
1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'feat: Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## ⚠️ Sorun Giderme

### Sistem Sesi Dinlenmiyor (Loopback)
Windows'ta loopback özelliği WASAPI ile çalışır. Eğer ses gelmiyor ise:
- Windows ses ayarlarında "Stereo Mix" veya "What U Hear" özelliğinin etkin olduğundan emin olun
- Ses çıkış cihazının aktif ve ses çalıyor olması gerekir
- Edge'de veya başka bir uygulamada ses çalın ve tekrar deneyin

### PyAudio Kurulum Hatası (Windows)
```bash
# Wheel dosyasını manuel indirin ve kurun
pip install pipwin
pipwin install pyaudio
```

### Ses Kartı Bulunamıyor
Uygulamayı başlattığınızda "Ses Kartı" açılır menüsünden cihazları görebilirsiniz.
- 🔊 işaretli olan sistem ses çıkışıdır (loopback)
- 🎤 işaretli olanlar mikrofon girişleridir

### Yüksek Latency
- Chunk size'ı azaltın (512 veya 256)
- Buffer sayısını azaltın
- Daha güçlü CPU kullanın

## 📝 TODO / Planlanan Özellikler

- [ ] Spektrum analyzer modu
- [ ] Ses kaydı özelliği
- [ ] FFT analizi
- [ ] Multi-channel desteği (>2 kanal)
- [ ] VST plugin desteği
- [ ] ASIO driver desteği (Windows)
- [ ] Skin sistemi
- [ ] Preset yönetimi

## 📄 Lisans

MIT License - Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👨‍💻 Geliştirici

Developed with ❤️ for audio enthusiasts

## 🔗 Bağlantılar

- [PyAudio Dokümantasyonu](https://people.csail.mit.edu/hubert/pyaudio/)
- [PyQt5 Dokümantasyonu](https://www.riverbankcomputing.com/static/Docs/PyQt5/)
- [Ses İşleme Rehberi](https://realpython.com/python-scipy-fft/)

## ⭐ Destek

Projeyi beğendiyseniz yıldız vermeyi unutmayın! ⭐

## 📧 İletişim

Sorularınız için issue açabilir veya pull request gönderebilirsiniz.

---

**Not**: Bu proje eğitim ve hobi amaçlıdır. Profesyonel ses uygulamaları için daha spesifik çözümler gerekebilir.

