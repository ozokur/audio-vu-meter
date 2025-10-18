# DMX Işık Kontrolü Rehberi

Audio VU Meter v1.6.0 ile gelen DMX UDMX ışık kontrolü özelliği, ses seviyelerinizi gerçek zamanlı olarak DMX ışıklarınıza yansıtmanızı sağlar.

## Özellikler

### Otomatik Renk Kontrolü (Channel 3)
Frekans bantlarına göre otomatik renk değişimi:

- **Düşük Frekanslar (Bass - Llow/Rlow)**: Kırmızı → Turuncu
- **Orta Frekanslar (Lmid/Rmid)**: Sarı → Yeşil  
- **Yüksek Frekanslar (Lhigh/Rhigh)**: Camgöbeği → Mavi
- **Sessiz**: Beyaz

### Parlaklık Kontrolü (Channel 5)
Tüm ses bantlarının ortalamasına göre otomatik parlaklık ayarı (0-255).

## Donanım Gereksinimleri

### UDMX Cihazı
Desteklenen USB-DMX interface'ler:
- ✅ **Anyma uDMX** (VID:16C0 PID:05DC)
- ✅ **DMXControl uDMX** (VID:03EB PID:8888)
- ✅ Uyumlu UDMX klonlar

### Driver Kurulumu

#### Windows
1. [Zadig](https://zadig.akeo.ie/) indirin
2. UDMX cihazını USB'ye takın
3. Zadig'i çalıştırın
4. Options → List All Devices
5. UDMX cihazınızı seçin
6. Driver olarak "libusb-win32" seçin
7. "Replace Driver" veya "Install Driver"

#### Linux
```bash
sudo apt-get install libusb-1.0-0-dev

# Udev kuralı ekleyin
sudo nano /etc/udev/rules.d/50-udmx.rules
```

İçeriği:
```
SUBSYSTEM=="usb", ATTR{idVendor}=="16c0", ATTR{idProduct}=="05dc", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="03eb", ATTR{idProduct}=="8888", MODE="0666"
```

Sonra:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

#### macOS
```bash
brew install libusb
```

## Kurulum

1. PyUSB kütüphanesini yükleyin:
```bash
pip install pyusb>=1.2.1
```

2. Uygulamayı çalıştırın:
```bash
python vu_meter_120fps.py
```

## Kullanım

### 1. DMX Bağlantısı
1. UDMX cihazınızı USB'ye takın
2. DMX ışığınızı UDMX'e bağlayın
3. Uygulamada "🎭 DMX Kontrol" panelinde "Yenile" butonuna tıklayın
4. Cihazınız listede görünmelidir
5. "DMX Bağlan" butonuna tıklayın
6. Durum "DMX: Aktif ✓" olmalı

### 2. Işık Konfigürasyonu
DMX ışığınızda şu kanalları ayarlayın:
- **Channel 1**: Pan (Horizontal) - GUI'den manuel kontrol
- **Channel 2**: Tilt (Vertical) - GUI'den manuel kontrol
- **Channel 3**: Renk kontrolü (otomatik)
- **Channel 5**: Dimmer/parlaklık (otomatik)
- **Channel 6**: Master dimmer (otomatik)

### 3. Pan/Tilt Pozisyon Ayarı (Manuel)
1. **Ch1 Pan** ve **Ch2 Tilt** kontrollerini kullanarak ışığın yönünü ayarlayın
2. "Merkez (127)" butonu ile varsayılan pozisyona dönün
3. Değerler: 0-255 (127 = merkez)

### 4. Range Lights Aktifleştirme
1. **Range Target** seçin (Llow, Lmid veya Lhigh)
2. **Min dB** ve **Max dB** değerlerini ayarlayın
3. **"Range Lights Enable"** checkbox'ını aktif edin ⚠️ **Bu şart!**

### 5. Ses Kaynağı Seçimi
1. "Ses Kartı" menüsünden kaynak seçin:
   - **[Sistem]**: YouTube, Spotify, müzik çalarlar
   - **[Mikrofon]**: Canlı mikrofon girişi
2. "Başlat" butonuna basın
3. Işıklar otomatik olarak sese tepki vermeye başlar!

## Renk Mapping Tablosu

| Ses Durumu | Dominant Frekans | Renk | DMX Değeri |
|-----------|------------------|------|------------|
| Sessiz | - | Beyaz | 5 |
| Güçlü Bass | Düşük (20-250 Hz) | Kırmızı | 15 |
| Orta Bass | Düşük (20-250 Hz) | Turuncu | 25 |
| Güçlü Mid | Orta (250-4000 Hz) | Sarı | 35 |
| Orta Mid | Orta (250-4000 Hz) | Yeşil | 50 |
| Güçlü Treble | Yüksek (4-20 kHz) | Mavi | 90 |
| Orta Treble | Yüksek (4-20 kHz) | Camgöbeği | 70 |

## Sorun Giderme

### "pyusb yüklü değil"
```bash
pip install pyusb
```

### "UDMX cihazı bulunamadı"
- USB kablosunu kontrol edin
- Farklı USB portu deneyin (hub kullanmayın)
- Driver kurulumunu kontrol edin
- `python -c "import usb.core; print(usb.core.find(idVendor=0x16C0, idProduct=0x05DC))"` komutu ile test edin

### "DMX: Bağlantı Hatası"
- **Windows**: Zadig ile driver kurulumunu yapın
- **Windows**: Yönetici olarak çalıştırın
- **Linux**: udev kurallarını kontrol edin
- **Linux**: Kullanıcınızı `plugdev` grubuna ekleyin:
  ```bash
  sudo usermod -a -G plugdev $USER
  # Çıkış yapıp tekrar girin
  ```

### Işıklar Tepki Vermiyor
1. DMX adresini kontrol edin (başlangıç: Kanal 1)
2. DMX kablosunu kontrol edin
3. Channel 6 (master dimmer) açık mı?
4. Ses seviyesi yeterli mi?
5. Logları kontrol edin: `audio_vu_meter.log`

## Gelişmiş Ayarlar

### Renk Sensitivity
DMXController sınıfındaki `color_map` değerlerini değiştirerek farklı renkler kullanabilirsiniz:

```python
self.color_map = {
    'white': 5,
    'red': 15,
    'orange': 25,
    'yellow': 35,
    'green': 50,
    'cyan': 70,
    'blue': 90,
    'purple': 110,
    'magenta': 120,
}
```

### Parlaklık Eşiği
`set_audio_reactive()` fonksiyonunda `avg_level` hesaplamasını değiştirerek hassasiyeti ayarlayabilirsiniz.

## DMX Kanal Detayları

| Kanal | Fonksiyon | Kontrol | Açıklama |
|-------|-----------|---------|----------|
| Ch1 | Pan (Horizontal) | Manuel | GUI'den 0-255 arası ayarlanabilir |
| Ch2 | Tilt (Vertical) | Manuel | GUI'den 0-255 arası ayarlanabilir |
| Ch3 | Renk | Otomatik | Range light durumuna göre otomatik |
| Ch5 | Dimmer | Otomatik | Aktif bantların ses seviyesine göre |
| Ch6 | Master Dimmer | Otomatik | Ses varsa 255, yoksa 0 |

## Teknik Detaylar

- **DMX Protokol**: DMX512
- **USB Protokol**: USB 2.0 Low Speed
- **Kontrol Transfer**: bmRequestType 0x40, bRequest 0x01
- **Update Rate**: ~120 Hz (ses sync ile)
- **Kanal Sayısı**: 512 (5 aktif kanal kullanılıyor: Ch1, Ch2, Ch3, Ch5, Ch6)
- **Latency**: <10ms

## Güvenlik Uyarıları

⚠️ **Önemli:**
- Güçlü ışıklara doğrudan bakmayın
- Elektrik bağlantılarını kontrol edin
- Stroboskop efektleri epilepsi tetikleyebilir
- DMX cihazların güvenlik talimatlarına uyun

## Örnek Senaryolar

### 1. Müzik Dinleme
```
Kaynak: [Sistem] Speakers (Loopback)
DMX: Aktif
Sonuç: Müziğin bas, mid, treble değişimine göre renkli ışık şovu
```

### 2. Canlı Performans
```
Kaynak: [Mikrofon] USB Mikrofon
DMX: Aktif
Sonuç: Vokal ve enstrümanlara reaktif ışıklandırma
```

### 3. DJ Setup
```
Kaynak: [Sistem] DJ Software Output
FPS: 240 (daha hızlı tepki)
DMX: Aktif
Sonuç: Ultra-responsive beat-synced ışıklar
```

## İletişim ve Destek

Sorunlar için:
1. `audio_vu_meter.log` dosyasını kontrol edin
2. GitHub Issues'da rapor edin
3. [DMX Repository](https://github.com/ozokur/dmx) dokümanlarına bakın

## Lisans

MIT License - Audio VU Meter projesi ile aynı lisans altındadır.

---

**Not**: DMX özelliği opsiyoneldir. UDMX cihazı olmadan da uygulama normal VU meter olarak çalışır.

