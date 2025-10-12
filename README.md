# Audio VU Meter GUI

Ses kartÄ±ndan gerÃ§ek zamanlÄ± ses seviyesi (VU - Volume Unit) verisini okuyup hÄ±zlÄ± ve gÃ¶rsel ÅŸekilde gÃ¶steren bir Python uygulamasÄ±. Hem mikrofon giriÅŸini hem de sistem ses Ã§Ä±kÄ±ÅŸÄ±nÄ± (Edge, YouTube, mÃ¼zik Ã§alar vb.) izleyebilirsiniz.

## Ã–zellikler
- GerÃ§ek zamanlÄ± ses giriÅŸi (PyAudio)
- Sistem ses Ã§Ä±kÄ±ÅŸÄ± (WASAPI loopback) dinleme
- DÃ¼ÅŸÃ¼k gecikmeli, modern PyQt5 arayÃ¼z
- GÃ¶rsel gÃ¶stergeler: VU Ã§ubuÄŸu, peak, dB, stereo (Sol/SaÄŸ)
- Ayarlanabilir cihaz/loopback seÃ§imi

## Gereksinimler
- Python 3.7+
- Windows/macOS/Linux
- Mikrofon veya line-in giriÅŸi (loopback iÃ§in Windows Ã¶nerilir)

## Kurulum
1) Gerekli paketleri yÃ¼kleyin:

```
pip install -r requirements.txt
```

Windowsâ€™ta PyAudio kurulumu hata verirse:

```
pip install pipwin
pipwin install pyaudio
```

2) UygulamayÄ± Ã§alÄ±ÅŸtÄ±rÄ±n:

```
python vu_meter.py
```

## KullanÄ±m
1. â€œSes KartÄ±â€ menÃ¼sÃ¼nden kaynaÄŸÄ± seÃ§in:
   - Sistem Ses Ã‡Ä±kÄ±ÅŸÄ± (Loopback): Edge, YouTube, Spotify vb. tÃ¼m sistem sesleri
   - Mikrofon: Fiziksel mikrofon giriÅŸi
2. â€œBaÅŸlatâ€a basÄ±n
3. MÃ¼zik Ã§alÄ±n veya mikrofona konuÅŸun; VU meter anlÄ±k seviyeyi gÃ¶sterir

## Ä°puÃ§larÄ±
- TarayÄ±cÄ± sesi iÃ§in â€œSistem Ses Ã‡Ä±kÄ±ÅŸÄ± (Loopback)â€ seÃ§eneÄŸini kullanÄ±n
- dB referansÄ±: ~-60 dB sessiz, 0 dB maksimuma yakÄ±n
- Peak gÃ¶stergesi en yÃ¼ksek seviyeyi gÃ¶sterir

## Teknik Notlar
- Ã–rnekleme hÄ±zÄ±: 44.1 kHz (ayarlanabilir)
- Chunk boyutu: 1024 samples
- RMS ve peak hesaplama, dB dÃ¶nÃ¼ÅŸÃ¼mÃ¼
- AyrÄ± ses okuma threadâ€™i, ~60 FPS GUI gÃ¼ncelleme

## Sorun Giderme
### Sistem sesi gelmiyor (Loopback)
- Windowsâ€™ta WASAPI loopback gerekir; ses Ã§Ä±kÄ±ÅŸÄ± aktifken deneyin
- â€œStereo Mix/What U Hearâ€ tÃ¼rÃ¼ seÃ§enekler aÃ§Ä±k olmalÄ± (bazÄ± cihazlarda)

### PyAudio kurulumu (Windows)
YukarÄ±daki `pipwin` adÄ±mlarÄ±nÄ± uygulayÄ±n.

### Cihaz gÃ¶rÃ¼nmÃ¼yor
- MenÃ¼yÃ¼ â€œYenileâ€ ile gÃ¼ncelleyin
- [Sistem] etiketli olanlar loopback, [Mikrofon] etiketli olanlar giriÅŸ cihazlarÄ±dÄ±r

## SÃ¼rÃ¼m ve DeÄŸiÅŸiklikler
- SÃ¼rÃ¼m: 1.3.0 â€” bkz. `CHANGELOG.md`

## Lisans
MIT License â€” detaylar iÃ§in `LICENSE` dosyasÄ±na bakÄ±n.
# Audio VU Meter GUI

GerÃ§ek zamanlÄ± ses seviyesi (VU) Ã¶lÃ§Ã¼mÃ¼ ve gÃ¶rselleÅŸtirmesi yapan bir Python uygulamasÄ±. Hem mikrofon giriÅŸini hem de Windowsâ€™ta WASAPI loopback ile sistem ses Ã§Ä±kÄ±ÅŸÄ±nÄ± (Edge, YouTube, mÃ¼zik Ã§alar vb.) izleyebilirsiniz.

## Ã–ne Ã‡Ä±kanlar
- GerÃ§ek zamanlÄ± Ã¶lÃ§Ã¼m (PyAudio/pyaudiowpatch)
- Sistem ses Ã§Ä±kÄ±ÅŸÄ± (WASAPI loopback) dinleme
- Modern PyQt5 arayÃ¼z, dÃ¼ÅŸÃ¼k gecikme
- dB aralÄ±ÄŸÄ± seÃ§imi (Ã¶rn. âˆ’90/âˆ’60/âˆ’48/âˆ’40/âˆ’30 dB)
- Ayarlanabilir FPS (30/60/120/240)
- 6 bant (L/R Lowâ€“Midâ€“High) barlarÄ± ve perâ€‘bant dB
- Tempo Ä±ÅŸÄ±klarÄ±: her kanal/bantta seviye darbelerine gÃ¶re yanÄ±p sÃ¶nme
- BPM tahmini (L/R)
- DÃ¶ner dosya logâ€™u: `audio_vu_meter.log`

## HÄ±zlÄ± BaÅŸlangÄ±Ã§
1) BaÄŸÄ±mlÄ±lÄ±klarÄ± yÃ¼kleyin

```
pip install -r requirements.txt
```

Windowsâ€™ta PyAudio kurulumu hata verirse:

```
pip install pipwin
pipwin install pyaudio
```

2) Ã‡alÄ±ÅŸtÄ±rÄ±n (GeliÅŸmiÅŸ, 120 FPS + 6 bant + tempo/BPM):

```
python vu_meter_120fps.py
```

Alternatif (Temel sÃ¼rÃ¼m):

```
python vu_meter.py
```

## KullanÄ±m
- â€œSes KartÄ±â€ menÃ¼sÃ¼nden kaynak seÃ§in:
  - [Sistem] â€¦ (Loopback): TÃ¼m sistem seslerini dinler
  - [Mikrofon] â€¦: Fiziksel mikrofon giriÅŸini dinler
- AralÄ±k: VU Ã§ubuÄŸunun dB alt sÄ±nÄ±rÄ±nÄ± seÃ§er (0 dB tepe referansÄ±)
- FPS: ArayÃ¼z gÃ¼ncelleme hÄ±zÄ±nÄ± ayarlar (CPU/GPU yÃ¼kÃ¼yle dengeli seÃ§in)
- Tempo parametreleri (vu_meter_120fps.py):
  - EÅŸik (delta): Darbe tespiti hassasiyeti (0â€“1)
  - Hold ms: IÅŸÄ±ÄŸÄ±n aÃ§Ä±k kalma sÃ¼resi
  - Min ms: Darbeler arasÄ± minimum sÃ¼re
  - Alfa: Zarf (envelope) yumuÅŸatma katsayÄ±sÄ±

6 bant gÃ¶rÃ¼nÃ¼mÃ¼ (vu_meter_120fps.py):
- L/R iÃ§in Low (20â€“250 Hz), Mid (250â€“4000 Hz), High (4â€“20 kHz)
- Her bant iÃ§in dB deÄŸeri ve tempo Ä±ÅŸÄ±ÄŸÄ±
- L/R genel satÄ±rlarÄ±nda BPM tahmini etiketi

## Teknik Notlar
- Ã–rnekleme hÄ±zÄ±: 44.1 kHz (cihaz varsayÄ±lanÄ±na uyarlanÄ±r)
- Chunk boyutu: 256 (120 FPS akÄ±cÄ±lÄ±k iÃ§in)
- RMS ve peak hesaplama; VU Ã§ubuÄŸu dB â†’ yÃ¼zde haritalamasÄ± (min_dB..0 dB)
- FFT (RFFT + Hann) ile bant gÃ¼Ã§lerinden RMS tahmini

## Sorun Giderme
Sistem sesi gelmiyor (Loopback):
- Windowsâ€™ta WASAPI loopback gerekir; ses Ã§Ä±kÄ±ÅŸÄ± aktifken deneyin
- BazÄ± cihazlarda â€œStereo Mix/What U Hearâ€ benzeri seÃ§enekleri etkinleÅŸtirin

PyAudio kurulumu (Windows):
- `pipwin install pyaudio` kullanÄ±n (Ã¶ncesinde `pip install pipwin`)

Cihaz gÃ¶rÃ¼nmÃ¼yor:
- â€œYenileâ€ ile listeyi gÃ¼ncelleyin
- [Sistem] etiketi loopback, [Mikrofon] etiketi giriÅŸ cihazÄ±dÄ±r

Loglar:
- Hatalar ve Ã¶nemli bilgiler `audio_vu_meter.log` dosyasÄ±na yazÄ±lÄ±r

## Lisans
MIT License â€” ayrÄ±ntÄ±lar iÃ§in `LICENSE` dosyasÄ±na bakÄ±n.

