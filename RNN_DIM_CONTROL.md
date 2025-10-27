# RNN Dim Control - Akıllı DMX Dimmer Kontrolü

Bu özellik, Recurrent Neural Network (RNN) kullanarak ses verilerinden DMX dimmer değerlerini tahmin eden akıllı bir sistemdir.

## 🧠 Nasıl Çalışır?

### 1. Veri Toplama
- VU meter çalışırken ses bantları (Llow, Lmid, Lhigh, Rlow, Rmid, Rhigh) sürekli kaydedilir
- Mevcut dimmer değerleri de aynı anda kaydedilir
- Bu veriler RNN'in öğrenmesi için kullanılır

### 2. RNN Modeli
- **LSTM (Long Short-Term Memory)** tabanlı RNN
- 6 ses bandı → 1 dimmer değeri tahmini
- Sequence length: 50 (son 50 örnekleme)
- Hidden size: 64, Layers: 2

### 3. Eğitim Süreci
- Toplanan verilerle model eğitilir
- Epoch sayısı ayarlanabilir (10-1000)
- Loss fonksiyonu: MSE (Mean Squared Error)
- Optimizer: Adam

## 🎛️ Kullanım

### 1. Veri Toplama
1. VU meter'ı başlatın
2. DMX'i bağlayın ve çalıştırın
3. Müzik çalın - RNN otomatik olarak veri toplar
4. "Samples" sayısını takip edin

### 2. Model Eğitimi
1. Yeterli veri toplandıktan sonra (100+ sample önerilir)
2. "RNN Eğit" butonuna basın
3. Epoch sayısını ayarlayın (varsayılan: 100)
4. Eğitim tamamlanana kadar bekleyin

### 3. RNN Modunu Aktifleştirme
1. "RNN Aktif" checkbox'ını işaretleyin
2. Artık dimmer değerleri RNN tarafından tahmin edilir
3. RNN tahminleri gerçek zamanlı olarak uygulanır

## 📊 Özellikler

### RNN Model Mimarisi
```python
LSTM(input_size=6, hidden_size=64, num_layers=2, dropout=0.2)
    ↓
Linear(hidden_size, hidden_size//2)
    ↓
Dropout(0.2)
    ↓
Linear(hidden_size//2, 1)
    ↓
Sigmoid()  # 0-1 arası çıktı
```

### Veri Formatı
- **Input**: [Llow, Lmid, Lhigh, Rlow, Rmid, Rhigh] (0-1 normalized)
- **Output**: Dimmer değeri (0-255)
- **Sequence**: Son 50 örnekleme

### Fallback Sistemi
- RNN mevcut değilse → Heuristic yöntem
- RNN hata verirse → Heuristic yöntem
- Yeterli veri yoksa → Heuristic yöntem

## 🔧 Teknik Detaylar

### Dosyalar
- `rnn_dim_controller.py` - RNN model ve dataset sınıfları
- `vu_meter_120fps.py` - Ana uygulama (RNN entegrasyonu)
- `rnn_dim_model.pth` - Eğitilmiş model (otomatik oluşturulur)
- `rnn_training_data.json` - Toplanan veriler (otomatik oluşturulur)

### Bağımlılıklar
- `torch>=1.9.0` - PyTorch framework
- `numpy>=1.19.0` - Veri işleme
- `json` - Veri kaydetme

### Performans
- **Inference**: ~1ms (CPU)
- **Training**: 100 epochs ~10-30 saniye
- **Memory**: ~50MB (model + data)

## 🎯 Kullanım Senaryoları

### 1. Müzik Performansı
- DJ setleri için akıllı ışık kontrolü
- Müziğin ritmine göre otomatik dimmer ayarı
- Farklı müzik türleri için özelleştirilmiş kontrol

### 2. Canlı Performans
- Gerçek zamanlı ses analizi
- Anlık dimmer değeri tahmini
- Smooth ve doğal geçişler

### 3. Öğrenme ve Adaptasyon
- Kullanıcı tercihlerini öğrenir
- Farklı ortamlar için adapte olur
- Sürekli iyileşme

## ⚠️ Önemli Notlar

### Veri Kalitesi
- Çeşitli müzik türleri ile eğitin
- Farklı ses seviyelerinde test edin
- Yeterli veri topladığınızdan emin olun

### Model Performansı
- Loss değerini takip edin
- Overfitting'e dikkat edin
- Düzenli olarak yeniden eğitin

### Sistem Gereksinimleri
- PyTorch kurulu olmalı
- Yeterli RAM (4GB+ önerilir)
- CPU eğitimi için sabır gerekir

## 🚀 Gelecek Geliştirmeler

- [ ] GPU desteği (CUDA)
- [ ] Daha gelişmiş model mimarileri
- [ ] Real-time fine-tuning
- [ ] Çoklu model desteği
- [ ] Otomatik hyperparameter tuning

## 📝 Changelog

### v1.8.0 - RNN Dim Control
- ✅ LSTM tabanlı RNN modeli
- ✅ Otomatik veri toplama
- ✅ Real-time prediction
- ✅ GUI kontrolleri
- ✅ Fallback sistemi
- ✅ Model kaydetme/yükleme

---

**Not**: Bu özellik deneysel bir geliştirmedir. Üretim ortamında kullanmadan önce kapsamlı test yapın.
