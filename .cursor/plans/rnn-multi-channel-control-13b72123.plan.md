<!-- 13b72123-b215-464c-a444-51480dfbc8bb 57cf595a-1cca-470d-b73d-7db8d6e9b0d0 -->
# Fix RNN Static Control Issue

## Problem

RNN aktif olduğunda sadece dimmer çalışıyor, pan/tilt/color sabit kalıyor. Muhtemelen model eğitilmemiş veya yeterli sequence yok, bu yüzden heuristic fallback (pan=128, tilt=128) kullanılıyor.

## Implementation Steps

### 1. Add Debug to RNN Prediction

**File: `rnn_dim_controller.py`**

- `predict_dmx_channels` metoduna debug print ekle
- Hangi yolu kullandığını göster (RNN vs fallback)
- 2 saniye throttle ile spam önle (son print zamanı kaydet)
```python
# Check if using RNN or fallback
if len(self.dataset.audio_sequences) < self.sequence_length:
    print(f"⚠️ RNN Fallback: Yetersiz veri ({len(self.dataset.audio_sequences)}/{self.sequence_length})")
    return self._heuristic_dmx_channels(audio_bands)
```


### 2. Add Model Status Check

**File: `rnn_dim_controller.py`**

- Model yüklü mü kontrol et
- Eğer model yoksa veya eğitilmemişse uyarı ver
```python
# Check if model is trained
if not hasattr(self, 'model') or self.model is None:
    print("⚠️ RNN Model eğitilmemiş - heuristic fallback")
```


### 3. Add User Warning in GUI

**File: `vu_meter_120fps.py`**

- RNN aktif edildiğinde model durumunu kontrol et
- Eğer model yoksa veya yeterli veri yoksa kullanıcıya uyarı göster
- Status label'da göster: "RNN: Model Eğitilmemiş"
```python
def on_rnn_enable_changed(self, state):
    if enabled:
        # Check if model is trained
        if not self.dmx_controller.rnn_controller.is_model_trained():
            self.rnn_status_label.setText("RNN: Model Eğitilmemiş")
            print("⚠️ RNN Model eğitilmemiş! Önce eğitim yapın.")
```


### 4. Add is_model_trained Method

**File: `rnn_dim_controller.py`**

- Model eğitilmiş mi kontrol eden metod ekle
- Model dosyası var mı ve yüklenmiş mi kontrol et
```python
def is_model_trained(self) -> bool:
    """Check if model is trained and loaded"""
    return (hasattr(self, 'model') and 
            self.model is not None and 
            os.path.exists(self.model_path))
```


### 5. Throttle Debug Messages

**File: `rnn_dim_controller.py`**

- Son debug print zamanını kaydet
- 2 saniyeden kısa sürede tekrar print yapma
```python
# Throttle debug messages (2 seconds)
current_time = time.time()
if not hasattr(self, '_last_debug_time'):
    self._last_debug_time = 0
if current_time - self._last_debug_time > 2.0:
    print(f"Debug message here")
    self._last_debug_time = current_time
```


## Testing

1. RNN'yi aktif et (model olmadan)
2. Terminal'de uyarı mesajını gör
3. GUI'de "RNN: Model Eğitilmemiş" görmeli
4. Otomatik eğitim yap
5. Model yüklendikten sonra RNN doğru çalışmalı

## Files to Modify

- `rnn_dim_controller.py` - Debug prints, model check, throttling
- `vu_meter_120fps.py` - GUI warning, model status check

### To-dos

- [ ] Add debug prints to predict_dmx_channels with 2-second throttle
- [ ] Add is_model_trained method to check model status
- [ ] Add user warning in GUI when RNN enabled without trained model
- [ ] Test warnings with untrained and trained model