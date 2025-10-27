<!-- 13b72123-b215-464c-a444-51480dfbc8bb 4f4bc01f-981e-487d-8e11-1da37a6c5030 -->
# RNN Control Improvements

## Overview

Enhance RNN system with 4-channel control (Pan/Tilt/Color/Dimmer), improve Tilt movement patterns, and add UI feedback for auto-retrain.

## Implementation Steps

### 1. Expand RNN Model to 4 Channels

**File: `rnn_dim_controller.py`**

Update model architecture:

- Change `output_size` from 3 to 4 (Pan, Tilt, Color, Dimmer)
- Update `AudioSequenceDataset` to store 4 DMX values
- Modify `predict_dmx_channels` to return color value
- Add color to heuristic fallback

Key changes:

```python
class RNN_DimController(nn.Module):
    def __init__(self, ..., output_size=4):  # 3 -> 4
        
class AudioSequenceDataset:
    def add_sample(self, audio_bands, dmx_values):
        # dmx_values = [pan, tilt, color, dimmer]
        
class RNNDimController:
    def predict_dmx_channels(self, audio_bands):
        return {
            'pan': ...,
            'tilt': ...,
            'color': ...,  # NEW
            'dimmer': ...
        }
```

### 2. Improve Tilt Movement

**File: `vu_meter_120fps.py`**

Enhance `_calculate_heuristic_pan_tilt`:

- Increase Tilt range (currently 50-200, make it 20-255)
- Add more aggressive beat response
- Increase modulation from audio levels
- Add different patterns based on beat intensity
```python
def _calculate_heuristic_pan_tilt(self, llow, lmid, beat_state):
    # More aggressive Tilt movement
    if beat_state['lmid_beat']:
        step = int((time.time() % 3.0) / 1.0)
        tilt = [20, 128, 235][step]  # Wider range: 20-235 instead of 50-200
        
    # Stronger audio modulation for Tilt
    if lmid > 0.1:
        tilt = int(tilt + (lmid - 0.1) * 80)  # Increased from 30 to 80
```


### 3. Add Silent Auto-Retrain Indicator

**File: `vu_meter_120fps.py`**

Add visual feedback during auto-retrain:

- Show icon/emoji when auto-retrain is running
- Update RNN status label with training indicator
- Optional: Flash RNN panel background

In `update_rnn_stats`:

```python
def update_rnn_stats(self):
    is_training = getattr(self.dmx_controller.rnn_controller, 'is_training', False)
    
    if is_training:
        self.rnn_status_label.setText("RNN: 🔄 Auto-Training...")
        self.rnn_status_label.setStyleSheet("color:orange; font-weight:bold; animation:blink;")
    else:
        # Normal status
```

### 4. Update DMX Integration for Ch3

**File: `vu_meter_120fps.py`**

In `set_audio_reactive`:

- Add Ch3 to data collection
- Apply color predictions from RNN
- Add heuristic color based on frequency bands
```python
# Data collection
current_pan = self.dmx_data[0]
current_tilt = self.dmx_data[1]
current_color = self.dmx_data[2]  # NEW
current_dimmer = self.dmx_data[5]
self.rnn_controller.add_audio_sample(audio_bands, current_pan, current_tilt, current_color, current_dimmer)

# Apply predictions
if self.rnn_enabled:
    prediction = self.rnn_controller.predict_dmx_channels(audio_bands)
    self.set_channel(1, prediction['pan'])
    self.set_channel(2, prediction['tilt'])
    self.set_channel(3, prediction['color'])  # NEW
    self.set_channel(6, prediction['dimmer'])
```


### 5. Color Heuristic

**File: `vu_meter_120fps.py`**

Add color selection based on frequency:

```python
def _calculate_heuristic_color(self, llow, lmid, lhigh):
    # Map frequency bands to colors
    if lhigh > 0.5:  # High freq -> Blue/Cyan
        return 70  # Cyan
    elif lmid > 0.5:  # Mid freq -> Green/Yellow
        return 35  # Yellow
    elif llow > 0.5:  # Low freq -> Red/Orange
        return 15  # Red
    else:
        return 5  # White (default)
```

### 6. Update UI Text

**File: `vu_meter_120fps.py`**

Update RNN info label:

```python
rnn_info = QLabel("🧠 RNN: LSTM ile ses verilerinden Pan/Tilt/Color/Dimmer değerleri tahmin eder. Auto-retrain active!")
```

## Testing Plan

1. Test 4-channel RNN output
2. Verify Tilt movement is more visible
3. Check auto-retrain indicator appears
4. Test color predictions with different music
5. Verify Ch5 still manual

## Notes

- Ch5 remains manual (no changes)
- Backward compatibility: old 3-channel models will be converted
- Auto-retrain indicator helps user know system is learning
- Tilt range increased from 150 to 235 for visibility

### To-dos

- [ ] Update RNN model to 3-channel output (Pan/Tilt/Dimmer)
- [ ] Modify AudioSequenceDataset to store 3 DMX values
- [ ] Implement beat-based Pan/Tilt heuristic fallback
- [ ] Add auto-retrain logic every N samples
- [ ] Update DMXController to use 3-channel predictions
- [ ] Update RNN UI to show Pan/Tilt/Dim control status
- [ ] Test heuristic and RNN modes with real audio