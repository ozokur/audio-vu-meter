#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Device Debug Script
12 kanallı mikrofon girişi sorununu tespit etmek için
"""

import pyaudio
import sys

def list_audio_devices():
    """Tüm ses cihazlarını listele"""
    print("=== AUDIO DEVICES DEBUG ===")
    print()
    
    try:
        pa = pyaudio.PyAudio()
        device_count = pa.get_device_count()
        print(f"Toplam cihaz sayısı: {device_count}")
        print()
        
        input_devices = []
        
        for i in range(device_count):
            try:
                info = pa.get_device_info_by_index(i)
                name = info['name']
                input_channels = info['maxInputChannels']
                output_channels = info['maxOutputChannels']
                sample_rate = info['defaultSampleRate']
                host_api = info['hostApi']
                
                print(f"Device {i}:")
                print(f"  Name: {name}")
                print(f"  Input Channels: {input_channels}")
                print(f"  Output Channels: {output_channels}")
                print(f"  Sample Rate: {sample_rate}")
                print(f"  Host API: {host_api}")
                
                if input_channels > 0:
                    input_devices.append({
                        'index': i,
                        'name': name,
                        'channels': input_channels,
                        'sample_rate': sample_rate
                    })
                    print(f"  ✅ INPUT DEVICE")
                else:
                    print(f"  ❌ No input channels")
                print()
                
            except Exception as e:
                print(f"Device {i}: ERROR - {e}")
                print()
        
        print("=== INPUT DEVICES SUMMARY ===")
        for device in input_devices:
            print(f"Index {device['index']}: {device['name']} ({device['channels']} channels, {device['sample_rate']} Hz)")
        
        print()
        print("=== 12-CHANNEL DEVICES ===")
        multi_channel = [d for d in input_devices if d['channels'] >= 12]
        if multi_channel:
            for device in multi_channel:
                print(f"✅ {device['name']} - {device['channels']} channels")
        else:
            print("❌ 12+ kanallı cihaz bulunamadı")
        
        pa.terminate()
        
    except Exception as e:
        print(f"PyAudio hatası: {e}")
        sys.exit(1)

if __name__ == "__main__":
    list_audio_devices()

