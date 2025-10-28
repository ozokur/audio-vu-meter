#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Output Device Debug Script
Sistem hoparlörleri sorununu tespit etmek için
"""

import pyaudio
import sys

def list_output_devices():
    """Tüm çıkış cihazlarını listele"""
    print("=== OUTPUT DEVICES DEBUG ===")
    print()
    
    try:
        pa = pyaudio.PyAudio()
        device_count = pa.get_device_count()
        print(f"Toplam cihaz sayısı: {device_count}")
        print()
        
        output_devices = []
        
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
                
                if output_channels > 0:
                    output_devices.append({
                        'index': i,
                        'name': name,
                        'channels': output_channels,
                        'sample_rate': sample_rate
                    })
                    print(f"  [OUTPUT DEVICE]")
                else:
                    print(f"  [NO OUTPUT]")
                print()
                
            except Exception as e:
                print(f"Device {i}: ERROR - {e}")
                print()
        
        print("=== OUTPUT DEVICES SUMMARY ===")
        for device in output_devices:
            print(f"Index {device['index']}: {device['name']} ({device['channels']} channels, {device['sample_rate']} Hz)")
        
        print()
        print("=== DEFAULT OUTPUT DEVICE ===")
        try:
            default_output = pa.get_default_output_device_info()
            print(f"Default Output: {default_output['name']} (Index: {default_output['index']})")
        except Exception as e:
            print(f"Default output device bulunamadı: {e}")
        
        pa.terminate()
        
    except Exception as e:
        print(f"PyAudio hatası: {e}")
        sys.exit(1)

if __name__ == "__main__":
    list_output_devices()

