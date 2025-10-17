#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gelişmiş DMX Test - Sorun giderme ve alternatif yöntemler
"""

import sys
import time

print("=" * 70)
print("DMX GELİŞMİŞ TEST ARACI")
print("=" * 70)
print()

# Test 1: Import kontrolü
print("[TEST 1] Python modülleri kontrol ediliyor...")
modules_ok = True

try:
    import usb.core
    import usb.util
    print("  [OK] usb.core")
    print("  [OK] usb.util")
except ImportError as e:
    print(f"  [HATA] usb modulu: {e}")
    modules_ok = False

try:
    import usb.backend.libusb1
    print("  [OK] libusb1 backend")
except:
    print("  [UYARI] libusb1 backend yok (normal)")

try:
    import usb.backend.libusb0
    print("  [OK] libusb0 backend") 
except:
    print("  [UYARI] libusb0 backend yok")

if not modules_ok:
    print("\n  HATA: Gerekli modüller eksik!")
    sys.exit(1)

print()

# Test 2: Backend kontrol
print("[TEST 2] USB backend kontrol ediliyor...")
try:
    import usb.core
    backend = usb.core.find().backend if usb.core.find() else None
    if backend:
        print(f"  [OK] Aktif backend: {type(backend).__name__}")
    else:
        print("  [UYARI] Backend bulunamadi - Zadig ile surucu kurulumu gerekebilir")
except Exception as e:
    print(f"  [HATA] Backend hatasi: {e}")

print()

# Test 3: uDMX ara ve detaylı bilgi göster
print("[TEST 3] uDMX cihazı detaylı kontrol...")
try:
    import usb.core
    dev = usb.core.find(idVendor=0x16c0, idProduct=0x05dc)
    
    if dev is None:
        print("  [HATA] uDMX bulunamadi!")
        print("\n  Diger USB cihazlar:")
        for d in usb.core.find(find_all=True):
            try:
                print(f"    - {d.idVendor:04x}:{d.idProduct:04x}")
            except:
                pass
        sys.exit(1)
    
    print(f"  [OK] uDMX bulundu: {dev.idVendor:04x}:{dev.idProduct:04x}")
    
    # Detaylı bilgi
    try:
        print(f"    Bus: {dev.bus}")
        print(f"    Address: {dev.address}")
    except:
        pass
    
    # Kernel driver kontrolü
    try:
        if dev.is_kernel_driver_active(0):
            print("    Kernel driver: Aktif (ayrılacak)")
            dev.detach_kernel_driver(0)
        else:
            print("    Kernel driver: Yok")
    except:
        print("    Kernel driver: Kontrol edilemedi (Windows için normal)")
    
    # Konfigürasyon
    try:
        dev.set_configuration()
        print("    Konfigürasyon: Ayarlandı")
    except Exception as e:
        print(f"    Konfigürasyon: Hata - {e}")
        
except Exception as e:
    print(f"  ✗ Hata: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Basit control transfer
print("[TEST 4] Control transfer testi...")
try:
    # Tek kanal test (Request 1)
    print("  Method 1: Tek kanal (bRequest=1)...")
    try:
        # Set channel 0 to value 128
        ret = dev.ctrl_transfer(
            bmRequestType=0x40,
            bRequest=1,        # Single channel
            wValue=128,        # Value
            wIndex=0,          # Channel
            data_or_wLength=0
        )
        print(f"    [OK] Basarili (return: {ret})")
        time.sleep(0.5)
        
        # Set to 0
        dev.ctrl_transfer(0x40, 1, 0, 0, 0)
        
    except Exception as e:
        print(f"    [HATA] Hata: {e}")
    
    # Çoklu kanal test (Request 2)
    print("  Method 2: Coklu kanal (bRequest=2)...")
    try:
        data = bytes([100, 100, 100, 100, 100])
        ret = dev.ctrl_transfer(
            bmRequestType=0x40,
            bRequest=2,        # Multiple channels
            wValue=len(data),  # Number of channels
            wIndex=0,          # Start channel
            data_or_wLength=data
        )
        print(f"    [OK] Basarili (return: {ret})")
        time.sleep(0.5)
        
        # Clear
        dev.ctrl_transfer(0x40, 2, 5, 0, bytes([0]*5))
        
    except Exception as e:
        print(f"    [HATA] Hata: {e}")
        
except Exception as e:
    print(f"  [HATA] Test hatasi: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 5: Animasyon
print("[TEST 5] DMX animasyon testi (10 saniye)...")
print("  Kanal 1-3: Fade in/out")
print("  Izleyin: Isiklar yavasca yanip sonmeli")
print()

try:
    for i in range(100):
        # Sinüs dalgası gibi fade
        brightness = int(127.5 + 127.5 * ((i % 20) / 20.0 - 0.5) * 2)
        
        if i % 10 == 0:
            print(f"    Step {i}/100: Brightness = {brightness}")
        
        try:
            # Set channels 0, 1, 2
            data = bytes([brightness, brightness, brightness])
            dev.ctrl_transfer(0x40, 2, len(data), 0, data)
        except Exception as e:
            if i == 0:  # Sadece ilk hatayı göster
                print(f"    [UYARI] Gonderim hatasi: {e}")
            pass
        
        time.sleep(0.1)
    
    # Blackout
    print("    Blackout...")
    dev.ctrl_transfer(0x40, 2, 10, 0, bytes([0]*10))
    
    print()
    print("=" * 70)
    print("TEST TAMAMLANDI")
    print("=" * 70)
    print()
    print("SONUCLAR:")
    print("  [OK] uDMX cihazi bulundu ve baglandi")
    print("  [OK] DMX komutlari gonderildi")
    print()
    print("ISIKLAR YANMADI MI?")
    print("  1. DMX isik cihazinin adresi kanal 1'den basliyor mu?")
    print("  2. DMX kablosu: uDMX OUT -> Isik IN (dogru mu?)")
    print("  3. Isik DMX modunda mi? (manuel menuden kontrol edin)")
    print("  4. Baska DMX controller bagli degil mi?")
    print()
    
except KeyboardInterrupt:
    print("\n  Test kullanici tarafindan durduruldu")
    dev.ctrl_transfer(0x40, 2, 10, 0, bytes([0]*10))
except Exception as e:
    print(f"\n  [HATA] Test hatasi: {e}")
    import traceback
    traceback.print_exc()

