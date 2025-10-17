#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basit DMX Test Aracı - Adım adım USB DMX test
"""

import sys
import time

print("=" * 60)
print("DMX Test Araci - Basit Versiyon")
print("=" * 60)
print()

# Adım 1: pyusb modülünü test et
print("[1] pyusb modulu kontrol ediliyor...")
try:
    import usb.core
    import usb.util
    print("    OK - pyusb modulu yuklendi")
except ImportError as e:
    print(f"    HATA - pyusb modulu bulunamadi: {e}")
    print("    Cozum: pip install pyusb")
    sys.exit(1)

print()

# Adım 2: USB cihazlarını listele
print("[2] USB cihazlar aranıyor...")
try:
    devices = list(usb.core.find(find_all=True))
    print(f"    Bulunan {len(devices)} USB cihaz:")
    
    if not devices:
        print("    UYARI - Hicbir USB cihaz bulunamadi!")
        print("    Zadig ile libusb-win32 surucusu kurmaniz gerekebilir")
    else:
        for dev in devices:
            try:
                vid = dev.idVendor
                pid = dev.idProduct
                print(f"      - VID:PID = {vid:04x}:{pid:04x}")
            except:
                pass
                
except Exception as e:
    print(f"    HATA - USB listelenemedi: {e}")

print()

# Adım 3: uDMX cihazını bul
print("[3] uDMX cihazi araniyor (VID:PID = 16c0:05dc)...")
try:
    udmx_dev = usb.core.find(idVendor=0x16c0, idProduct=0x05dc)
    
    if udmx_dev is None:
        print("    UYARI - uDMX cihazi bulunamadi!")
        print()
        print("    Kontrol listesi:")
        print("    1. USB DMX cihazi takili mi?")
        print("    2. Windows Aygit Yoneticisi'nde gorunuyor mu?")
        print("    3. Zadig ile libusb-win32 surucusu kuruldu mu?")
        print()
        
        # Diğer DMX cihazlarını ara
        print("    Diger DMX cihazlari aranıyor...")
        ftdi_dev = usb.core.find(idVendor=0x0403, idProduct=0x6001)
        if ftdi_dev:
            print("    BULUNDU - FTDI DMX cihazi (Enttec/OpenDMX)")
            udmx_dev = ftdi_dev
        else:
            print("    HATA - Hicbir DMX cihazi bulunamadi")
            sys.exit(1)
    else:
        print("    OK - uDMX cihazi bulundu!")
        
except Exception as e:
    print(f"    HATA - {e}")
    sys.exit(1)

print()

# Adım 4: Cihaza bağlan
print("[4] DMX cihazina baglaniliyor...")
try:
    # Kernel driver'ı ayır (Linux için)
    try:
        if udmx_dev.is_kernel_driver_active(0):
            udmx_dev.detach_kernel_driver(0)
    except:
        pass
    
    # Konfigürasyon ayarla
    try:
        udmx_dev.set_configuration()
        print("    OK - Konfigürasyon ayarlandi")
    except Exception as e:
        print(f"    UYARI - Konfigürasyon ayarlanamadi: {e}")
        print("    Devam ediliyor...")
        
except Exception as e:
    print(f"    HATA - Baglanti kurulamadi: {e}")
    sys.exit(1)

print()

# Adım 5: Test animasyonu
print("[5] Test animasyonu (5 saniye)...")
print("    DMX kanallari 1-10 parlak/sondurulecek")
print("    Isiklar yaniyorsa her sey dogru calisiyordur!")
print()

try:
    for cycle in range(5):
        # 0-255 arası değer
        value = int((cycle % 2) * 255)
        
        print(f"    Cycle {cycle + 1}/5: Kanal 1-10 = {value}")
        
        # uDMX'e 10 kanal gönder (Kanal 0-9)
        try:
            # Request 2: Set multiple channels
            # wValue: kanal sayısı
            # wIndex: başlangıç kanalı (0-based)
            # data: kanal değerleri
            data = [value] * 10
            
            udmx_dev.ctrl_transfer(
                bmRequestType=0x40,  # Vendor, Out
                bRequest=2,           # Set channels
                wValue=len(data),     # Kanal sayısı
                wIndex=0,             # Başlangıç kanalı
                data_or_wLength=bytes(data)
            )
            
        except Exception as e:
            print(f"    UYARI - Veri gonderilemedi: {e}")
        
        time.sleep(1)
    
    # Son olarak tümünü sıfırla
    print("    Isiklar kapatiliyor...")
    try:
        data = [0] * 10
        udmx_dev.ctrl_transfer(
            bmRequestType=0x40,
            bRequest=2,
            wValue=len(data),
            wIndex=0,
            data_or_wLength=bytes(data)
        )
    except:
        pass
    
    print()
    print("=" * 60)
    print("TEST TAMAMLANDI!")
    print("=" * 60)
    print()
    print("Sonuc:")
    print("  - USB DMX cihazi bulundu ve baglandi")
    print("  - Test animasyonu gonderildi")
    print()
    print("Eger isiklar yanmadiysa:")
    print("  1. DMX isik cihazinizin DMX adresi 1 olarak ayarli mi?")
    print("  2. DMX kablosu dogru mu bagli? (uDMX OUT -> Isik DMX IN)")
    print("  3. Isik cihazi DMX modunda mi?")
    
except KeyboardInterrupt:
    print("\n    Test kullanici tarafindan durduruldu")
except Exception as e:
    print(f"\n    HATA - Test sirasinda hata: {e}")
    import traceback
    traceback.print_exc()

