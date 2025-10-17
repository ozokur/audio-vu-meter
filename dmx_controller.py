#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMX Controller for Audio VU Meter
USB DMX (uDMX) üzerinden ses seviyelerine göre DMX ışık kontrolü
"""

import logging
import time
import threading
from typing import Optional, List

logger = logging.getLogger("dmx_controller")

# Serial port desteği
try:
    import serial
    import serial.tools.list_ports as list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    logger.warning("pyserial bulunamadı - Serial DMX desteği devre dışı")


class DMXController:
    """
    DMX ışık kontrolcüsü - VU Meter verilerine göre DMX sinyalleri gönderir
    uDMX ve diğer USB DMX cihazlarını destekler
    """
    
    def __init__(self, mode='auto', serial_port=None):
        """
        Args:
            mode: 'auto', 'usb', 'serial'
            serial_port: Serial port adı (mode='serial' için)
        """
        self.mode = mode
        self.serial_port = serial_port
        self.controller = None
        self.serial = None
        self.running = False
        self.dmx_data = [0] * 512  # DMX universe (512 kanal)
        self.lock = threading.Lock()
        
        # Kanal atamaları (özelleştirilebilir)
        self.channel_map = {
            'master': 1,      # Master dimmer
            'left': 2,        # Sol kanal
            'right': 3,       # Sağ kanal
            'low': 4,         # Bas
            'mid': 5,         # Orta
            'high': 6,        # Tiz
            'strobe': 7,      # Strobe/flash
            'color_r': 8,     # Kırmızı
            'color_g': 9,     # Yeşil
            'color_b': 10,    # Mavi
        }
        
    def connect(self) -> bool:
        """DMX cihazına bağlan"""
        try:
            if self.mode == 'auto':
                # Önce USB DMX dene
                if self._connect_usb():
                    logger.info("USB DMX bağlantısı başarılı")
                    return True
                # Sonra serial DMX dene
                if self._connect_serial():
                    logger.info("Serial DMX bağlantısı başarılı")
                    return True
                logger.error("Hiçbir DMX cihazı bulunamadı")
                return False
                
            elif self.mode == 'usb':
                return self._connect_usb()
                
            elif self.mode == 'serial':
                return self._connect_serial()
                
        except Exception as e:
            logger.exception(f"DMX bağlantı hatası: {e}")
            return False
            
    def _connect_usb(self) -> bool:
        """USB DMX bağlantısı (uDMX, OpenDMX vs.)"""
        try:
            import usb.core
            import usb.util
            
            # uDMX cihazını ara (VID:PID = 16c0:05dc)
            dev = usb.core.find(idVendor=0x16c0, idProduct=0x05dc)
            
            if dev is None:
                # Diğer yaygın DMX USB cihazları
                # FTDI (Enttec, OpenDMX)
                dev = usb.core.find(idVendor=0x0403, idProduct=0x6001)
            
            if dev is None:
                logger.info("USB DMX cihazı bulunamadı")
                return False
            
            # Kernel driver'ı ayır (Linux için)
            try:
                if dev.is_kernel_driver_active(0):
                    dev.detach_kernel_driver(0)
            except:
                pass
            
            # Konfigürasyon ayarla
            try:
                dev.set_configuration()
            except:
                pass
            
            self.controller = dev
            self.mode = 'usb'
            logger.info(f"USB DMX bağlandı: {hex(dev.idVendor)}:{hex(dev.idProduct)}")
            return True
            
        except Exception as e:
            logger.warning(f"USB DMX bağlanamadı: {e}")
            return False
            
    def _connect_serial(self) -> bool:
        """Serial DMX bağlantısı"""
        if not SERIAL_AVAILABLE:
            return False
        try:
            # Serial port otomatik tespit
            if not self.serial_port:
                ports = list(list_ports.comports())
                if not ports:
                    return False
                self.serial_port = ports[0].device
                
            self.serial = serial.Serial(
                self.serial_port,
                baudrate=250000,  # DMX512 standart baud rate
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_TWO,
                timeout=0
            )
            self.mode = 'serial'
            logger.info(f"Serial DMX açıldı: {self.serial_port}")
            return True
        except Exception as e:
            logger.warning(f"Serial DMX bağlanamadı: {e}")
            return False
            
    def disconnect(self):
        """DMX bağlantısını kes"""
        self.running = False
        try:
            if self.controller:
                # USB cihazı için cleanup
                try:
                    import usb.util
                    usb.util.dispose_resources(self.controller)
                except:
                    pass
                self.controller = None
        except:
            pass
        try:
            if self.serial and self.serial.is_open:
                self.serial.close()
                self.serial = None
        except:
            pass
            
    def set_channel(self, channel: int, value: int):
        """
        DMX kanalına değer yaz (1-512)
        Args:
            channel: Kanal numarası (1-512)
            value: DMX değeri (0-255)
        """
        if 1 <= channel <= 512:
            with self.lock:
                self.dmx_data[channel - 1] = max(0, min(255, int(value)))
                
    def set_channels(self, start_channel: int, values: List[int]):
        """
        Birden fazla kanala değer yaz
        Args:
            start_channel: Başlangıç kanalı (1-512)
            values: Değerler listesi
        """
        with self.lock:
            for i, val in enumerate(values):
                ch = start_channel + i
                if 1 <= ch <= 512:
                    self.dmx_data[ch - 1] = max(0, min(255, int(val)))
                    
    def update_from_vu(self, left: float, right: float, bands: List[float] = None):
        """
        VU Meter verilerinden DMX kanallarını güncelle
        Args:
            left: Sol kanal seviyesi (0.0-1.0)
            right: Sağ kanal seviyesi (0.0-1.0)
            bands: [Llow, Lmid, Lhigh, Rlow, Rmid, Rhigh] (0.0-1.0)
        """
        try:
            # Linear seviyeyi 0-255 DMX değerine çevir
            left_dmx = int(left * 255)
            right_dmx = int(right * 255)
            
            # Master seviye (maksimum)
            master = max(left_dmx, right_dmx)
            
            # Bantlar
            if bands and len(bands) >= 6:
                low_dmx = int(max(bands[0], bands[3]) * 255)   # L+R Low
                mid_dmx = int(max(bands[1], bands[4]) * 255)   # L+R Mid  
                high_dmx = int(max(bands[2], bands[5]) * 255)  # L+R High
            else:
                low_dmx = mid_dmx = high_dmx = 0
                
            # Kanalları ayarla
            self.set_channel(self.channel_map['master'], master)
            self.set_channel(self.channel_map['left'], left_dmx)
            self.set_channel(self.channel_map['right'], right_dmx)
            self.set_channel(self.channel_map['low'], low_dmx)
            self.set_channel(self.channel_map['mid'], mid_dmx)
            self.set_channel(self.channel_map['high'], high_dmx)
            
            # Renk değerleri (frekansa göre RGB)
            if bands and len(bands) >= 6:
                r = int((bands[2] + bands[5]) * 127.5)  # High -> Red
                g = int((bands[1] + bands[4]) * 127.5)  # Mid -> Green
                b = int((bands[0] + bands[3]) * 127.5)  # Low -> Blue
                self.set_channel(self.channel_map['color_r'], r)
                self.set_channel(self.channel_map['color_g'], g)
                self.set_channel(self.channel_map['color_b'], b)
                
        except Exception as e:
            logger.exception(f"VU güncelleme hatası: {e}")
            
    def send(self):
        """DMX verilerini cihaza gönder"""
        try:
            if self.mode == 'usb' and self.controller:
                # uDMX için özel protokol
                # Control transfer kullanarak veri gönder
                try:
                    import usb.core
                    # uDMX command: Set multiple channels (request 2)
                    # Request type: 0x40 (vendor, out)
                    
                    # Çoklu kanal gönderimi (32 kanallık paketler)
                    for i in range(0, 512, 32):
                        chunk = self.dmx_data[i:min(i+32, 512)]
                        if any(chunk):  # En az bir kanal 0'dan farklı ise
                            try:
                                self.controller.ctrl_transfer(
                                    bmRequestType=0x40,  # Vendor, Out
                                    bRequest=2,           # Set channels
                                    wValue=len(chunk),    # Kanal sayısı
                                    wIndex=i,             # Başlangıç kanalı
                                    data_or_wLength=bytes(chunk)
                                )
                            except:
                                pass
                except Exception as e:
                    logger.debug(f"USB DMX gönderim hatası: {e}")
                
            elif self.mode == 'serial' and self.serial and self.serial.is_open:
                # Serial DMX protokolü
                # Break signal (88μs minimum, ~200μs tipik)
                self.serial.break_condition = True
                time.sleep(0.0002)  # 200μs
                self.serial.break_condition = False
                
                # MAB (Mark After Break) - 8μs minimum, ~16μs tipik  
                time.sleep(0.000016)  # 16μs
                
                # Start code (0x00) + 512 kanal verisi
                packet = bytes([0x00] + self.dmx_data)
                self.serial.write(packet)
                
        except Exception as e:
            logger.exception(f"DMX gönderim hatası: {e}")
            
    def run_loop(self, fps: float = 40.0):
        """
        DMX gönderme döngüsü (ayrı thread'de çalışır)
        Args:
            fps: Saniyedeki güncelleme sayısı
        """
        self.running = True
        interval = 1.0 / fps
        
        while self.running:
            try:
                self.send()
                time.sleep(interval)
            except Exception as e:
                logger.exception(f"DMX loop hatası: {e}")
                time.sleep(0.1)
                
    def start_thread(self, fps: float = 40.0):
        """DMX gönderme thread'ini başlat"""
        thread = threading.Thread(target=self.run_loop, args=(fps,), daemon=True)
        thread.start()
        logger.info(f"DMX thread başlatıldı ({fps} FPS)")
        return thread
        
    def blackout(self):
        """Tüm kanalları sıfırla"""
        with self.lock:
            self.dmx_data = [0] * 512
        self.send()
        
    def is_connected(self) -> bool:
        """Bağlantı durumunu kontrol et"""
        if self.mode == 'usb':
            return self.controller is not None
        elif self.mode == 'serial':
            return self.serial is not None and self.serial.is_open
        return False


# Test kodu
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("DMX Controller Test")
    print("===================")
    
    dmx = DMXController(mode='auto')
    
    if dmx.connect():
        print(f"OK - DMX baglantisi basarili!")
        print(f"  Mode: {dmx.mode}")
        
        # Test: Parlak-söndür animasyonu
        print("\nTest animasyonu baslatiyor...")
        dmx.start_thread(fps=30)
        
        try:
            for i in range(50):
                level = (i % 20) / 20.0  # 0-1 arası
                dmx.update_from_vu(level, level, [level]*6)
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
            
        dmx.blackout()
        dmx.disconnect()
        print("Test tamamlandi")
        
    else:
        print("HATA - DMX cihazi bulunamadi!")
        print("\nKontrol edilecekler:")
        print("1. USB DMX cihazi takili mi?")
        print("2. Zadig ile libusb-win32 surucusu kuruldu mu?")

