#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN Eğitim Scripti
Ayrı bir process'te RNN eğitimi yapar
"""

import sys
import os
import time
import json
import argparse
from rnn_dim_controller import RNNDimController

def train_rnn(epochs=100, data_file="rnn_training_data.json", model_file="rnn_dim_model.pth"):
    """RNN modelini eğit"""
    # Suppress PyTorch warnings
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    print(f"RNN Eğitimi Başlatılıyor...")
    print(f"Epochs: {epochs}")
    print(f"Data File: {data_file}")
    print(f"Model File: {model_file}")
    print("-" * 50)
    
    try:
        # RNN Controller oluştur
        rnn_controller = RNNDimController(
            model_path=model_file,
            data_path=data_file,
            sequence_length=10
        )
        
        # Veri yükle
        if os.path.exists(data_file):
            rnn_controller.dataset.load_from_file(data_file)
            samples_count = len(rnn_controller.dataset.audio_sequences)
            print(f"Veri yüklendi: {samples_count} sample")
            
            # Yeterli veri var mı kontrol et (1200 sample gerekli)
            if samples_count < 1200:
                print(f"Yetersiz veri! {samples_count} sample var, 1200 gerekli.")
                print("Lütfen daha fazla sample toplayın.")
                return False
        else:
            print(f"Veri dosyası bulunamadı: {data_file}")
            return False
        
        # Eğitim başlat
        print(f"Eğitim başlıyor...")
        start_time = time.time()
        
        # Eğitim öncesi bilgi
        sequences = rnn_controller.dataset.get_sequences()
        print(f"Eğitim için {len(sequences[0])} sequence hazırlandı")
        
        success = rnn_controller.train_model_with_progress(
            epochs=epochs,
            progress_callback=lambda epoch, total, loss: print(f"Epoch {epoch}/{total}, Loss: {loss:.6f}")
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        if success:
            print(f"Egitim tamamlandi!")
            print(f"Sure: {training_time:.1f} saniye")
            print(f"Final Loss: {rnn_controller.training_losses[-1]:.6f}")
            
            # Eğitim tarihini kaydet
            rnn_controller.last_training_time = time.time()
            
            # Buffer sistemi - samples'ları temizleme kaldırıldı
            # rnn_controller.clear_training_data()
            
            return True
        else:
            print(f"Egitim basarisiz!")
            print(f"Detayli hata: Veri yetersiz veya model hatasi")
            return False
            
    except Exception as e:
        print(f"Hata: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='RNN Eğitim Scripti')
    parser.add_argument('--epochs', type=int, default=100, help='Eğitim epoch sayısı')
    parser.add_argument('--data', type=str, default='rnn_training_data.json', help='Veri dosyası')
    parser.add_argument('--model', type=str, default='rnn_dim_model.pth', help='Model dosyası')
    
    args = parser.parse_args()
    
    success = train_rnn(
        epochs=args.epochs,
        data_file=args.data,
        model_file=args.model
    )
    
    # Sonucu dosyaya yaz
    result = {
        'success': success,
        'timestamp': time.time(),
        'epochs': args.epochs
    }
    
    with open('training_result.json', 'w') as f:
        json.dump(result, f)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
