# 📊 Nitel Araştırma Örneklem Büyüklüğü Tahmin Sistemi

Bu proje, nitel araştırmalarda veri doygunluğuna ulaşmak için gereken optimal örneklem büyüklüğünün makine öğrenmesi algoritmalarıyla tahmin edilmesini sağlayan web tabanlı bir karar destek sistemidir.

> **TÜBİTAK 3005** - Sosyal ve Beşeri Bilimlerde Yenilikçi Çözümler Araştırma Projeleri Destek Programı kapsamında geliştirilmiştir.

---

## 🎯 Projenin Amacı ve Özellikleri

- **AI Destekli Tahmin:** %74.8 doğruluk oranına sahip Stacking Ensemble Makine Öğrenmesi Modeli
- **5 Farklı Araştırma Deseni:**
  - Anlatı Araştırması (Narrative Research)
  - Etnografik Araştırma (Ethnographic Research)
  - Fenomenoloji (Phenomenology)
  - Gömülü Kuram (Grounded Theory)
  - Örnek Olay (Case Study)
- **11 Çok Boyutlu Parametre:** Araştırma kapsamı, araştırmacı yetkinliği, bilgi gücü, veri çeşitliliği ve kalitesi vb.
- **Hızlı ve Modern Arayüz:** Flask & Bootstrap 5 tabanlı responsive kullanıcı arayüzü

---

## 🛠️ Kurulum ve Yerel Çalıştırma

### 1. Gereksinimlerin Yüklenmesi
```bash
pip install -r requirements.txt
```

### 2. Uygulamanın Başlatılması
```bash
python app.py
```
Uygulama `http://localhost:8080` adresinde çalışacaktır.

---

## 🚀 Canlıya Dağıtım (Deployment)

Proje; **Render**, **Railway**, **Fly.io** veya herhangi bir Linux sanal sunucu (Ubuntu / Gunicorn / Nginx) üzerinde doğrudan çalıştırılmaya uygun şekilde yapılandırılmıştır.

---

## 📄 Lisans ve Kullanım
Bu sistem rehber niteliğindedir. Bilimsel araştırma metodolojisi gereği uzman görüşü ve veri doygunluğu kriterleri ile birlikte değerlendirilmelidir.
