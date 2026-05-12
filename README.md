# Semantic Style Transfer with Person Preservation

Bu proje, bir **content image** ve bir **style image** kullanarak Neural Style Transfer işlemi yapar. Klasik stil aktarımından farklı olarak, görüntüdeki insan bölgesi tespit edilir ve mümkün olduğunca korunur. Böylece stil daha çok arka plana uygulanır; kişi, yüz ve vücut bölgesi daha doğal kalır.

Proje sade bir OOP yapısıyla hazırlanmıştır. Hem **Streamlit arayüzü** ile kullanılabilir hem de terminal üzerinden `main.py` dosyasıyla çalıştırılabilir.

---

## Projenin Amacı

Klasik Neural Style Transfer yöntemlerinde stil tüm görüntüye uygulanır. Bu durum özellikle insan fotoğraflarında yüz, kıyafet ve vücut detaylarının bozulmasına neden olabilir.

Bu projede amaç:

- Content görselindeki insan bölgesini tespit etmek,
- Style görselinden sanatsal doku ve renk bilgisini almak,
- Arka plana stil aktarımı uygulamak,
- İnsan bölgesini mümkün olduğunca korumak,
- Kullanıcının rahatlıkla kullanableceği bir arayüz ile sonucu üretmektir.


<img width="835" height="722" alt="image" src="https://github.com/user-attachments/assets/73aa313d-c997-4f1b-b823-31dea484d437" />

**Yurdumuz sevilen ozanı Neşet Ertaş saygı ve rahmetle anıyoruz**
---

## Kullanılan Yöntemler

Projede temel olarak şu yöntemler kullanılmıştır:

- **VGG19 tabanlı Neural Style Transfer**
- **Content Loss**
- **Style Loss**
- **Gram Matrix**
- **DeepLabV3 person segmentation**
- **Semantic mask blending**
- **Streamlit kullanıcı arayüzü**

---

## Proje Klasör Yapısı

```text
semantic-style-transfer-oop-simple/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── src/
    ├── __init__.py
    ├── image_utils.py
    ├── segmenter.py
    ├── nst_model.py
    └── semantic_style_transfer.py
```

---

## Dosyaların Görevleri

| Dosya | Açıklama |
|---|---|
| `app.py` | Streamlit arayüzünü çalıştırır. Kullanıcı content ve style görsellerini buradan yükler. |
| `main.py` | Terminal üzerinden çalıştırmak için kullanılan ana dosyadır. |
| `src/image_utils.py` | Görsel okuma, dönüştürme ve kaydetme işlemlerini içerir. |
| `src/segmenter.py` | DeepLabV3 modeli ile insan maskesi çıkarır. |
| `src/nst_model.py` | VGG19 tabanlı Neural Style Transfer modelini içerir. |
| `src/semantic_style_transfer.py` | İnsan korumalı semantic style transfer akışını yönetir. |
| `requirements.txt` | Proje için gerekli Python kütüphanelerini içerir. |
| `.gitignore` | GitHub’a yüklenmemesi gereken dosyaları belirtir. |

---

## Kurulum

Projeyi bilgisayarına klonla:

```bash
git clone REPO_LINKI
cd semantic-style-transfer
```

Gerekli kütüphaneleri yükle:

```bash
pip install -r requirements.txt

```

İlk çalıştırmada VGG19 ve DeepLabV3 model ağırlıkları otomatik indirilebilir. Bu yüzden ilk çalıştırma biraz uzun sürebilir.

---

## Arayüz ile Kullanım

Streamlit arayüzünü başlat:

```bash
streamlit run app.py
```
<img width="1782" height="794" alt="image" src="https://github.com/user-attachments/assets/2e492786-77a2-4f79-8af9-dbfefbcf68b8" />


Eğer `streamlit` komutu tanınmazsa şu komutu kullan:

```bash
python -m streamlit run app.py
```

Arayüz açıldıktan sonra:

1. **Content Image** yükle.
2. **Style Image** yükle.
3. Görsel boyutu, step sayısı ve ağırlıkları seç.
4. **Run Style Transfer** butonuna bas.
5. Oluşan sonucu ekranda görüntüle.

Arayüz kullanımında `inputs/` klasörüne fotoğraf koymana gerek yoktur. Fotoğraflar doğrudan arayüzden yüklenir.

<img width="800" height="450" alt="ezgif-2a612b8c10c2c535" src="https://github.com/user-attachments/assets/74d7d168-fda5-4937-a783-d62f9f1745b8" />


---

## Terminal ile Kullanım

Terminal üzerinden çalıştırmak için:

```bash
python main.py --content content.jpg --style style.jpg
```

Örnek kullanım:

```bash
python main.py \
  --content images/content.jpg \
  --style images/style.jpg \
  --image-size 384 \
  --steps 200 \
  --content-weight 5 \
  --style-weight 100000
```

---

## Yüklenebilir Görsel Formatları

Arayüzde aşağıdaki formatlar kullanılmalıdır:

```text
.jpg
.jpeg
.png
```

Önerilen formatlar:

```text
.jpg veya .png
```

---

## Yüklenmemesi Gereken Formatlar

Aşağıdaki formatlar önerilmez:

```text
.heic
.webp
.gif
.pdf
.tiff
.raw
.psd
.mp4
.mov
.avi
```

Bu formatlar bazı sistemlerde okunmayabilir veya model girişine uygun olmayabilir.

---

## Önerilen Görsel Boyutları

Model, görselleri çalıştırmadan önce yeniden boyutlandırır. Fakat daha hızlı ve stabil sonuç almak için çok büyük görseller yüklenmemesi önerilir.

| Kullanım Amacı | Önerilen Boyut |
|---|---|
| Hızlı test | `256 x 256` |
| Dengeli kullanım | `384 x 384` |
| Daha kaliteli sonuç | `512 x 512` |

En ideal kullanım:

```text
384 x 384 veya 512 x 512
```

Çok büyük görseller özellikle CPU üzerinde çalışırken işlemi yavaşlatabilir.

---

## Content Görseli İçin Öneriler

Content image, stil uygulanacak ana görseldir.

İyi sonuç için:

- Kişi net görünmelidir.
- Görsel çok karanlık olmamalıdır.
- İnsan ile arka plan birbirinden ayrılabilir olmalıdır.
- Aşırı bulanık fotoğraflar kullanılmamalıdır.
- Tek kişi içeren görseller daha iyi sonuç verebilir.

Önerilen content görseli:

```text
.jpg veya .png
384 x 384 / 512 x 512
net ve aydınlık fotoğraf
```

---

## Style Görseli İçin Öneriler

Style image, content görseline aktarılacak sanatsal dokuyu sağlar.

İyi sonuç için:

- Renk ve doku açısından belirgin bir stil içermelidir.
- Tablo, desen, sanat görseli veya renkli kompozisyon olabilir.
- Çok düz ve detaysız görseller zayıf sonuç verebilir.
- Çok karmaşık görsellerde sonuç fazla yoğun olabilir.

Önerilen style görseli:

```text
.jpg veya .png
renkli, dokulu ve belirgin stile sahip görsel
```

---

## Arayüzdeki Parametreler

| Parametre | Açıklama |
|---|---|
| `Image Size` | Görselin modele girmeden önce yeniden boyutlandırılacağı değerdir. |
| `Steps` | Optimizasyon adım sayısıdır. Artarsa sonuç iyileşebilir ama süre uzar. |
| `Content Weight` | Content görselinin korunma ağırlığıdır. |
| `Style Weight` | Style görselinin etkisini belirler. |
| `Blur Size` | İnsan maskesi ile arka plan geçişini yumuşatır. |

---

## Önerilen Parametreler

Başlangıç için önerilen değerler:

```text
Image Size: 384
Steps: 150 - 300
Content Weight: 5
Style Weight: 100000
Blur Size: 21
```

Daha hızlı deneme için:

```text
Image Size: 256
Steps: 80 - 120
```

Daha kaliteli sonuç için:

```text
Image Size: 512
Steps: 300+
```

---

## CPU ve GPU Kullanımı

Proje CPU üzerinde çalışabilir. Fakat Neural Style Transfer optimizasyon tabanlı olduğu için CPU üzerinde yavaş olabilir.

Daha hızlı çalışmak için GPU önerilir. CUDA destekli ekran kartı varsa PyTorch’un uygun CUDA sürümü kurulabilir. CPU kullanımı için normal `requirements.txt` kurulumu yeterlidir.

---

## requirements.txt İçeriği

Proje için gerekli temel kütüphaneler:

```text
streamlit
torch
torchvision
Pillow
numpy
matplotlib
scikit-image
opencv-python
```

Kurulum:

```bash
pip install -r requirements.txt
```

---

## .gitignore Nedir?

`.gitignore`, GitHub’a gönderilmemesi gereken dosya ve klasörleri belirtir.

Bu projede örnek olarak şunlar yok sayılabilir:

```gitignore
__pycache__/
*.pyc
outputs/
*.zip
.env
.DS_Store
```

Böylece gereksiz çıktı görselleri, Python cache dosyaları ve zip dosyaları repoya yüklenmez.

---

## Projenin Genel Akışı

```text
Content Image
     ↓
Person Segmentation
     ↓
Style Image
     ↓
Neural Style Transfer
     ↓
Semantic Mask Blending
     ↓
Final Output
```

---

## Geliştirme Fikirleri

Bu proje daha sonra şu özelliklerle geliştirilebilir:

- Farklı segmentation modellerinin eklenmesi
- Birden fazla kişi için daha başarılı maskeleme
- Arayüzden çıktı indirme butonu
- Farklı style preset seçenekleri
- Metrik hesaplama ekranı
- Batch processing desteği
- Daha hızlı NST için önceden eğitilmiş style transfer ağı

---

## Notlar

- İlk çalıştırmada PyTorch bazı model ağırlıklarını indirebilir.
- Büyük görseller işlem süresini artırır.
- En stabil kullanım için `.jpg` veya `.png` formatı önerilir.
- Arayüz kullanılıyorsa fotoğraflar doğrudan Streamlit ekranından yüklenir.
- Terminal kullanılıyorsa görsel dosya yolları komut satırında verilmelidir.

---

## Lisans

Bu proje eğitim ve araştırma amacıyla hazırlanmıştır.
