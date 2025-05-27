# Şarkı Açıklamaları İle Otomatik Tür Tahmini
Bu projede şarkı açıklamalarının  metinsel içeriğine göre şarkıyı söyleyen sanatçıyı veya grubu tahmin etme işlemi üzerine işlemler gerçekleştirilecektir.
Özellikleri tam anlamıyla karşılayan bir veri seti bulunamadığı için şarkı açıklamalarının metinsel içeriğine göre sanatçıların ürün ortaya koyduğu belirli türler 
üzerinden ilerleme sağlanacaktır. Çeşitli yöntemler denendiysede sitelerin güvenlik politikaları nedeniyle verilere ulaşılamamıştır.
İnternet üzerinden ulaşabilecek tüm veriler analiz edilerek kaggle üzerinden Top 500 Songs.csv dosyası özellikler açısından daha uygun olduğu için işleme alınmıştır.
Mevcut veri, 500 satır ve 9 tane sütündan oluşmaktadır. 281 kb ve .csv formatında bulunmaktadır. 
Bu çalışma yapıldığı sırada örnek alınan veri seti kaggle kullanıcısı tarafından değiştirilmiştir. Aşağıda paylaşılan bağlantı üzerinden yeni verilere erişim sağlayabilirsiniz.
https://www.kaggle.com/code/mpwolke/500-greatest-songs/input

# Metin Benzerliği ve Şarkı Öneri Sistemi

Bu proje, popüler şarkıların açıklamalarını (description) kullanarak metin ön işleme, vektörleştirme (TF-IDF, Word2Vec) ve metin benzerliği analizlerini (Kosinüs Benzerliği, Jaccard Benzerliği) gerçekleştiren bir çalışmadır. Proje, farklı metin işleme tekniklerinin (lemmatization ve stemming) metin benzerliği üzerindeki etkilerini karşılaştırmalı olarak incelemeyi ve buna dayalı bir şarkı öneri sistemi prototipi oluşturmayı hedeflemektedir.

## İçindekiler

- [Proje Amacı](#proje-amacı)
- [Veri Seti](#veri-seti)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Dosya Yapısı](#dosya-yapısı)
- [Metodoloji](#metodoloji)
  - [Veri Ön İşleme](#veri-ön-işleme)
  - [Vektörleştirme](#vektörleştirme)
  - [Metin Benzerliği Analizi](#metin-benzerliği-analizi)
- [Sonuçlar ve Değerlendirme](#sonuçlar-ve-değerlendirme)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)
- [İletişim](#iletişim)

## Proje Amacı

Bu projenin temel amaçları şunlardır:

1.  **Metin Ön İşleme:** Şarkı açıklamalarını temizleme, normalleştirme ve farklı metin işleme teknikleri (lemmatization ve stemming) uygulayarak analiz için hazır hale getirme.
2.  **Vektörleştirme:** Ön işlenmiş metinleri TF-IDF ve Word2Vec gibi teknikler kullanarak sayısal vektörlere dönüştürme.
3.  **Metin Benzerliği:** Vektörleştirilmiş metinler arasında Kosinüs Benzerliği ve Jaccard Benzerliği gibi metriklerle benzerlik analizi yapma.
4.  **Model Karşılaştırması:** Lemmatization ve stemming'in farklı vektörleştirme yöntemleri ve benzerlik metrikleri üzerindeki etkilerini karşılaştırma.

## Veri Seti

Projede kullanılan veri seti `top500song.csv` adlı dosyadan gelmektedir. Bu veri seti, her bir şarkının başlığını, sanatçısını ve açıklamasını içeren popüler şarkıların bir listesini barındırmaktadır.

-   `top500song.csv`: 500 adet şarkının bilgisini içeren veri dosyası.

## Kurulum

Projeyi yerel olarak çalıştırmak için aşağıdaki adımları izleyin:

1.  Depoyu klonlayın:
    ```bash
    git clone [https://github.com/emnrmn0963/Sarki-aciklamasiyla-otomatik-tur-tahmini-1.git](https://github.com/emnrmn0963/Sarki-aciklamasiyla-otomatik-tur-tahmini-1.git)
    cd Sarki-aciklamasiyla-otomatik-tur-tahmini-1
    ```
2.  Gerekli kütüphaneleri yükleyin (Python 3.x önerilir):
    ```bash
    pip install pandas numpy scikit-learn nltk gensim matplotlib
    ```
3.  NLTK veri paketlerini indirin (gerekirse):
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

## Kullanım

Proje, Jupyter Notebook'lar aracılığıyla adım adım incelenebilir. Sırasıyla aşağıdaki notebook'ları çalıştırmanız önerilir:

1.  `02_datapreproccess.ipynb`: Ham verinin ilk ön işleme adımlarını (temizleme, küçük harfe çevirme, noktalama işaretlerini kaldırma vb.) içerir.
2.  `lemmatizeandstemmed.ipynb`: Ön işlenmiş metinlere lemmatization ve stemming işlemlerini uygular ve sonuçları karşılaştırır.
3.  `vectorization_tf-idf.ipynb`: TF-IDF vektörleştirmesini gerçekleştirir ve TF-IDF matrislerini oluşturur.
4.  `word2vec.ipynb`: Word2Vec modellerini eğitir ve kelime vektörlerini oluşturur.
5.  `metin_benzerligi.ipynb`: Farklı vektörleştirme yöntemleri ve ön işleme teknikleri kullanılarak metinler arası benzerlik analizini (Kosinüs ve Jaccard) yapar ve sonuçları görselleştirir.

## Dosya Yapısı
├── data/ <br>
│   ├── top500song.csv                  # Ham veri seti  <br>
│   ├── preprocessed_data_lemmatized_only.csv # Lemmatized metinlerin ön işlenmiş hali  <br>
│   ├── preprocessed_data_stemmed_only.csv    # Stemmed metinlerin ön işlenmiş hali    <br>
│   ├── tfidf_lemmatized.csv            # Lemmatized veriden oluşturulan TF-IDF matrisi <br>
│   └── tfidf_stemmed.csv               # Stemmed veriden oluşturulan TF-IDF matrisi    <br>
├── models/   <br>
│   ├── Eğitilmiş 16 adet model dosyası  <br>
├── notebooks/   <br>
│   ├── 02_datapreproccess.ipynb        # Veri ön işleme adımları     <br> 
│   ├── lemmatizeandstemmed.ipynb       # Lemmatization ve Stemming karşılaştırması   <br>
│   ├── vectorization_tf-idf.ipynb      # TF-IDF vektörleştirme    <br>
│   ├── word2vec.ipynb                  # Word2Vec model eğitimi   <br>
│   └── metin_benzerligi.ipynb          # Metin benzerliği analizi ve sonuçlar   <br>
└── README.md                           # Bu dosya   <br>

## Metodoloji

Proje, aşağıdaki adımları takip eden bir metodolojiye sahiptir:

### Veri Ön İşleme

`02_datapreproccess.ipynb` ve `lemmatizeandstemmed.ipynb` notebook'larında, `top500song.csv` dosyasındaki `description` sütunu üzerinde kapsamlı bir ön işleme yapılır:

* Küçük harfe çevirme
* Noktalama işaretlerini ve özel karakterleri kaldırma
* Sayıları kaldırma
* Stop word'leri kaldırma
* **Lemmatization:** Kelimeleri kök hallerine dönüştürme (örneğin, "running" -> "run").
* **Stemming:** Kelimelerin eklerini kaldırarak daha kısa kök formlarına indirgeme (örneğin, "runner" -> "run").

Bu işlemler sonucunda `preprocessed_data_lemmatized_only.csv` ve `preprocessed_data_stemmed_only.csv` dosyaları oluşturulur.

### Vektörleştirme

Ön işlenmiş metinler iki farklı yöntemle sayısal vektörlere dönüştürülür:

* **TF-IDF (`vectorization_tf-idf.ipynb`):** Her bir kelimenin bir dokümandaki sıklığını ve tüm dokümanlardaki nadirliğini ölçerek önemini belirleyen bir tekniktir. Hem lemmatize edilmiş hem de stemmed veri setleri için TF-IDF matrisleri (`tfidf_lemmatized.csv`, `tfidf_stemmed.csv`) oluşturulur.
* **Word2Vec (`word2vec.ipynb`):** Kelimelerin bağlamlarına göre vektör temsillerini öğrenen sinir ağı tabanlı bir yöntemdir. Farklı parametrelerle (CBOW/Skip-gram, window size, vector size) Word2Vec modelleri eğitilir.

### Metin Benzerliği Analizi

`metin_benzerligi.ipynb` notebook'unda, farklı vektörleştirme yöntemleri ve ön işleme teknikleriyle oluşturulan metin temsilleri arasında benzerlik analizi yapılır:

* **Kosinüs Benzerliği:** Vektörler arasındaki açının kosinüsünü ölçerek benzerliği belirler. Özellikle TF-IDF ve Word2Vec vektörleri için kullanılır.
* **Jaccard Benzerliği:** İki kümenin kesişiminin birleşimine oranını hesaplayarak benzerliği ölçer. Genellikle kelime setleri veya belge kümeleri için kullanılır.

Bu analizler sonucunda, farklı modellerin benzer şarkılar önerme yetenekleri karşılaştırılır ve değerlendirilir.

## Sonuçlar ve Değerlendirme

`metin_benzerligi.ipynb` notebook'u, farklı ön işleme ve vektörleştirme yöntemlerinin metin benzerliği skorları üzerindeki etkilerini gösteren görselleştirmeler ve matrisler sunar. Bu bölümde, projenin bulguları özetlenebilir ve hangi kombinasyonların belirli senaryolar için daha iyi sonuçlar verdiği belirtilebilir. Örneğin, lemmatization'ın stemming'e göre daha iyi sonuçlar verip vermediği, TF-IDF'in mi yoksa Word2Vec'in mi belirli görevler için daha uygun olduğu gibi çıkarımlar yapılabilir.




