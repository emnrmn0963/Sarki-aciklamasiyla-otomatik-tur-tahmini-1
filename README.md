# Şarkı Açıklamaları İle Otomatik Tür Tahmini
Bu projede şarkı açıklamalarının  metinsel içeriğine göre şarkıyı söyleyen sanatçıyı veya grubu tahmin etme işlemi üzerine işlemler gerçekleştirilecektir.
Özellikleri tam anlamıyla karşılayan bir veri seti bulunamadığı için şarkı açıklamalarının metinsel içeriğine göre sanatçıların ürün ortaya koyduğu belirli türler 
üzerinden ilerleme sağlanacaktır. Çeşitli yöntemler denendiysede sitelerin güvenlik politikaları nedeniyle verilere ulaşılamamıştır.
İnternet üzerinden ulaşabilecek tüm veriler analiz edilerek kaggle üzerinden Top 500 Songs.csv dosyası özellikler açısından daha uygun olduğu için işleme alınmıştır.
Mevcut veri, 500 satır ve 9 tane sütündan oluşmaktadır. 281 kb ve .csv formatında bulunmaktadır. 
Bu çalışma yapıldığı sırada örnek alınan veri seti kaggle kullanıcısı tarafından değiştirilmiştir. Aşağıda paylaşılan bağlantı üzerinden yeni verilere erişim sağlayabilirsiniz.
https://www.kaggle.com/code/mpwolke/500-greatest-songs/input

# Model Oluşturma Aşaması
 1- Kütüphane kurulum işlemleri ve veriyi çekme işlemleri
import pandas as pd
import numpy as np

import nltk # Metin tabanlı bir işlem gerçekleştireceğimiz için bu kütüphaneyi kullanıyoruz.
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import csv # Dosyamızın hangi formatta olduğunu dikkate alarak bu kısmı değiştiriyoruz.

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # nltk işlemlerini gerçekleştirebilmemiz için gerekli ayarlamaları yapıyoruz.

df = pd.read_csv('top500song.csv')
df.head() # daha sonrasında mevcut dosyamızı çekerek ilk 5 verimizi görüntülüyoruz.
2- Ön İşleme (Pre-processing) işlemleri
texts = df['description'].dropna().tolist() # işlem yapmak istediğimiz sütunu seçiyoruz.

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() # Lemmatizer ve Stemmer işlemini başlatıyoruz.

text = df['description'][0]  # Açıklama kısmını ele alarak cümle ayırma işlemini gerçekleştiriyoruz.
sentences = sent_tokenize(text)
sentences[:10]

stop_words = set(stopwords.words('english')) # Stopwords listesini ingilizce olarak alıyoruz.

stop_words_list = list(stop_words)
print(stop_words_list[:50]) # 50 tanesini görüyoruz.

filtered_sentences = []  Kelimeleri tokenleştirip sadece harf olan kelimeleri alıyoruz ve stopword'leri çıkartıyoruz


for sentence in sentences:
     tokens = word_tokenize(sentence) #cümleleri kelimelere bölüyoruz ve boş bir liste oluşturuyoruz.
     filtered_tokens = [] 
3- Lemmatization işlemi
      Kelimeleri tokenleştirip, lemmatize etme ve stemleme
def preprocess_sentence(sentence):
     tokens = word_tokenize(sentence)

     for token in tokens:
        if token.isalpha(): Tokenlerin metin olup olmadığını kontrol ediyor (nümerik olan verileri işleme almıyor).
            token_lower = token.lower()  küçük harfe çevirme işlemi
            if token_lower not in stop_words:  Eğer küçük harfe çevrilmiş mevcut kelimeler stopword içinde yer almıyorsa
                               filtered_tokens.append(token_lower)  filtered_tokens listesine yukarıda belirtilen kurallara uygun yeni kelime ekle.
        filtered_sentences.append(filtered_tokens) filtre edilmiş cümleleri filtered_sentences listesine ekle.
        
print(filtered_sentences[:10])  ilk 10 cümleyi ekrana yazdır.

lemmatizer = WordNetLemmatizer()  Her cümleyi lemetize et ve Lemmatizeri başlat
tokenized_corpus_lemmatized = []   Lemma edilmiş cümleleri saklamak için yeni bir boş liste oluştur.
for filtered_tokens in filtered_sentences:
    lemmatized_tokens = []  Lemma edilmiş kelimeleri saklamak için boş bir liste oluştur.
    for token in filtered_tokens:
        lemma = lemmatizer.lemmatize(token)  Tokenleri tek tek lemma etme işlemi.
        lemmatized_tokens.append(lemma)  Lemma edilmiş tokenleri lemmatized_tokens listesine ekleme işlemi.
        tokenized_corpus_lemmatized.append(lemmatized_tokens)  Lemma edilmiş cümleleri 
        #tokenized_corpus_lemmatized ekleme işlemi.
print(tokenized_corpus_lemmatized[:10])


 lemmatize edilmiş cümleleri bir csv dosyasına kaydedilmesi gerekmektedir.
with open(r"C:\Users\lenovo\Desktop\metin_verileri_ile_dogal_dil_isleme\lemmatized_sentences.csv", mode="w", newline="", 
          encoding="utf-8") as file:
    writer = csv.writer(file)
    # Her cümleyi bir satır olarak yazma komutu.
    for tokens in tokenized_corpus_lemmatized:
        writer.writerow([' '.join(tokens)])

 Aynı kod parçalarını Stemmed edilmiş veriler içinde bazı parametreleri değiştirerek yapabiliriz. Daha sonrasında kaydedip vektörleştirme işlemini gerçekleştireceğiz.

4- TF-IDF vektörleştirme işlemi için çağırıyoruz.
import pandas as pd
dflemma = pd.read_csv(r"C:\Users\lenovo\Desktop\metin_verileri_ile_dogal_dil_isleme\lemmatized_sentences.csv", encoding='utf-8')
dflemma.head(5) # İlk 5 verimizi bu kod ile ekrana yansıtıyoruz.

from sklearn.feature_extraction.text import TfidfVectorizer
 Ön işlenmiş token listelerini tekrar metne çeviriyoruz
lemmatized_texts = [' '.join(tokens) for tokens in tokenized_corpus_lemmatized]
lemmatized_texts[:3] 

 TF-IDF vektörizer'ı başlatıyoruz
vectorizer = TfidfVectorizer()
 TF-IDF matrisini oluşturuyoruz
 Terim frekanslarını, belge frekanslarını hesaplar
 TF-IDF vektörlerine dönüştürür
tfidf_matrix = vectorizer.fit_transform(lemmatized_texts)
 TF-IDF vektörleştirme işleminde kullanılan tüm kelimelerin eşsiz bir listesini döndürür
feature_names = vectorizer.get_feature_names_out()
 TF-IDF matrisini pandas DataFrame'e çevir – görünürlük açısından – çalışması kolaydır
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
 İlk birkaç satırı gösterelim – ilk 5 cümle
print(tfidf_df.head())
tfidf_df.to_csv("tfidf_lemmatized.csv",index=False)
 Her satır bir cümleyi temsil eder
 Her sütun bir kelimeyi temsil eder
 Hücreler ise o kelimenin o cümledeki TF-IDF skorudur – her cümle için değişiklik gösterir.

 İlk cümle için TF-IDF skorlarını alıyoruz
first_sentence_vector = tfidf_df.iloc[0]
 Skorlara göre sırala (yüksekten düşüğe)
top_5_words = first_sentence_vector.sort_values(ascending=False).head(5)
 Sonucu yazdır
print("İlk cümlede en yüksek TF-IDF skoruna sahip 5 kelime:")
print(top_5_words)  İlk cümlede yer alan 5 kelimeyi skorları yüksekten düşüğe olacak şekilde sıralıyor.
* TF-IDF vektörleştirme işlemi için "cosine similarty" ile kelimelerin benzerlik oranları hesaplanır.
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
 like kelimesinin vektörünü alalım
like_index = feature_names.tolist().index('like')  'like' kelimesinin indeksini bul
 like kelimesinin TF-IDF vektörünü alıyoruz ve 2D formatta yapıyoruz
like_vector = tfidf_matrix[:, like_index].toarray()
 Tüm kelimelerin TF-IDF vektörlerini alıyoruz
tfidf_vectors = tfidf_matrix.toarray()
 Cosine similarity hesaplayalım
similarities = cosine_similarity(like_vector.T, tfidf_vectors.T)
#Benzerlikleri sıralayalım ve en yüksek 5 kelimeyi seçelim
similarities = similarities.flatten()
top_5_indices = similarities.argsort()[-6:][::-1]  6. en büyükten başlıyoruz çünkü kendisi de dahil
 Sonuçları yazdıralım
for index in top_5_indices:
    print(f"{feature_names[index]}: {similarities[index]:.4f}")
 Örnek bir kelime üzerinden vektör işlemi yapıyoruz ve "like" kelimesine en benzer olan kelimeleri benzerlik skorları ile birlikte görüyoruz.
 TF-IDF vektörleme işlemini bazı parametreleri değiştirerek Stemm edilmiş veriler içinde kullanabiliriz.

5- Word2vec Vektörleştirme işlemi 
Stemmed edilmiş veriler üzerinden görelim.
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import csv  Öncelikle kütüphanelerimizi import ederek başlıyoruz. Tek bir dosyada çalışıyorsanız sadece eksik kütüphaneleri import etmeniz yeterli.

import pandas as pd
dfstemmsen = pd.read_csv(r"C:\Users\lenovo\Desktop\metin_verileri_ile_dogal_dil_isleme\stemmed_sentences.csv", encoding='utf-8')  İşlem yapacağımız dosyamızı çekiyoruz.

parameters = [
{'model_type': 'cbow', 'window': 2, 'vector_size': 100},
{'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
{'model_type': 'cbow', 'window': 4, 'vector_size': 100},
{'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
{'model_type': 'cbow', 'window': 2, 'vector_size': 300},
{'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
{'model_type': 'cbow', 'window': 4, 'vector_size': 300},
{'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
]  Parametrelerimizi düzgünce ve sırasıyla yazıyoruz.

def train_and_save_model(corpus, params, model_name):
    model = Word2Vec(corpus, vector_size=params['vector_size'],
 window=params['window'], min_count=1, sg=1 if params['model_type'] == 'skipgram' else 0)
    
    model.save(f"{model_name}_{params['model_type']}_window{params['window']}_dim{params['vector_size']}.model")
    print(f"{model_name}_{params['model_type']}_window{params['window']}_dim{params['vector_size']}.model saved!")

    
Lemmatize edilmiş corpus ile modelleri eğitme ve kaydetme
for param in parameters:
    train_and_save_model(tokenized_corpus_lemmatized, param, "lemmatized_model")
    # Stemlenmiş corpus ile modelleri eğitme ve kaydetme
for param in parameters:
    train_and_save_model(tokenized_corpus_stemmed, param, "stemmed_model")  Sonucunda 8 tane Stemmed ve 8 tane Lemmatized olmak üzere 16 tane model eğitilmiştir.

    from gensim.models import Word2Vec # Kütüphaneyi import ediyoruz.
 Model dosyalarını yüklemek
model_1 = Word2Vec.load("lemmatized_model_cbow_window2_dim100.model")
model_2 = Word2Vec.load("stemmed_model_skipgram_window4_dim100.model")
model_3 = Word2Vec.load("lemmatized_model_skipgram_window2_dim300.model")
 'young' kelimesi ile en benzer 3 kelimeyi ve skorlarını yazdırmak
def print_similar_words(model, model_name):
    similarity = model.wv.most_similar('young', topn=3)
    print(f"\n{model_name} Modeli - 'young' ile En Benzer 3 Kelime:")
    for word, score in similarity:
        print(f"Kelime: {word}, Benzerlik Skoru: {score}")
 3 model için benzer kelimeleri yazdır
print_similar_words(model_1, "Lemmatized CBOW Window 2 Dim 100")
print_similar_words(model_2, "Stemmed Skipgram Window 4 Dim 100")
print_similar_words(model_3, "Lemmatized Skipgram Window 2 Dim 300")
 Örnek olarak "young" kelimesi için en benzer 3 kelimeyi skorları ile birlikte ekrana yazdırmasını istiyoruz.
 Stemmed edilmiş veriler için uyguladığımız kod parçalarını Lemmatized edilmiş veriler için bazı parametreleri değiştirerek kullanabiliriz. 
6- Sonuçların Değerlendirilmesi
Şarkı açıklamaları üzerinden tür tahmini yapılmaya çalışılır.

# Veri Setinin Kullanılabileceği Diğer Analizler 
+ Keşifsel veri analizi
+ Zamana bağlı trend analizi
+ Sanatçı şarkı analizi 
+ Görselleştirme işlemleri

# Projede Kullanılan Kütüphaneler
Tüm kütüphaneler kod kısmında açıkça belirtilmiştir.

- Kütüphane             -Özellikleri
numpy -                   Çok boyutlu diziler,
                        yüksek performanslı sayısal işlemler ve
                        matrisleri işleme ve analiz etme konusunda
                        başarılıdır.

pandas -                 Veri işleme ve analiz için kullanıldı.

nltk -                   Doğal dil işleme görevlerini basitleştirir.

matplotlib -              Grafiklerin çizimi için kulanılmıştır.

scikit-learn -           TF-IDF vektörleştirme ve cosine similarity hesaplama işlemi için kullanıldı.

gensim -                Word2vec vektörleştirme için kullanıldı.




