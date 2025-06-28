import pandas as pd  # Digunakan untuk manipulasi dan analisis data dalam bentuk tabel (dataframe)
pd.options.mode.chained_assignment = None  # Menonaktifkan peringatan terkait chained assignment pada dataframe
import numpy as np  # Digunakan untuk melakukan komputasi numerik dan manipulasi array
seed = 0
np.random.seed(seed)  # Mengatur nilai seed agar hasil random dapat direproduksi

import matplotlib.pyplot as plt  # Digunakan untuk membuat visualisasi data berupa grafik
import seaborn as sns  # Digunakan untuk visualisasi data statistik dengan tampilan yang lebih menarik

import datetime as dt  # Digunakan untuk memanipulasi data yang berhubungan dengan waktu dan tanggal
import re  # Modul untuk memproses teks menggunakan pola ekspresi reguler (regex)
import string  # Berisi konstanta yang berkaitan dengan karakter teks, seperti tanda baca

import nltk  # Mengimpor pustaka Natural Language Toolkit (NLTK) yang digunakan untuk pemrosesan bahasa alami
nltk.download('punkt_tab')  # Mengunduh data pendukung yang dibutuhkan untuk proses tokenisasi teks (pemecahan teks menjadi kata atau kalimat)
nltk.download('stopwords')  # Mengunduh daftar stopwords, yaitu kata-kata umum yang sering diabaikan dalam analisis teks
from nltk.tokenize import word_tokenize  # Berfungsi untuk memecah teks menjadi token atau kata-kata
from nltk.corpus import stopwords  # Menyediakan daftar kata umum (stopwords) yang dapat dihapus dari teks

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # Digunakan untuk melakukan stemming (menghilangkan imbuhan) pada kata-kata dalam bahasa Indonesia
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory  # Digunakan untuk menghapus kata-kata umum (stopwords) dalam bahasa Indonesia
from wordcloud import WordCloud  # Digunakan untuk membuat visualisasi data teks dalam bentuk awan kata (word cloud)

import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import os
import joblib

# load dataset
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  
data_path = os.path.join(base_path, "datasets")

# Path lengkap ke file CSV
df = pd.read_csv("clean_merged.csv", encoding="ISO-8859-1")

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text)  # remove hashtags
    text = re.sub(r'RT[\s]', '', text)         # remove RT
    text = re.sub(r"http\S+", '', text)        # remove links
    text = re.sub(r'[0-9]+', '', text)         # remove numbers
    text = re.sub(r'[^\w\s]', '', text)        # remove punctuation
    text = text.replace('\n', ' ')             # replace newline with space
    text = text.replace('Â', '')               # remove character Â
    text = text.replace('â€˜', '')             # remove character â€˜
    text = text.replace('â€™', '')             # remove character â€™

    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip(' ')  # remove leading/trailing whitespace

    # Drop satuan atau kata tidak penting
    useless_words = [
        'centimeter', 'cm', 'kg', 'gram', 'x', 'buah', 'ready', 'order', 'unboxing', 'note', 'pembelian', 'ready', 'stock',
        'satuan', 'min', 'pcs', 'pack', 'ml', 'meter', 'mm', 'ltr', 'liter', 'unit', 'item', 'asli', 'unboxing', 'video', 'detail', 'size', 'diamater',
        'xx', 'senin', 'jumat', 'sabtu', 'minggu','wa', 'ukuran', 'deskripsi', 'pemesanan', 'kategori', 'tersedia', 'harga', 'grosir', 'hubungi', 'wa',
        'barang', 'pengiriman', 'indonesia', 'etalase', 'kondisi', 'berat', 'satuan', 'etalase', 'baru'
    ]

    text = ' '.join([word for word in text.split() if word.lower() not in [w.lower() for w in useless_words]])

    # Kalau hasilnya cuma 1 kata atau kosong, balikin kosong saja
    if len(text.split()) <= 1:
        return ''

    return text

def casefoldingText(text): # Converting all the characters in a text into lower case
    text = text.lower()
    return text

def tokenizingText(text): # Tokenizing or splitting a string, text into a list of tokens
    text = word_tokenize(text)
    return text

def filteringText(text): # Remove stopwors in a text
    listStopwords = set(stopwords.words('indonesian'))
    listStopwords1 = set(stopwords.words('english'))
    listStopwords.update(listStopwords1)
    listStopwords.update(['iya','yaa','gak','nya','na','sih','ku',"di","ga","ya","gaa","loh","kah","woi","woii","woy"])
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text

def stemmingText(text): # Reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words
    # Membuat objek stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Memecah teks menjadi daftar kata
    words = text.split()

    # Menerapkan stemming pada setiap kata dalam daftar
    stemmed_words = [stemmer.stem(word) for word in words]

    # Menggabungkan kata-kata yang telah distem
    stemmed_text = ' '.join(stemmed_words)

    return stemmed_text

    return stemmed_text
def toSentence(list_words): # Convert list of words into sentence
    sentence = ' '.join(word for word in list_words)
    return sentence

slangwords = slangwords = {"@": "di", "abis": "habis", "wtb": "beli", "epek":"efek", "dll": "dan lain lain", "masi": "masih", "wts": "jual", "wtt": "tukar", "bgt": "banget", "maks": "maksimal",
              "tpi":"tapi","tp":"tapi","g":"tidak","knp":"kenapa","tibatiba":"tiba-tiba","ad":"ada", "mm" : "milimeter", "sugarcane" : "tebu", "plastic" : "plastik", "detergen" : "deterjen", "liquid":"cairan",
              "tbtb":"tiba-tiba","yt":"youtube","ig":"instagram","gk":"tidak","yg":"yang","moga":"semoga","pake":"pakai","ngirim":"kirim", "top" : "atas", "paper" : "kertas", "bukaan" : "bukan",
              "muas":"puas","sdh":"sudah","lg":"lagi","sya":"saya","klo":"kalau","knpa":"kenapa","tdk":"tidak","sampe":"sampai","kayak":"seperti", "color" : "warna", "organic":"organik", "bagasse":"ampas",
              "cuman":"hanya","prose":"proses","ny":"nya","jd":"jadi","dgn":"dengan","jg":"juga","tf":"transfer","sampe":"sampai","ngirim":"kirim", "dye" : "perwarna",
              "bagu":"bagus","skrg":"sekarang","nunggu":"tunggu","udah":"sudah","uda":"sudah","pk":"pakai", "plisss": "tolong", "bgttt": "banget", "indo": "indonesia",
              "bgtt": "banget", "ad": "ada", "rv": "redvelvet", "plis": "tolong", "pls": "tolong", "cr": "sumber", "cod": "bayar ditempat", "adlh": "adalah", "chemical":"kimia",
              "afaik": "as far as i know", "aj": "saja", "ajep-ajep": "dunia gemerlap", "out of stock" : "stock habis", "silicone" : "silikon", "toothbrush":"sikat gigi",
              "akuwh": "aku", "alay": "norak", "alow": "halo", "ambilin": "ambilkan", "ancur": "hancur","anter": "antar", "ap2": "apa-apa",
              "apasih": "apa sih", "apes": "sial", "aps": "apa", "aq": "saya", "aquwh": "aku", "asbun": "asal bunyi","hp" : "ponsel", "ory" : "original",
              "asem": "asam", "aspal": "asli tetapi palsu", "astul": "asal tulis", "ato": "atau", "au ah": "tidak mau tahu", "awak": "saya", "ay": "sayang",
              "ayank": "sayang", "b4": "sebelum", "bakalan": "akan", "bangedh": "banget","ingredient" : "bahan", "baby":"bayi",
              "basbang": "basi", "bcanda": "bercanda", "bdg": "bandung", "beliin": "belikan", "bencong": "banci", "bentar": "sebentar",
              "ber3": "bertiga", "beresin": "membereskan", "bg": "abang", "bgmn": "bagaimana", "bgt": "banget", "bijimane": "bagaimana",
              "bintal": "bimbingan mental", "bkl": "akan", "bknnya": "bukannya", "blh": "boleh", "bln": "bulan", "blum": "belum", "bnci": "benci", "bnran": "yang benar",
              "bodor": "lucu", "bokap": "ayah", "bokis": "bohong", "boljug": "boleh juga", "bonek": "bocah nekat", "boyeh": "boleh", "br": "baru", "brg": "bareng",
              "bro": "saudara laki-laki", "bru": "baru", "bs": "bisa", "bsen": "bosan", "bt": "buat", "btw": "ngomong-ngomong", "bubbu": "tidur", "bubu": "tidur",
              "bumil": "ibu hamil", "bw": "bawa", "bwt": "buat", "byk": "banyak", "byrin": "bayarkan","calo": "makelar",
              "caper": "cari perhatian", "ce": "cewek", "cekal": "cegah tangkal", "cemen": "penakut", "cengengesan": "tertawa", "cepet": "cepat", "cew": "cewek", "chuyunk": "sayang", "cimeng": "ganja",
              "cipika cipiki": "cium pipi kanan cium pipi kiri","cmiiw": "correct me if i'm wrong", "cmpur": "campur", "bubble wrap" : "plastik",
              "curcol": "curahan hati colongan", "cwek": "cewek", "d": "di", "dah": "deh", "dapet": "dapat", "de": "adik", "dek": "adik", "demen": "suka", "deyh": "deh", "dgn": "dengan",
              "diancurin": "dihancurkan", "dimaafin": "dimaafkan", "dimintak": "diminta", "disono": "di sana", "dket": "dekat", "dkk": "dan kawan-kawan", "dll": "dan lain-lain", "dlu": "dulu", "dngn": "dengan",
              "dodol": "bodoh", "doku": "uang", "dongs": "dong", "dpt": "dapat", "dri": "dari", "drmn": "darimana", "drtd": "dari tadi", "dst": "dan seterusnya", "dtg": "datang", "duh": "aduh", "duren": "durian",
              "ed": "edisi", "egp": "emang gue pikirin", "emng": "memang", "endak": "tidak", "enggak": "tidak", "envy": "iri", "ex": "mantan", "fax": "facsimile",
              "fifo": "first in first out", "folbek": "follow back", "fyi": "sebagai informasi", "gaada": "tidak ada uang", "gag": "tidak", "gaje": "tidak jelas", "gak papa": "tidak apa-apa", "gan": "juragan"}

def normalize_slang(text):
    tokens = text.split()
    normalized_tokens = []

    for token in tokens:
        normalized_token = slangwords.get(token.lower(), token)
        normalized_tokens.append(normalized_token)

    return ' '.join(normalized_tokens)

clean_df = df.copy()

clean_df['text_clean'] = clean_df['Description'].apply(cleaningText)
clean_df['text_casefoldingText'] = clean_df['text_clean'].apply(casefoldingText)
clean_df['text_slangwords'] = clean_df['text_casefoldingText'].apply(normalize_slang)
clean_df['text_tokenizingText'] = clean_df['text_slangwords'].apply(tokenizingText)
clean_df['text_stopword'] = clean_df['text_tokenizingText'].apply(filteringText)
clean_df.head()

# Melakukan pembersihan awal pada teks dan menyimpannya ke dalam kolom 'text_clean'
clean_df['text_clean'] = clean_df['Description'].apply(cleaningText)

# Mengubah seluruh huruf pada teks menjadi huruf kecil (case folding) dan menyimpannya ke dalam kolom 'text_casefoldingText'
clean_df['text_casefoldingText'] = clean_df['text_clean'].apply(casefoldingText)

# Menormalisasi kata-kata tidak baku (slang) menjadi bentuk baku dan menyimpannya ke dalam kolom 'text_slangwords'
clean_df['text_slangwords'] = clean_df['text_casefoldingText'].apply(normalize_slang)

# Melakukan tokenisasi (memecah teks menjadi kata-kata) dan menyimpannya ke dalam kolom 'text_tokenizingText'
clean_df['text_tokenizingText'] = clean_df['text_slangwords'].apply(tokenizingText)

# Menghapus kata-kata umum yang tidak memiliki makna penting (stopwords) dan menyimpannya ke dalam kolom 'text_stopword'
clean_df['text_stopword'] = clean_df['text_tokenizingText'].apply(filteringText)

# Menggabungkan kembali token menjadi satu kalimat utuh dan menyimpannya ke dalam kolom 'text_akhir'
clean_df['text_akhir'] = clean_df['text_stopword'].apply(toSentence)

# Fungsionalitas prediksi
def predict_sentiment(text):
    # 1. Preprocessing
    cleaned = cleaningText(text)
    folded = casefoldingText(cleaned)
    fixed = normalize_slang(folded)
    tokens = tokenizingText(fixed)
    filtered = filteringText(tokens)
    sentence = toSentence(filtered)

    # 2. Vektorisasi
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    vectorized = vectorizer.transform([sentence])

    # 3. Prediksi
    svm = SVC()
    svm = joblib.load('svm_model.pkl')
    pred_probs = svm.predict_proba(vectorized)[0]

    # 4. Interpretasi
    labels = svm.classes_  # ['greenwashing', 'greenhonesty']
    label_dict = dict(zip(labels, pred_probs))

    # Menampilkan probabilitas untuk setiap label
    result = ""
    for label, prob in label_dict.items():
        result += f"{label}: {prob:.4f}\n"  # Menampilkan probabilitas dengan format 4 desimal

    # Prediksi label yang memiliki probabilitas tertinggi
    predicted_label = max(label_dict, key=label_dict.get)
    
    # Menambahkan hasil prediksi ke output
    result += f"\nPrediksi Sentimen: {predicted_label}"

    return result, label_dict

# --- Layout & Config ---
st.set_page_config(page_title="Greenwashing Detector", page_icon="🌿", layout="wide")

# --- Sidebar with Option Menu ---
with st.sidebar:
    selected = option_menu("Navigasi", 
        ["📌 Apa Itu Greenwashing?", "🧪 Cek Deskripsi Produk", "📚 Edukasi & Tips"], 
        icons=["info-circle", "search", "book"], 
        menu_icon="leaf", 
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f0f2f6"},
            "icon": {"color": "green", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
            "nav-link-selected": {"background-color": "#A4E5D9"},
        }
    )

# --- Judul Halaman Utama ---
st.title("🌿 Greenwashing Detector: Cek Klaim Lingkunganmu!")

# --- Halaman: Apa Itu Greenwashing? ---
if selected == "📌 Apa Itu Greenwashing?":
    st.header("📌 Apa Itu Greenwashing?")
    st.markdown("""
        **Greenwashing** adalah strategi pemasaran yang membuat produk atau perusahaan terlihat ramah lingkungan,
        padahal sebenarnya tidak. Tujuannya? Supaya konsumen seperti kamu tertarik membeli karena kesan 'hijau' 🌿

        #### Contoh Greenwashing:
        - 🌱 Kemasan hijau dan gambar daun, tapi isinya bahan kimia berbahaya  
        - ♻️ Klaim "eco-friendly" tanpa bukti atau sertifikasi resmi  
        - 🔍 Fokus pada satu aspek kecil yang 'hijau', padahal keseluruhan prosesnya tidak ramah lingkungan

        Mari jadi konsumen cerdas dengan mendeteksi greenwashing sejak dari deskripsinya!
    """)

# --- Halaman: Cek Deskripsi Produk ---

elif selected == "🧪 Cek Deskripsi Produk":
    st.header("🧪 Cek Deskripsi Produkmu di Sini")

    st.markdown(
    """
    <style>
    /* Styling button submit di form */
    div.stButton > button:first-child {
        background-color: #A4E5D9;
        color: black;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 20px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #A4E5D9; /* warna hover bisa diubah */
        color: #A4E5D9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

    with st.form(key='cek_deskripsi_form'):
        st.markdown(
            """
            <style>
            div.stButton > button {
                background-color: #4CAF50;  /* hijau */
                color: white;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 24px;
                border: none;
                transition: background-color 0.3s ease;
            }
            div.stButton > button:hover {
                background-color: #A4E5D9;  
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        user_input = st.text_area("Masukkan deskripsi produk yang ingin kamu cek:", height=150)
        submit_button = st.form_submit_button(label='Cek Deskripsi')


    if submit_button and user_input:    
        with st.spinner('Memproses deskripsi... Mohon tunggu sebentar.'):
            result, label_dict = predict_sentiment(user_input)

        st.subheader("Hasil Prediksi Sentimen:")
        st.text_area("Hasil Sentimen dan Probabilitas", result, height=250)

        if "greenwashing" in label_dict and label_dict["greenwashing"] > 0.5:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("greenwashing_detected.png", caption="Oh no! Greenwashing detected! 🚨", width=400)
            st.warning("Hati-hati! Produk ini mungkin greenwashing. Pastikan klaimnya benar! 🚨")

        elif "greenhonesty" in label_dict and label_dict["greenhonesty"] > 0.5:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("greenhonesty_detected.png", caption="Yeay! Greenhonesty detected! 🌱", width=400)
            st.success("Produk ini tampaknya memang ramah lingkungan! 🌍")

        st.subheader("Visualisasi Probabilitas Sentimen")
        fig, ax = plt.subplots(figsize=(5, 5))
        categories = list(label_dict.keys())
        values = list(label_dict.values())

        # Buat pie chart
        colors = ['#ff9999', '#66b3ff', '#99ff99']  # warna bisa kamu sesuaikan
        ax.pie(values, labels=categories, autopct='%1.1f%%', startangle=140, colors=colors)
        ax.axis('equal')  # buat supaya pie chart bulat sempurna
        st.pyplot(fig)

# --- Halaman: Edukasi & Tips ---
elif selected == "📚 Edukasi & Tips":
    st.header("📚 Edukasi & Tips Konsumen Cerdas")
    st.markdown("""
        ### Kenapa penting mengenali greenwashing?
        Karena konsumen punya kekuatan untuk menuntut kejujuran dan keberlanjutan dari brand.

        ### Tips Mendeteksi Greenwashing:
        - Cari **sertifikasi resmi** (misalnya: Ecolabel, USDA Organic, dll)  
        - Waspadai klaim umum seperti "natural", "eco", tanpa penjelasan lanjut  
        - Baca deskripsi lengkap dan cek apakah klaim didukung data  
        - Lihat praktik nyata brand, bukan hanya kemasannya  

        🌱 Semakin kita sadar, semakin kecil ruang untuk praktik greenwashing.
    """)
