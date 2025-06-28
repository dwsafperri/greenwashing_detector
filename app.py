import pandas as pd  # Digunakan untuk manipulasi dan analisis data dalam bentuk tabel (dataframe)
pd.options.mode.chained_assignment = None  # Menonaktifkan peringatan terkait chained assignment pada dataframe
import numpy as np  # Digunakan untuk melakukan komputasi numerik dan manipulasi array
import matplotlib.pyplot as plt  # Digunakan untuk membuat visualisasi data berupa grafik
import seaborn as sns  # Digunakan untuk visualisasi data statistik dengan tampilan yang lebih menarik
import streamlit as st
from streamlit_option_menu import option_menu
import datetime as dt  # Digunakan untuk memanipulasi data yang berhubungan dengan waktu dan tanggal
import re  # Modul untuk memproses teks menggunakan pola ekspresi reguler (regex)
import string  # Berisi konstanta yang berkaitan dengan karakter teks, seperti tanda baca
import nltk  # Mengimpor pustaka Natural Language Toolkit (NLTK) yang digunakan untuk pemrosesan bahasa alami
import os
import joblib
import csv
import requests
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize  # Berfungsi untuk memecah teks menjadi token atau kata-kata
from nltk.corpus import stopwords  # Menyediakan daftar kata umum (stopwords) yang dapat dihapus dari teks
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # Digunakan untuk melakukan stemming (menghilangkan imbuhan) pada kata-kata dalam bahasa Indonesia
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory  # Digunakan untuk menghapus kata-kata umum (stopwords) dalam bahasa Indonesia
from wordcloud import WordCloud  # Digunakan untuk membuat visualisasi data teks dalam bentuk awan kata (word cloud)

# Set random seed for reproducibility
seed = 0
np.random.seed(seed)  

# Configure NLTK resources
nltk.download('punkt_tab', quiet=True)  
nltk.download('stopwords', quiet=True)  

# Set page config
st.set_page_config(
    page_title="Greenwashing Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan yang lebih modern
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
      /* Image centering and responsiveness */
    .stImage {
        text-align: center;
        margin: 0 auto;
        display: block;
    }
    
    /* Center all images */
    img {
        display: block;
        margin-left: auto !important;
        margin-right: auto !important;
        max-width: 100%;
        height: auto;
        object-fit: contain;
    }
    
    /* Results section styling */
    .result-container {
        animation: fadeIn 0.5s ease-in-out;
        transition: all 0.3s ease;
    }
    
    /* Animation for results */
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    
    /* Card styling */
    .stcard {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Header styling */
    h1 {
        color: #1E5631;
        font-weight: 800;
        margin-bottom: 1.5rem;
    }
    
    h2, h3 {
        color: #2E7D32;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #3e8e41;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Text input styling */
    .stTextArea>div>div>textarea {
        border: 1px solid #4CAF50;
        border-radius: 5px;
    }
      /* Result box styling */
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    
    /* Responsive styling for results */
    .result-container {
        transition: all 0.3s ease;
    }
    
    .image-container img {
        transition: all 0.3s ease;
    }
    
    /* Metric cards styling */
    .metric-card {
        transition: all 0.3s ease;
        min-height: 150px;
    }
    
    /* Media query for smaller screens */
    @media screen and (max-width: 768px) {
        .image-container img {
            max-width: 200px;
        }
        
        .stcard {
            padding: 15px;
        }
    }
    
    /* Media query for mobile screens */
    @media screen and (max-width: 480px) {
        .image-container img {
            max-width: 180px;
        }
        
        .stcard {
            padding: 12px;
        }
    }
    
    .greenwashing {
        background-color: #FFEBEE;
        border-left: 6px solid #F44336;
    }
    
    .greenhonesty {
        background-color: #E8F5E9;
        border-left: 6px solid #4CAF50;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid #e0e0e0;
        color: #666;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f5f7f9;
    }
</style>
""", unsafe_allow_html=True)

# load dataset
df = pd.read_csv("clean_merged.csv", encoding="ISO-8859-1")

# Text Processing Functions
def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text)  # remove hashtags
    text = re.sub(r'RT[\s]', '', text)         # remove RT
    text = re.sub(r"http\S+", '', text)        # remove links
    text = re.sub(r'[0-9]+', '', text)         # remove numbers
    text = re.sub(r'[^\w\s]', '', text)        # remove punctuation
    text = text.replace('\n', ' ')             # replace newline with space

    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip(' ')  # remove leading/trailing whitespace

    # Drop satuan atau kata tidak penting
    useless_words = [
        'centimeter', 'cm', 'kg', 'gram', 'x', 'buah', 'ready', 'order', 'unboxing', 'note', 'pembelian', 'ready', 'stock',
        'satuan', 'min', 'pcs', 'pack', 'ml', 'meter', 'mm', 'ltr', 'liter', 'unit', 'item', 'asli', 'unboxing', 'video', 'detail', 'size', 'diamater',
        'xx', 'senin', 'jumat', 'sabtu', 'minggu','wa', 'ukuran', 'deskripsi', 'pemesanan', 'kategori', 'tersedia', 'harga', 'grosir', 'hubungi', 'wa',
        'barang', 'pengiriman', 'indonesia', 'etalase', 'kondisi', 'berat', 'satuan', 'etalase', 'baru', 'mx', 'a', 'gr', 'amp', 'ongkir', 'cod', 'g',
        'Ã¢', 'Ã°', 'â€™', 'â€˜', 'Â', 'gr', 'p', 't', 'l', 'terima', 'kasih', 'xcm', 'â', 'ââ', 'denganâ' 'chat', 'untukâ', 'danâ', 'chat', 'admin'
    ]

    text = ' '.join([word for word in text.split() if word.lower() not in [w.lower() for w in useless_words]])

    # Kalau hasilnya cuma 1 kata atau kosong, balikin kosong saja
    if len(text.split()) <= 1:
        return ''

    return text

def casefoldingText(text):
    """Convert all text to lowercase"""
    return text.lower() if isinstance(text, str) else ""

def tokenizingText(text):
    """Split text into tokens/words"""
    return word_tokenize(text) if isinstance(text, str) else []

def filteringText(text):
    """Remove stopwords from tokenized text"""
    listStopwords = set(stopwords.words('indonesian'))
    listStopwords.update(set(stopwords.words('english')))
    listStopwords.update(['iya','yaa','gak','nya','na','sih','ku',"di","ga","ya","gaa","loh","kah","woi","woii","woy"])
    
    return [txt for txt in text if txt not in listStopwords] if isinstance(text, list) else []

def stemmingText(text):
    """Reduce words to their root form"""
    if not isinstance(text, str):
        return ""
        
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    
    return ' '.join(stemmed_words)

def toSentence(list_words):
    """Convert list of words back to sentence"""
    return ' '.join(word for word in list_words) if isinstance(list_words, list) else ""

# Slang words dictionary for normalization
slangwords = {
    "@": "di", "abis": "habis", "wtb": "beli", "epek":"efek", "dll": "dan lain lain", "masi": "masih", 
    "wts": "jual", "wtt": "tukar", "bgt": "banget", "maks": "maksimal", "tpi":"tapi","tp":"tapi",
    "g":"tidak","knp":"kenapa","tibatiba":"tiba-tiba","ad":"ada", "mm" : "milimeter", 
    "sugarcane" : "tebu", "plastic" : "plastik", "detergen" : "deterjen", "liquid":"cairan",
    "tbtb":"tiba-tiba","yt":"youtube","ig":"instagram","gk":"tidak","yg":"yang","moga":"semoga",
    "pake":"pakai","ngirim":"kirim", "top" : "atas", "paper" : "kertas", "bukaan" : "bukan",
    "muas":"puas","sdh":"sudah","lg":"lagi","sya":"saya","klo":"kalau","knpa":"kenapa",
    "tdk":"tidak","sampe":"sampai","kayak":"seperti", "color" : "warna", "organic":"organik", 
    "bagasse":"ampas", "cuman":"hanya","prose":"proses","ny":"nya","jd":"jadi","dgn":"dengan",
    "jg":"juga","tf":"transfer","sampe":"sampai","ngirim":"kirim", "dye" : "perwarna",
    "bagu":"bagus","skrg":"sekarang","nunggu":"tunggu","udah":"sudah","uda":"sudah","pk":"pakai", 
    "plisss": "tolong", "bgttt": "banget", "indo": "indonesia", "bgtt": "banget", "ad": "ada", 
    "rv": "redvelvet", "plis": "tolong", "pls": "tolong", "cr": "sumber", "cod": "bayar ditempat", 
    "adlh": "adalah", "chemical":"kimia", "afaik": "as far as i know", "aj": "saja", 
    "ajep-ajep": "dunia gemerlap", "out of stock" : "stock habis", "silicone" : "silikon", 
    "toothbrush":"sikat gigi", "akuwh": "aku", "alay": "norak", "alow": "halo", "ambilin": "ambilkan", 
    "ancur": "hancur","anter": "antar", "ap2": "apa-apa", "apasih": "apa sih", "apes": "sial", 
    "aps": "apa", "aq": "saya", "aquwh": "aku", "asbun": "asal bunyi","hp" : "ponsel", "ory" : "original",
    "asem": "asam", "aspal": "asli tetapi palsu", "astul": "asal tulis", "ato": "atau", 
    "au ah": "tidak mau tahu", "awak": "saya", "ay": "sayang", "ayank": "sayang", "b4": "sebelum", 
    "bakalan": "akan", "bangedh": "banget","ingredient" : "bahan", "baby":"bayi",
    "basbang": "basi", "bcanda": "bercanda", "bdg": "bandung", "beliin": "belikan", 
    "bencong": "banci", "bentar": "sebentar", "ber3": "bertiga", "beresin": "membereskan", 
    "bg": "abang", "bgmn": "bagaimana", "bgt": "banget", "bijimane": "bagaimana",
    "bintal": "bimbingan mental", "bkl": "akan", "bknnya": "bukannya", "blh": "boleh", 
    "bln": "bulan", "blum": "belum", "bnci": "benci", "bnran": "yang benar",
    "bodor": "lucu", "bokap": "ayah", "bokis": "bohong", "boljug": "boleh juga", 
    "bonek": "bocah nekat", "boyeh": "boleh", "br": "baru", "brg": "bareng",
    "bro": "saudara laki-laki", "bru": "baru", "bs": "bisa", "bsen": "bosan", "bt": "buat", 
    "btw": "ngomong-ngomong", "bubbu": "tidur", "bubu": "tidur",
    "bumil": "ibu hamil", "bw": "bawa", "bwt": "buat", "byk": "banyak", "byrin": "bayarkan",
    "calo": "makelar", "caper": "cari perhatian", "ce": "cewek", "cekal": "cegah tangkal", 
    "cemen": "penakut", "cengengesan": "tertawa", "cepet": "cepat", "cew": "cewek", 
    "chuyunk": "sayang", "cimeng": "ganja", "cipika cipiki": "cium pipi kanan cium pipi kiri",
    "cmiiw": "correct me if i'm wrong", "cmpur": "campur", "bubble wrap" : "plastik",
    "curcol": "curahan hati colongan", "cwek": "cewek"
}

def normalize_slang(text):
    """Normalize slang words to their standard form"""
    if not isinstance(text, str):
        return ""
        
    tokens = text.split()
    normalized_tokens = []

    for token in tokens:
        normalized_token = slangwords.get(token.lower(), token)
        normalized_tokens.append(normalized_token)

    return ' '.join(normalized_tokens)

# Process the dataframe
clean_df = df.copy()

# Apply text preprocessing pipeline
clean_df['text_clean'] = clean_df['Description'].apply(cleaningText)
clean_df['text_casefoldingText'] = clean_df['text_clean'].apply(casefoldingText)
clean_df['text_slangwords'] = clean_df['text_casefoldingText'].apply(normalize_slang)
clean_df['text_tokenizingText'] = clean_df['text_slangwords'].apply(tokenizingText)
clean_df['text_stopword'] = clean_df['text_tokenizingText'].apply(filteringText)
clean_df['text_akhir'] = clean_df['text_stopword'].apply(toSentence)

# --- Load and Cache Lexicons from GitHub ---
@st.cache_data
def load_lexicons():
    """Load lexicon data from GitHub and cache it to improve performance"""
    lexicon_greenhonesty = dict()
    lexicon_greenwashing = dict()
    
    # Load greenhonesty lexicon
    gh_url = 'https://raw.githubusercontent.com/dwsafperri/lexicon_greenwashing_greenhonesty/refs/heads/main/lexicon_positive_greenhonesty.csv'
    gw_url = 'https://raw.githubusercontent.com/dwsafperri/lexicon_greenwashing_greenhonesty/refs/heads/main/lexicon_negative_greenwashing.csv'
    
    try:
        gh_response = requests.get(gh_url)
        if gh_response.status_code == 200:
            reader = csv.reader(StringIO(gh_response.text), delimiter=',')
            for row in reader:
                lexicon_greenhonesty[row[0]] = int(row[1])
        else:
            st.error("Failed to fetch green honesty lexicon data")
    except Exception as e:
        st.error(f"Error loading greenhonesty lexicon: {e}")
        
    try:
        gw_response = requests.get(gw_url)
        if gw_response.status_code == 200:
            reader = csv.reader(StringIO(gw_response.text), delimiter=',')
            for row in reader:
                lexicon_greenwashing[row[0]] = int(row[1])
        else:
            st.error("Failed to fetch greenwashing lexicon data")
    except Exception as e:
        st.error(f"Error loading greenwashing lexicon: {e}")
        
    return lexicon_greenhonesty, lexicon_greenwashing

# Load lexicons
lexicon_greenhonesty, lexicon_greenwashing = load_lexicons()

# --- Lexicon-based Sentiment Analysis ---
def sentiment_analysis_lexicon_indonesia(text_tokens):
    """
    Analyze sentiment of tokenized text using greenhonesty and greenwashing lexicons
    
    Args:
        text_tokens: List of tokens/words
        
    Returns:
        Tuple of (score, polarity)
    """
    score = 0
    
    # Convert to string for easier searching
    if isinstance(text_tokens, list):
        text_str = ' '.join(text_tokens).lower()
    else:
        text_str = str(text_tokens).lower()
    
    # Add scores from both lexicons
    for phrase in lexicon_greenhonesty:
        if phrase in text_str:
            score += lexicon_greenhonesty[phrase]
            
    for phrase in lexicon_greenwashing:
        if phrase in text_str:
            score += lexicon_greenwashing[phrase]
    
    # Determine polarity based on score
    polarity = 'greenhonesty' if score > 0 else 'greenwashing'
    
    return score, polarity

# Apply sentiment analysis to the dataframe
results = clean_df['text_akhir'].apply(sentiment_analysis_lexicon_indonesia)
results = list(zip(*results))
clean_df['polarity_score'] = results[0]
clean_df['polarity'] = results[1]

# Load ML models for prediction
@st.cache_resource
def load_models():
    """Load the trained models and vectorizer for prediction"""
    try:
        tfidf_vectorizer = joblib.load("tfidf_vectorizer (1).pkl")
        svm_model = joblib.load("svm_model (3).pkl")
        return tfidf_vectorizer, svm_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models
tfidf_vectorizer, svm_model = load_models()

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/leaf.png", width=80)
    st.markdown("<h1 style='text-align: center; color: #2E7D32;'>Greenwashing Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Deteksi klaim lingkungan palsu pada produk</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    selected = option_menu(
        menu_title=None,
        options=["📚 Tentang Greenwashing", "🔍 Cek Deskripsi Produk", "🌿 Edukasi & Tips"], 
        icons=["info-circle", "search", "lightbulb"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f0f8f1"},
            "icon": {"color": "green", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "padding": "10px"},
            "nav-link-selected": {"background-color": "#A4E5D9"},
        }
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Author info box
    st.markdown("""
    <div style='background-color: #f0f8f1; 
                border-left: 4px solid #2E7D32; 
                padding: 15px; 
                border-radius: 5px; 
                margin-top: 20px;'>
        <h4 style='margin:0; color: #1E5631;'>Profile</h4>
        <div style='display: flex; align-items: center; margin-top: 10px;'>
            <div style='margin-left: 10px;'>
                <p style='margin:0; font-weight: bold; color:#d97706;'>Dewi Safira Permata Sari</p>
                <p style='margin:0; font-size: 14px;'>50422411</p>
                <p style='margin:0; font-size: 14px;'>Universitas Gunadarma (2025)</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Main Content Area Based on Selected Menu ---
if selected == "📚 Tentang Greenwashing":
    # About Greenwashing Page
    st.markdown("<h1 style='text-align: center;'>🌿 Greenwashing Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 20px; margin-bottom: 30px;'>Kenali klaim lingkungan yang jujur dan yang menipu</p>", unsafe_allow_html=True)
    
    # Introduction card
    st.markdown("""
    <div class="stcard">
        <h2>📌 Apa Itu Greenwashing?</h2>
        <p style='text-align: justify;'>
        <b>Greenwashing</b> adalah sebuah strategi pemasaran di mana perusahaan atau produk sengaja menampilkan citra yang ramah lingkungan untuk menarik simpati konsumen — padahal kenyataannya, tindakan mereka tidak sejalan dengan nilai-nilai keberlanjutan. Praktik ini sering dilakukan untuk mengecoh publik yang semakin sadar akan isu lingkungan dan cenderung memilih produk yang dianggap "hijau".
        </p>
        <p style='text-align: justify;'>
        Istilah ini pertama kali digunakan oleh seorang aktivis lingkungan bernama Jay Westerveld pada tahun 1986. Ia mengkritik hotel yang meminta tamu untuk tidak mengganti handuk demi alasan "menyelamatkan lingkungan", padahal hotel tersebut tidak menunjukkan inisiatif lain dalam keberlanjutan.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Common forms of greenwashing
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="stcard" style='height: 100%;'>
            <h3>Bentuk Umum Greenwashing</h3>
            <ul>
                <li>✅ Menggunakan label seperti "eco-friendly", "organik", atau "alami" tanpa ada bukti atau sertifikasi yang sah</li>
                <li>🌿 Mendesain kemasan berwarna hijau atau menggunakan simbol daun dan bumi untuk menciptakan kesan alami</li>
                <li>🔍 Menyoroti satu aspek kecil yang 'hijau', padahal produksi dan distribusinya tetap mencemari lingkungan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stcard" style='height: 100%;'>
            <h3>Mengapa Greenwashing Berbahaya?</h3>
            <ul>
                <li>❌ Membuat konsumen tidak sadar sedang mendukung perusahaan yang merusak lingkungan</li>
                <li>❌ Menurunkan kepercayaan publik terhadap klaim ramah lingkungan yang sah</li>
                <li>❌ Menghambat perusahaan yang benar-benar menerapkan praktik keberlanjutan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Detection methods
    st.markdown("""
    <div class="stcard">
        <h3>Cara Mendeteksi Greenwashing</h3>
        <p>Agar tidak mudah tertipu, kita bisa melakukan beberapa langkah berikut:</p>
        <ul>
            <li>Telusuri apakah ada <b>sertifikasi resmi</b> dari lembaga kredibel, seperti:
                <ul>
                    <li>✅ <b>Ekolabel Nasional Indonesia</b> yang dikeluarkan oleh Kementerian Lingkungan Hidup dan Kehutanan (KLHK)</li>
                    <li>✅ <b>Sertifikasi PROPER</b> (Program Penilaian Peringkat Kinerja Perusahaan) dari KLHK</li>
                    <li>🌱 Sertifikasi internasional seperti Ecolabel, Energy Star, atau USDA Organic</li>
                </ul>
            </li>
            <li>Baca dengan seksama <b>laporan keberlanjutan</b> atau tanggung jawab sosial perusahaan</li>
            <li>Hindari produk dengan klaim yang terlalu umum seperti "alami", "green", "eco" tanpa penjelasan lebih lanjut</li>
            <li>Cek apakah perusahaan tersebut juga memiliki jejak karbon yang besar, pelanggaran lingkungan, atau tidak terbuka soal praktik produksinya</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Word Cloud Section
    st.markdown("<h2 style='margin-top: 30px;'>Analisis Kata Kunci dalam Deskripsi Produk</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='margin-bottom: 20px;'>
    Word Cloud adalah representasi visual dari kata-kata yang paling sering muncul dalam deskripsi produk. 
    Kata-kata yang lebih besar menunjukkan frekuensi kemunculan yang lebih tinggi, sehingga kita bisa dengan 
    mudah melihat tema atau isu yang sering diangkat dalam deskripsi produk.
    </p>
    """, unsafe_allow_html=True)
    
    # Word clouds in tabs
    tab1, tab2, tab3 = st.tabs(["📊 Semua Deskripsi", "✅ Greenhonesty", "⚠️ Greenwashing"])
    
    with tab1:
        # All descriptions word cloud
        list_words = ' '.join([' '.join(tweet) for tweet in clean_df['text_stopword']])
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white', 
                             colormap='viridis',
                             min_font_size=10,
                             max_font_size=100,
                             contour_width=1, 
                             contour_color='steelblue').generate(list_words)
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        st.markdown("<p style='text-align: center;'><em>Kata-kata umum dari seluruh deskripsi produk</em></p>", unsafe_allow_html=True)
    
    with tab2:
        # Greenhonesty word cloud
        greenhonesty = clean_df[clean_df['polarity'] == 'greenhonesty'][['text_akhir', 'polarity_score', 'polarity', 'text_stopword']]
        greenhonesty = greenhonesty.sort_values(by='polarity_score', ascending=False).reset_index(drop=True)
        greenhonesty.index += 1
        
        list_words = ' '.join([' '.join(tweet) for tweet in greenhonesty['text_stopword']])
        wordcloud_greenhonesty = WordCloud(width=800, height=400, 
                                         background_color='white',
                                         colormap='YlGn',
                                         min_font_size=10,
                                         max_font_size=100,
                                         contour_width=1, 
                                         contour_color='steelblue').generate(list_words)
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.imshow(wordcloud_greenhonesty, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        st.markdown("<p style='text-align: center;'><em>Kata-kata umum dalam deskripsi produk greenhonesty</em></p>", unsafe_allow_html=True)
    
    with tab3:
        # Greenwashing word cloud
        greenwashing = clean_df[clean_df['polarity'] == 'greenwashing'][['text_akhir', 'polarity_score', 'polarity', 'text_stopword']]
        greenwashing = greenwashing.sort_values(by='polarity_score', ascending=True).reset_index(drop=True)
        greenwashing.index += 1
        
        list_words = ' '.join([' '.join(tweet) for tweet in greenwashing['text_stopword']])
        wordcloud_greenwashing = WordCloud(width=800, height=400, 
                                         background_color='white',
                                         colormap='Reds',
                                         min_font_size=10,
                                         max_font_size=100,
                                         contour_width=1, 
                                         contour_color='steelblue').generate(list_words)
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.imshow(wordcloud_greenwashing, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        st.markdown("<p style='text-align: center;'><em>Kata-kata umum dalam deskripsi produk greenwashing</em></p>", unsafe_allow_html=True)
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(clean_df['text_akhir'])
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df = tfidf_df.sum().reset_index(name='jumlah').sort_values('jumlah', ascending=False).head(20)
    
    # Model performance
    st.markdown("<h3 style='margin-top: 30px;'>Performa Model Deteksi</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="stcard">
            <h4>Model SVM</h4>
            <p>Support Vector Machine (SVM) merupakan model machine learning yang digunakan untuk mengklasifikasikan deskripsi produk sebagai greenhonesty atau greenwashing.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # SVM accuracy visualization
        X_vectorized = vectorizer.transform(clean_df['text_akhir'])
        y = clean_df['polarity']
        
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
        svm = SVC(probability=True)
        svm.fit(X_train, y_train)
        y_pred_train = svm.predict(X_train)
        y_pred_test = svm.predict(X_test)
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=['Train', 'Test'], y=[acc_train, acc_test], palette='Greens', ax=ax)
        
        # Add percentage labels
        for i, v in enumerate([acc_train, acc_test]):
            ax.text(i, v + 0.01, f'{v:.1%}', ha='center')
            
        ax.set_ylim(0, 1)
        ax.set_ylabel("Akurasi")
        ax.set_title("Akurasi Model SVM")
        st.pyplot(fig)
      # Second column intentionally left empty for balanced layout
    with col2:
        st.markdown("""
        <div class="stcard">
            <h4>Tentang Model</h4>
            <p>Support Vector Machine (SVM) dipilih sebagai model utama untuk mengklasifikasikan teks deskripsi produk. 
            Model ini efektif dalam mengklasifikasikan data teks dengan dimensi tinggi.</p>
            <br>
            <p>Dengan menggunakan fitur TF-IDF untuk ekstraksi fitur, model SVM mampu memisahkan dua kelas 
            (greenhonesty dan greenwashing) dengan akurasi yang tinggi.</p>
        </div>
        """, unsafe_allow_html=True)

elif selected == "🔍 Cek Deskripsi Produk":
    st.markdown("<h1 style='text-align: center;'>🔍 Analisis Deskripsi Produk</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; margin-bottom: 30px;'>Cek apakah deskripsi produk mengandung klaim greenwashing</p>", unsafe_allow_html=True)
    
    # Product description checker card
    st.markdown("""
    <div class="stcard">
        <h3>📝 Masukkan Deskripsi Produk</h3>
        <p>Tempel deskripsi produk yang ingin kamu analisis di kolom di bawah ini. 
        Sistem akan mengidentifikasi apakah deskripsi tersebut terindikasi greenwashing atau bukan.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # User input area
    user_input = st.text_area("", height=150, placeholder="Contoh: Botol ini terbuat dari bahan ramah lingkungan dan biodegradable...")
    
    col1, col2 = st.columns([1,3])
    with col1:
        check_button = st.button("🔍 Analisis Deskripsi")
    
    with col2:
        st.markdown("")  # For spacing
    
    # Process input when button is clicked
    if check_button and user_input:
        with st.spinner('Memproses deskripsi produk...'):
            # Process the text using the same pipeline as training data
            cleaned = cleaningText(user_input)
            cased = casefoldingText(cleaned)
            slang_normalized = normalize_slang(cased)
            tokenized = tokenizingText(slang_normalized)
            filtered = filteringText(tokenized)
            
            # Lexicon-based analysis
            score, lexicon_result = sentiment_analysis_lexicon_indonesia(filtered)
            
            # Model-based prediction if models are loaded
            model_predictions = {}
            prediction_confidence = {}
            
            if tfidf_vectorizer is not None and svm_model is not None:                # Prepare text for prediction
                text_final = toSentence(filtered)
                X_pred = tfidf_vectorizer.transform([text_final])
                
                # SVM prediction
                svm_pred = svm_model.predict(X_pred)[0]
                svm_proba = svm_model.predict_proba(X_pred)[0]
                model_predictions['SVM'] = svm_pred
                prediction_confidence['SVM'] = max(svm_proba) * 100
          # Display results in a nice layout
        st.markdown("<h3 style='margin-top: 30px;'>Hasil Analisis</h3>", unsafe_allow_html=True)
          # Main content area based on lexicon result
        result_color = "#FFEBEE" if lexicon_result == "greenwashing" else "#E8F5E9"
        result_icon = "⚠️" if lexicon_result == "greenwashing" else "✅"
        result_title = "Greenwashing" if lexicon_result == "greenwashing" else "Greenhonesty"
        result_border = "#F44336" if lexicon_result == "greenwashing" else "#4CAF50"
        result_img = "greenwashing_detected.png" if lexicon_result == "greenwashing" else "greenhonesty_detected.png"        # Create a responsive container with streamlit
        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
        results_container = st.container()
        
        with results_container:
            # Create styled box for the result
            col1, col2, col3 = st.columns([1, 10, 1])
            
            with col2:
                # Result header with styling
                st.markdown(f"""
                <div class="result-container" style="background-color:{result_color}; 
                        padding:20px; 
                        border-radius:10px; 
                        border-left:6px solid {result_border}; 
                        margin-bottom:20px; 
                        width:100%; 
                        box-sizing:border-box;
                        text-align:center;">
                    <h2>{result_icon} Prediksi: {result_title}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Centered image container
                img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
                with img_col2:
                    # Use Streamlit's image display instead of HTML
                    st.image(
                        image=result_img,
                        caption=None,  # No caption needed since we have header
                        width=400
                    )
                
                # Add space between image and text
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                
                # Styled container for text
                st.markdown(f"""
                <div style="background-color: white; 
                           border-radius: 8px; 
                           padding: 15px; 
                           box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                           text-align: center;
                           margin: 10px 0;">
                """, unsafe_allow_html=True)
                
                # Text content with improved responsiveness
                if lexicon_result == "greenwashing":
                    st.markdown("""
                    <p style="font-size:18px; line-height:1.5; margin:0;">
                        Deskripsi ini <strong>terindikasi</strong> mengandung klaim greenwashing. Perhatikan bahwa produk 
                        mungkin tidak seramah lingkungan seperti yang diklaim.
                    </p>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <p style="font-size:18px; line-height:1.5; margin:0;">
                        Deskripsi ini menunjukkan indikasi positif dari klaim ramah lingkungan yang lebih transparan 
                        dan bertanggung jawab (greenhonesty).
                    </p>
                    """, unsafe_allow_html=True)
                    
                # Close the text container
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)
          # Responsive metrics display with better mobile compatibility
        st.markdown("<div class='metrics-container' style='margin:20px 0;'></div>", unsafe_allow_html=True)
        
        # Use equal width columns for a more consistent layout
        col1, col2, col3 = st.columns([1,1,1])
        
        with col1:
            st.markdown(f"""
            <div class="stcard metric-card" style="text-align:center; height:100%; display:flex; flex-direction:column; justify-content:space-between;">
                <h4 style="margin-top:0;">Score Lexicon</h4>
                <p style="font-size:24px; font-weight:bold; margin:10px 0;">{score}</p>
                <p style="font-size:14px; margin-bottom:0;">(Negatif: Greenwashing, Positif: Greenhonesty)</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            if 'SVM' in prediction_confidence:
                st.markdown(f"""
                <div class="stcard metric-card" style="text-align:center; height:100%; display:flex; flex-direction:column; justify-content:space-between;">
                    <h4 style="margin-top:0;">Prediksi SVM</h4>
                    <p style="font-size:24px; font-weight:bold; margin:10px 0;">{model_predictions['SVM']}</p>
                    <p style="font-size:14px; margin-bottom:0;">Confidence: {prediction_confidence['SVM']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="stcard metric-card" style="text-align:center; height:100%; display:flex; flex-direction:column; justify-content:space-between;">
                    <h4 style="margin-top:0;">Prediksi SVM</h4>
                    <p style="font-size:18px; margin:10px 0;">Model tidak tersedia</p>
                    <p style="font-size:14px; margin-bottom:0;">&nbsp;</p>
                </div>
                """, unsafe_allow_html=True)
                
        with col3:
            confidence_level = "Tinggi" if 'SVM' in prediction_confidence and prediction_confidence['SVM'] > 80 else "Sedang"
            confidence_color = "#4CAF50" if confidence_level == "Tinggi" else "#FFC107"
            
            st.markdown(f"""
            <div class="stcard metric-card" style="text-align:center; height:100%; display:flex; flex-direction:column; justify-content:space-between;">
                <h4 style="margin-top:0;">Tingkat Kepercayaan</h4>
                <p style="font-size:24px; font-weight:bold; color:{confidence_color}; margin:10px 0;">{confidence_level}</p>
                <p style="font-size:14px; margin-bottom:0;">Berdasarkan hasil analisis model dan leksikon</p>
            </div>
            """, unsafe_allow_html=True)
                  # Add spacing for better layout
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        
        # Display processed text in a more responsive container
        with st.expander("🔍 Lihat Proses NLP"):
            st.markdown("<h3 style='color:#2E7D32; margin-bottom:15px;'>Langkah-langkah Pemrosesan Teks</h3>", unsafe_allow_html=True)
            
            # Create scrollable text containers with consistent styling
            st.markdown("""
            <style>
            .nlp-text-container {
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 10px;
                max-height: 120px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 14px;
                border-left: 3px solid #4CAF50;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f"<strong>Original Text:</strong>", unsafe_allow_html=True)
            st.markdown(f"<div class='nlp-text-container'>{user_input}</div>", unsafe_allow_html=True)
            
            st.markdown(f"<strong>Cleaned Text:</strong>", unsafe_allow_html=True)
            st.markdown(f"<div class='nlp-text-container'>{cleaned}</div>", unsafe_allow_html=True)
            
            st.markdown(f"<strong>Casefolded Text:</strong>", unsafe_allow_html=True)
            st.markdown(f"<div class='nlp-text-container'>{cased}</div>", unsafe_allow_html=True)
            
            st.markdown(f"<strong>Normalized Slang:</strong>", unsafe_allow_html=True)
            st.markdown(f"<div class='nlp-text-container'>{slang_normalized}</div>", unsafe_allow_html=True)
            
            st.markdown(f"<strong>Tokenized Text:</strong>", unsafe_allow_html=True)
            st.markdown(f"<div class='nlp-text-container'>{tokenized}</div>", unsafe_allow_html=True)
            
            st.markdown(f"<strong>Filtered Text (Stopwords Removed):</strong>", unsafe_allow_html=True)
            st.markdown(f"<div class='nlp-text-container'>{filtered}</div>", unsafe_allow_html=True)
    
    # Display tips/guide if no input
    elif not user_input:
        st.markdown("""
        <div style="background-color:#f5f5f5; padding:20px; border-radius:10px; margin-top:20px; text-align:center;">
            <img src="https://img.icons8.com/color/96/000000/info-popup.png" width="60">
            <h3>Bagaimana cara menggunakan alat ini?</h3>
            <ol style="text-align:left; padding-left:20px;">
                <li>Salin deskripsi produk dari toko online, kemasan produk, atau iklan</li>
                <li>Tempel deskripsi tersebut di kolom input di atas</li>
                <li>Klik tombol "Analisis Deskripsi" untuk mendapatkan hasil</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

elif selected == "🌿 Edukasi & Tips":
    st.markdown("<h1 style='text-align: center;'>🌿 Edukasi & Tips Konsumen Cerdas</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; margin-bottom: 30px;'>Tingkatkan pengetahuan tentang konsumsi berkelanjutan dan hindari jebakan greenwashing</p>", unsafe_allow_html=True)
    
    # Main education section
    st.markdown("""
    <div class="stcard">
        <h2>Mengapa Penting Mengenali Greenwashing?</h2>
        <p style='text-align: justify;'>
        Sebagai konsumen, kita memiliki kekuatan untuk menuntut kejujuran dan keberlanjutan dari brand yang kita dukung. 
        Dengan kemampuan mengenali greenwashing, kita dapat:
        </p>
        <ul>
            <li>Mengalokasikan uang kita untuk mendukung usaha yang benar-benar ramah lingkungan</li>
            <li>Mendorong perusahaan untuk lebih transparan tentang praktik keberlanjutan mereka</li>
            <li>Berkontribusi pada perubahan positif untuk lingkungan</li>
            <li>Menghindari produk yang potensi mencemari lingkungan meski mengklaim ramah lingkungan</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Tips in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="stcard" style='height: 100%;'>
            <h3>🔍 Tips Mendeteksi Greenwashing</h3>
            <ul>
                <li><strong>Cari sertifikasi resmi</strong> dari lembaga kredibel seperti Ecolabel, USDA Organic, atau FSC</li>
                <li><strong>Waspadai klaim umum</strong> seperti "natural", "eco", atau "green" tanpa penjelasan spesifik</li>
                <li><strong>Perhatikan proporsi klaim</strong> - apakah hanya satu aspek kecil yang ramah lingkungan?</li>
                <li><strong>Periksa transparansi perusahaan</strong> tentang rantai pasok dan proses produksi</li>
                <li><strong>Bandingkan dengan standar industri</strong> - apakah klaim tersebut benar-benar istimewa?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="stcard" style='height: 100%;'>
            <h3>🌱 Panduan Konsumsi Berkelanjutan</h3>
            <ul>
                <li><strong>Kurangi, gunakan kembali, daur ulang</strong> - dalam urutan prioritas tersebut</li>
                <li><strong>Beralih ke produk massal/isi ulang</strong> untuk mengurangi sampah kemasan</li>
                <li><strong>Pilih produk lokal</strong> untuk mengurangi jejak karbon dari transportasi</li>
                <li><strong>Investasikan pada kualitas</strong> - produk yang tahan lama meski harga lebih mahal</li>
                <li><strong>Dukung usaha dengan kebijakan pengembalian produk</strong> di akhir masa pakainya</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Case studies
    st.markdown("<h3 style='margin-top: 30px;'>Studi Kasus: Greenwashing dalam Praktik</h3>", unsafe_allow_html=True)
    with st.expander('Kasus 1: Produk Pembersih "Eco-Friendly" dengan Zat Kimia Berbahaya'):
        st.markdown("""
        Sebuah merek pembersih rumah tangga mengklaim produknya "eco-friendly", "green", dan "biodegradable" dengan kemasan berwarna hijau dan gambar daun. Namun, daftar bahan menunjukkan penggunaan:
        
        - Sodium Lauryl Sulfate (SLS) konsentrasi tinggi
        - Pewangi sintetis
        - Pewarna buatan
        - Pengawet seperti methylisothiazolinone
        
        **Analisis:** Meski label menyebutkan "biodegradable", bahan-bahan tersebut dapat menyebabkan iritasi pada organisme akuatik dan beberapa di antaranya tidak mudah terurai. Klaim "eco-friendly" tidak didukung oleh bukti atau sertifikasi pihak ketiga.
        
        **Pelajaran:** Selalu periksa daftar bahan dan carilah sertifikasi resmi seperti Ecolabel atau Green Seal, jangan hanya terpengaruh oleh desain kemasan hijau.
        """)
    with st.expander('Kasus 2: Tas Belanja "Ramah Lingkungan" dari Plastik'):
        st.markdown("""
        Sebuah perusahaan ritel mengkampanyekan penggunaan tas belanja "ramah lingkungan" yang diklaim terbuat dari "bahan daur ulang" dan "mengurangi limbah plastik".
        
        Setelah diteliti lebih lanjut, terungkap bahwa:
        - Tas tersebut terbuat dari plastik non-woven polypropylene
        - Hanya mengandung 10% material daur ulang
        - Tidak dapat terurai secara alami
        - Memerlukan penggunaan berulang minimal 104 kali untuk mengimbangi dampak lingkungan dibanding kantong plastik sekali pakai
        
        **Analisis:** Perusahaan melebih-lebihkan manfaat lingkungan dari produk mereka dan tidak transparan tentang komposisi sebenarnya.
        
        **Pelajaran:** Carilah informasi tentang siklus hidup produk secara keseluruhan dan berapa persen sebenarnya kandungan daur ulangnya.
        """)
        
    # Tips for sustainable living
    st.markdown("<h3 style='margin-top: 30px;'>Langkah Kecil untuk Gaya Hidup Lebih Berkelanjutan</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color:#E8F5E9; padding:15px; border-radius:10px; height:100%;">
            <h4 style="color:#2E7D32; text-align:center;">🏠 Di Rumah</h4>
            <ul>
                <li>Gunakan lampu LED hemat energi</li>
                <li>Kompos limbah organik</li>
                <li>Pilih pembersih multi-fungsi</li>
                <li>Gunakan kain lap daripada tisu</li>
                <li>Matikan peralatan listrik saat tidak digunakan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div style="background-color:#E3F2FD; padding:15px; border-radius:10px; height:100%;">
            <h4 style="color:#1565C0; text-align:center;">🛒 Saat Berbelanja</h4>
            <ul>
                <li>Bawa tas belanja sendiri</li>
                <li>Pilih produk dengan kemasan minimal</li>
                <li>Beli dalam ukuran ekonomis</li>
                <li>Utamakan produk lokal dan musiman</li>
                <li>Cek sertifikasi keberlanjutan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div style="background-color:#FFF3E0; padding:15px; border-radius:10px; height:100%;">
            <h4 style="color:#E65100; text-align:center;">👗 Fashion & Gaya Hidup</h4>
            <ul>
                <li>Pilih pakaian berbahan alami</li>
                <li>Jual atau donasikan barang bekas</li>
                <li>Perbaiki daripada membuang</li>
                <li>Cari produk "pre-loved" atau vintage</li>
                <li>Dukung brand lokal dengan praktik etis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Motivational closing
    st.markdown("""
    <div style="background-color:#f9f9f9; padding:25px; border-radius:10px; text-align:center; margin-top:30px;">
        <h3>Kekuatan Perubahan Ada di Tangan Kita</h3>
        <p style="font-size:18px;">
        Dengan menjadi konsumen yang lebih cermat dan kritis terhadap klaim ramah lingkungan, 
        kita tidak hanya melindungi diri dari manipulasi pemasaran, tetapi juga mendorong 
        industri untuk mengadopsi praktik yang benar-benar berkelanjutan.
        </p>
        <p style="font-style:italic; margin-top:20px;">
        "Bumi tidak diwariskan dari nenek moyang kita, melainkan dipinjam dari anak cucu kita."
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2025 Greenwashing Detector | Dibuat oleh Dewi Safira Permata Sari | Universitas Gunadarma</p>
    </div>
    """, unsafe_allow_html=True)
