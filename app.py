import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from google import genai

# ==========================================
# 1. KONFIGURASI HALAMAN & VARIABEL UTAMA
# ==========================================
st.set_page_config(
    page_title="AgroScan - Intelligent Plant Disease Advisor",
    page_icon="🌿",
    layout="wide"
)

# ⚠️ GANTI API KEY DI SINI DENGAN API KEY KAMU SENDIRI
GOOGLE_API_KEY = "AIzaSyBlwWxQqr4UHMls8y5OwGSrOMF5Veh6Zwc"

# Daftar Kelas (Hasil Training Model Utama - 38 Kelas PlantVillage)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ==========================================
# 2. SETUP MODEL (CNN & GEMINI)
# ==========================================

# A. Load Model Utama (CNN)


@st.cache_resource
def load_cnn_model():
    try:
        # Memuat model format .keras yang lebih modern dan stabil
        model = tf.keras.models.load_model('model_agroscan.keras')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model CNN: {e}")
        return None


cnn_model = load_cnn_model()

# B. Setup Model Sekunder (Gemini) dengan System Instruction
try:
    client = genai.Client(api_key=GOOGLE_API_KEY)

    system_prompts = """
    Kamu adalah AgroScan, konsultan ahli pertanian dan patologi tanaman.
    Tugasmu memberikan saran penanganan penyakit tanaman kepada petani.
    
    Setiap jawaban WAJIB memuat 4 poin berikut secara LENGKAP:
    1. PENJELASAN SINGKAT (Ciri fisik)
    2. PENYEBAB (Jamur/Virus/Bakteri/Lingkungan)
    3. PENANGANAN (SOLUSI):
       - Cara Organik/Alami
       - Cara Kimiawi (Rekomendasi bahan aktif)
    4. PENCEGAHAN (Tips masa depan)
    
    Jawablah dengan Bahasa Indonesia yang jelas, tegas, dan mudah dipahami.
    JANGAN memotong jawaban. Pastikan poin 4 selalu tertulis sampai selesai.
    """

    api_status = "✅ Terhubung"

except Exception as e:
    api_status = f"❌ Error API: {e}"
    client = None

# ==========================================
# 3. FUNGSI UTILITAS
# ==========================================


def preprocess_image(image):
    # Resize ke 128x128 sesuai training model utama
    image = image.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Tambah batch dimension
    return img_array


def clean_label(label):
    # Membersihkan teks label (contoh: "Tomato___Early_blight" -> "Tomato", "Early Blight")
    parts = label.split("___")
    plant = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ")
    return plant, disease


# ==========================================
# 4. TAMPILAN APLIKASI (UI)
# ==========================================
st.title("🌿 AgroScan: Deteksi & Konsultasi Tanaman")
st.markdown("**Kelompok: Masih Training** | Study Group AI Telkom University")
st.write("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Status Sistem")
    if cnn_model:
        st.success("✅ Model Utama (CNN) Siap")
    else:
        st.error("❌ Model .keras Hilang!")
    st.info(f"🤖 Model Sekunder: {api_status}")

    st.markdown("---")
    st.markdown("**Panduan:**")
    st.markdown("1. Upload foto daun tanaman.")
    st.markdown("2. AI akan mendeteksi penyakit yang ada pada daun.")
    st.markdown("3. AI Consultant akan memberi solusi pengobatan!.")

# Layout 2 Kolom
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📸 Upload Daun!")
    uploaded_file = st.file_uploader(
        "Format: JPG, PNG, JPEG", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Preview Citra", use_column_width=True)

with col2:
    if uploaded_file and cnn_model:
        st.subheader("🔍 Hasil Analisis")

        with st.spinner('Model Utama sedang bekerja...'):
            # 1. Prediksi Model Utama
            processed_img = preprocess_image(image)
            predictions = cnn_model.predict(processed_img)
            score = predictions[0]

            class_idx = np.argmax(score)
            confidence = 100 * np.max(score)
            raw_label = CLASS_NAMES[class_idx]
            plant_name, disease_name = clean_label(raw_label)

        # Tampilkan Hasil CNN
        st.info(f"**Tanaman:** {plant_name}")

        if "healthy" in disease_name.lower():
            st.success(f"**Status:** {disease_name} (Sehat) ✨")
            st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")
            st.balloons()
            st.write(
                "Tanaman sehat! Pertahankan perawatan rutin penyiraman dan pemupukan.")
        else:
            st.error(f"**Terdeteksi Penyakit:** {disease_name} ⚠️")
            st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")

            # 2. Panggil Model Sekunder (Jika Terdeteksi Sakit)
            st.markdown("---")
            st.subheader("🤖 Saran AgroScan Consultant")

            if client:
                with st.spinner('Menghubungi konsultan ahli untuk solusi...'):
                    try:
                        # Kita kirim inputnya saja, instruksi sudah ditanam di system_prompts
                        user_prompt = f"Tanaman: {plant_name}, Penyakit: {disease_name}"

                        response = client.models.generate_content(
                            model="gemini-flash-latest",
                            contents=f"{system_prompts}\n\n{user_prompt}"
                        )

                        # Tampilkan Output
                        st.markdown(response.text)

                    except Exception as e:
                        st.warning(f"Gagal mendapatkan saran: {e}")
            else:
                st.warning("API Key belum diset atau salah.")
