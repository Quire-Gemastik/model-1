from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

load_model = tf.keras.models.load_model
# from tensorflow.keras.models import load_model

model = load_model('model/content/model.h5', compile=False)
label_encoder = joblib.load('model/content/label_encoder.pkl')
tfidf_vectorizer = joblib.load('model/content/tfidf_vectorizer.pkl')
keywords = joblib.load('model/content/keywords.pkl')
feature_vec = joblib.load('model/content/feature_vec.pkl')
labels = joblib.load('model/content/labels.pkl')


job_descriptions = {
    "Software Developer": "Mengembangkan aplikasi canggih yang menjadi tulang punggung bisnis modern. Anda akan menulis kode yang elegan dan efisien, memecahkan masalah yang kompleks, dan berkolaborasi dengan tim lintas fungsi untuk menciptakan solusi perangkat lunak yang inovatif dan user-friendly.",
    "Frontend Developer": "Menghidupkan desain menjadi pengalaman pengguna yang memukau dan interaktif. Anda akan bekerja dengan HTML, CSS, dan JavaScript untuk membangun antarmuka yang responsif dan dinamis, serta mengoptimalkan kinerja aplikasi web untuk berbagai perangkat.",
    "Backend Developer": "Merancang dan mengimplementasikan logika bisnis yang kuat di balik layar aplikasi web. Anda akan menggunakan bahasa pemrograman seperti Python, Java, atau Node.js untuk memastikan keamanan, keandalan, dan skala dari sistem server, database, dan API.",
    "Data Scientist": "Menggali wawasan dari tumpukan data yang kompleks. Anda akan menggunakan algoritma machine learning dan teknik statistik untuk menemukan pola tersembunyi, memprediksi tren masa depan, dan mengembangkan model prediktif yang mendukung keputusan strategis bisnis.",
    "Data Analyst": "Mengubah data mentah menjadi laporan yang bermakna dan actionable insights. Anda akan menggunakan alat seperti Excel, SQL, dan visualisasi data untuk menganalisis data bisnis, mengidentifikasi peluang peningkatan, dan mendukung strategi perusahaan.",
    "Machine Learning Engineer": "Membangun sistem AI yang dapat belajar dan berkembang dari data. Anda akan mengembangkan, melatih, dan mengoptimalkan model machine learning yang dapat diterapkan pada berbagai aplikasi, mulai dari rekomendasi produk hingga deteksi anomali.",
    "Cyber Security": "Menjadi penjaga gawang digital perusahaan. Anda akan mengidentifikasi dan menanggulangi ancaman siber, mengembangkan strategi keamanan, dan memastikan bahwa data dan sistem perusahaan terlindungi dari serangan berbahaya.",
    "UI/UX Designer": "Merancang pengalaman pengguna yang tidak hanya fungsional tetapi juga menyenangkan. Anda akan membuat wireframe, prototipe, dan desain akhir yang intuitif dan estetis, serta bekerja erat dengan tim pengembang untuk merealisasikan visi Anda.",
    "Quality Assurance": "Menjadi garda terdepan dalam memastikan kualitas produk perangkat lunak. Anda akan merancang dan menjalankan berbagai pengujian untuk mendeteksi bug, bekerja sama dengan tim pengembang untuk memperbaikinya, dan memastikan bahwa produk akhir memenuhi standar tertinggi.",
    "Game Developer": "Menciptakan dunia virtual yang memikat dan menghibur. Anda akan merancang mekanika permainan, grafik, dan cerita yang menarik, serta bekerja dengan tim untuk mengembangkan game yang mengasyikkan dan imersif untuk berbagai platform.",
    "Data Engineer": "Membangun infrastruktur data yang menjadi tulang punggung analisis dan pemrosesan data. Anda akan merancang dan mengelola pipeline data, memastikan integritas dan kualitas data, serta bekerja dengan teknologi big data untuk mendukung kebutuhan analitik perusahaan.",
    "Cloud Architect": "Merancang solusi cloud yang handal dan skalabel untuk perusahaan. Anda akan menggunakan platform seperti AWS, Azure, atau Google Cloud untuk mengimplementasikan arsitektur yang mendukung operasi bisnis secara efisien dan aman.",
    "IT Support": "Menyediakan solusi cepat dan efektif untuk masalah teknis pengguna. Anda akan menangani instalasi perangkat lunak dan perangkat keras, menyelesaikan masalah jaringan, dan memberikan dukungan teknis untuk memastikan kelancaran operasional sehari-hari.",
    "Business Analyst": "Menjembatani kebutuhan bisnis dengan solusi teknologi. Anda akan menganalisis proses bisnis, mengidentifikasi area peningkatan, dan bekerja sama dengan tim untuk mengembangkan solusi yang mengoptimalkan efisiensi dan kinerja perusahaan.",
    "Digital Marketing": "Menggunakan kreativitas dan data untuk meningkatkan visibilitas dan keterlibatan merek di dunia digital. Anda akan merancang kampanye pemasaran yang memanfaatkan SEO, SEM, media sosial, dan konten kreatif untuk menjangkau dan menarik audiens target secara efektif.",
    "Marketing Specialist": "Membangun dan mengelola kampanye pemasaran yang efektif untuk memperkenalkan produk atau layanan ke pasar. Anda akan melakukan riset pasar, mengidentifikasi peluang, dan mengembangkan strategi pemasaran yang inovatif untuk meningkatkan brand awareness dan penjualan.",
    "Human Resources Manager": "Menjaga kesejahteraan dan pengembangan karier karyawan. Anda akan mengelola rekrutmen, pelatihan, dan kepatuhan hukum, serta mengembangkan kebijakan HR yang mendukung tujuan organisasi dan budaya kerja yang positif.",
    "Project Manager": "Menggerakkan proyek dari awal hingga selesai dengan efisiensi dan keahlian. Anda akan merencanakan, mengoordinasikan, dan mengawasi proyek, memastikan semua aspek berjalan sesuai rencana, tepat waktu, dan sesuai anggaran, serta berkomunikasi dengan tim dan pemangku kepentingan.",
    "Product Manager": "Menjadi penghubung antara visi produk dan eksekusi teknis. Anda akan mengidentifikasi kebutuhan pasar, merancang fitur produk, dan bekerja sama dengan tim pengembang untuk menciptakan produk yang memuaskan kebutuhan pelanggan dan memajukan bisnis.",
    "Finance Analyst": "Menggali data keuangan untuk memberikan wawasan yang berharga. Anda akan menganalisis laporan keuangan, memproyeksikan tren masa depan, dan memberikan rekomendasi strategis untuk meningkatkan kinerja keuangan dan mendukung keputusan bisnis.",
    "Accountant": "Menjaga keseimbangan keuangan perusahaan dengan akurasi dan integritas. Anda akan mengelola buku besar, mempersiapkan laporan keuangan, dan memastikan kepatuhan terhadap standar akuntansi serta peraturan perpajakan.",
    "Copywriter": "Menggunakan kata-kata untuk mempengaruhi dan menginspirasi audiens. Anda akan menulis konten yang menarik dan persuasif untuk berbagai media, termasuk iklan, situs web, dan materi pemasaran, dengan tujuan meningkatkan brand engagement dan konversi.",
    "Legal Counsel": "Memberikan panduan hukum yang strategis dan praktis untuk melindungi kepentingan perusahaan. Anda akan meninjau dan menyusun kontrak, menangani isu hukum, dan memastikan kepatuhan terhadap peraturan yang berlaku untuk mengurangi risiko hukum.",
    "Public Relation Officer": "Mengelola komunikasi publik dan media untuk membangun dan mempertahankan citra positif perusahaan. Anda akan menyusun siaran pers, mengoordinasikan acara publik, dan menjalin hubungan baik dengan media untuk meningkatkan reputasi perusahaan.",
    "Social Media Specialist": "Menciptakan konten yang menarik dan mengelola kehadiran merek di platform media sosial. Anda akan merancang strategi media sosial, berinteraksi dengan pengikut, dan menganalisis kinerja konten untuk meningkatkan visibilitas dan engagement merek.",
    "Video Editor": "Mengubah rekaman mentah menjadi cerita visual yang menarik dan profesional. Anda akan menggunakan perangkat lunak pengeditan video untuk memotong, menyusun, dan memperbaiki rekaman, serta menambahkan efek dan musik untuk menciptakan konten yang memukau."
}


app = FastAPI()

class UserInput(BaseModel):
    text: str

def modelFunction(user_input, tfidf_vectorizer):
    user_tfidf = tfidf_vectorizer.transform([user_input])

    predicted_probabilities = model.predict(user_tfidf.toarray())
    predicted_label = np.argmax(predicted_probabilities, axis=1)[0]
    predicted_label_string = label_encoder.inverse_transform([predicted_label])[0]

    input_keywords = set(word.lower() for word in user_input.split() if word.lower() in keywords.get(predicted_label_string, []))

    user_similarity = cosine_similarity(user_tfidf, feature_vec)[0]

    keyword_indices = [tfidf_vectorizer.vocabulary_.get(word) for word in input_keywords if word in tfidf_vectorizer.vocabulary_]

    if keyword_indices:
        keyword_mask = feature_vec[:, keyword_indices].sum(axis=1).A1 > 0
        user_similarity[keyword_mask] *= 2.0

    index_similar_texts = user_similarity.argsort()[::-1]

    unique_labels = set()
    recommended_labels = []
    similarities = []

    for idx in index_similar_texts:
        if len(recommended_labels) >= 5:
            break

        label = labels[idx]
        label_string = label_encoder.inverse_transform([label])[0]
        if label_string not in unique_labels:
            unique_labels.add(label_string)
            recommended_labels.append(label_string)
            similarities.append(user_similarity[idx])

    if predicted_label_string in recommended_labels:
        index_to_remove = recommended_labels.index(predicted_label_string)
        del recommended_labels[index_to_remove]
        del similarities[index_to_remove]

    return predicted_label, predicted_label_string, recommended_labels[:5], similarities[:5]

@app.post("/predict/")
async def predict(input: UserInput):
    try:
        user_input = input.text
        predicted_label, predicted_label_string, recommended_labels, similarities = modelFunction(user_input, tfidf_vectorizer)

        job_description = job_descriptions.get(predicted_label_string, "Deskripsi tidak ditemukan")

        response = {
            "predicted_label": predicted_label_string,
            "job_description": job_description,
            "recommendations": [
                {"label": label, "similarity": similarity}
                for label, similarity in zip(recommended_labels, similarities)
            ]
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"Hello": "World"}