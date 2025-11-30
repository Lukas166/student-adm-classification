import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import plotly.graph_objects as go

st.set_page_config(
    page_title="Sistem Prediksi Penerimaan Pascasarjana",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
        color: #000000;
    }
    
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .admin-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
    }
    
    .stNumberInput label, .stSelectbox label, .stSlider label, .stRadio label, .stTextInput label {
        color: #212529 !important;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    div[data-baseweb="input"] input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #ced4da !important;
    }
    
    div[data-baseweb="input"] {
        background-color: #ffffff !important;
    }
    
    input[type="text"], input[type="number"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #ced4da !important;
    }
    
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    button[kind="primary"],
    button[kind="secondary"],
    .stButton > button,
    div[data-testid="stButton"] > button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover,
    .stButton > button:hover,
    div[data-testid="stButton"] > button:hover {
        background-color: #212121 !important;
        color: #ffffff !important;
    }
    
    button[kind="primary"] p,
    button[kind="secondary"] p,
    .stButton > button p,
    div[data-testid="stButton"] > button p {
        color: #ffffff !important;
    }
    
    h1, h2, h3, h4 {
        color: #212529 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .streamlit-expanderHeader {
        background-color: #f1f3f5 !important;
        color: #212529 !important;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #e9ecef !important;
        color: #212529 !important;
    }
    
    details[open] summary {
        color: #212529 !important;
    }
    
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #212529 !important;
    }
    
    .element-container p {
        color: #212529 !important;
    }
    
    div[data-testid="stExpander"] div[role="button"] p {
        color: #212529 !important;
    }
    
    div[data-testid="stMarkdownContainer"] p {
        color: #212529 !important;
    }
    
    .stAlert p {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open('dss_admission_model.pkl', 'rb'))
        scaler = pickle.load(open('dss_admission_scaler.pkl', 'rb'))
        return model, scaler
    except FileNotFoundError:
        st.error("File model atau scaler tidak ditemukan. Pastikan dss_admission_model.pkl dan dss_admission_scaler.pkl ada di folder yang sama.")
        st.stop()

model, scaler = load_assets()

def get_prediction(input_data):
    scaled_data = scaler.transform(input_data)
    pred = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0]
    return pred, prob

def show_help_info(field_name):
    help_info = {
        "GRE Score": "Graduate Record Examination (260-340). Nilai kompetitif: 315+. Tes standar untuk masuk pascasarjana.",
        "TOEFL Score": "Test of English as a Foreign Language (0-120). Nilai kompetitif: 100+. Mengukur kemampuan bahasa Inggris akademik.",
        "University Rating": "Peringkat universitas S1 Anda. 1=Tidak Dikenal, 3=Rata-rata, 5=Top Tier. Rating tinggi dapat kompensasi GPA rendah.",
        "SOP": "Statement of Purpose (1-5). Esai motivasi studi. 1=Sangat Buruk, 3=Cukup, 5=Outstanding. Harus spesifik dan jelas.",
        "LOR": "Letter of Recommendation (1-5). Surat rekomendasi dari dosen/atasan. 1=Lemah, 3=Standar, 5=Sangat Kuat.",
        "GPA": "Grade Point Average/IPK (0.00-4.00). Nilai kompetitif: 3.20+. Pastikan sudah dalam skala 4.0.",
        "Research": "Pengalaman riset/publikasi jurnal. Faktor yang SANGAT berpengaruh dalam penerimaan pascasarjana."
    }
    
    info = help_info.get(field_name, "")
    if info:
        st.caption(info)

def generate_recommendations(input_data, prediction, prob):
    recs = []
    prob_percent = prob[1] * 100
    
    scaled_vals = scaler.transform(input_data)[0]
    coefs = model.coef_[0]
    contributions = scaled_vals * coefs
    
    df_contrib = pd.DataFrame({
        'Faktor': input_data.columns,
        'Kontribusi': contributions,
        'Nilai_Asli': input_data.values[0],
        'Koefisien': coefs
    }).sort_values('Kontribusi', ascending=True)
    
    negative_factors = df_contrib[df_contrib['Kontribusi'] < 0].nsmallest(3, 'Kontribusi')
    
    for _, row in negative_factors.iterrows():
        faktor = row['Faktor']
        nilai_asli = row['Nilai_Asli']
        kontribusi = row['Kontribusi']
        koef = row['Koefisien']
        
        if koef != 0:
            nilai_scaled_sekarang = scaled_vals[list(input_data.columns).index(faktor)]
            target_scaled = -kontribusi / koef + nilai_scaled_sekarang
            
            mean_val = scaler.mean_[list(input_data.columns).index(faktor)]
            std_val = scaler.scale_[list(input_data.columns).index(faktor)]
            target_nilai_asli = target_scaled * std_val + mean_val
            
            peningkatan = target_nilai_asli - nilai_asli
            
            if faktor == 'GRE Score':
                recs.append(f"GRE Score saat ini {int(nilai_asli)} memberikan kontribusi negatif sebesar {kontribusi:.4f}. Tingkatkan minimal {int(abs(peningkatan))} poin menjadi sekitar {int(target_nilai_asli)} untuk meningkatkan peluang.")
            elif faktor == 'TOEFL Score':
                recs.append(f"TOEFL Score {int(nilai_asli)} di bawah ekspektasi model. Tingkatkan minimal {int(abs(peningkatan))} poin menjadi sekitar {int(target_nilai_asli)} untuk meningkatkan peluang penerimaan.")
            elif faktor == 'GPA':
                recs.append(f"IPK {nilai_asli:.2f} memberikan kontribusi negatif. Target minimal {target_nilai_asli:.2f}. Jika tidak memungkinkan, kompensasi dengan memperkuat faktor lain yang memiliki koefisien tinggi.")
            elif faktor == 'SOP':
                recs.append(f"Kualitas SOP ({nilai_asli:.1f}/5.0) perlu ditingkatkan menjadi minimal {target_nilai_asli:.1f}. Minta review dari mentor atau konsultan pendidikan untuk meningkatkan kejelasan tujuan dan kecocokan program.")
            elif faktor == 'LOR':
                recs.append(f"Kualitas LOR ({nilai_asli:.1f}/5.0) perlu ditingkatkan menjadi minimal {target_nilai_asli:.1f}. Pilih pemberi rekomendasi dengan kredibilitas lebih tinggi yang benar-benar mengenal kualitas akademik Anda.")
            elif faktor == 'University Rating':
                recs.append(f"Rating universitas asal ({int(nilai_asli)}/5) relatif rendah memberikan kontribusi negatif. Kompensasi dengan prestasi akademik yang sangat kuat (publikasi, GPA tinggi, penghargaan).")
            elif faktor == 'Research':
                if nilai_asli == 0:
                    recs.append("Pengalaman riset memberikan kontribusi negatif signifikan karena belum ada. Sangat disarankan untuk: (1) Bergabung dengan proyek penelitian dosen, (2) Publikasi di jurnal/konferensi, atau (3) Magang riset di lab/industri.")
    
    positive_factors = df_contrib[df_contrib['Kontribusi'] > 0].nlargest(2, 'Kontribusi')
    
    if not positive_factors.empty:
        for _, row in positive_factors.iterrows():
            recs.append(f"Kekuatan utama: {row['Faktor']} (nilai: {row['Nilai_Asli']}) memberikan kontribusi positif sebesar {row['Kontribusi']:.4f}. Pertahankan dan tonjolkan ini dalam aplikasi Anda.")
    
    if prob_percent > 80:
        recs.append("Probabilitas penerimaan sangat tinggi (>80%). Fokus pada konsistensi dokumen dan persiapan interview.")
    elif prob_percent > 60:
        recs.append("Probabilitas penerimaan cukup baik (60-80%). Perbaiki faktor-faktor negatif di atas untuk meningkatkan peluang.")
    elif prob_percent > 40:
        recs.append("Probabilitas penerimaan menengah (40-60%). Sangat disarankan memperbaiki minimal 2 faktor dengan kontribusi negatif terbesar.")
    else:
        recs.append("Probabilitas penerimaan rendah (<40%). Pertimbangkan menunda aplikasi dan fokus pada peningkatan kualifikasi secara menyeluruh.")
    
    if len(negative_factors) > 0 and prob_percent < 60:
        faktor_terkuat = df_contrib.nlargest(1, 'Koefisien')['Faktor'].values[0]
        koef_terkuat = df_contrib.nlargest(1, 'Koefisien')['Koefisien'].values[0]
        recs.append(f"Strategi kompensasi: Fokus memaksimalkan {faktor_terkuat} karena memiliki pengaruh terbesar menurut model (koefisien: {koef_terkuat:.4f}).")
    
    return recs

def plot_feature_importance(input_data):
    scaled_vals = scaler.transform(input_data)[0]
    coefs = model.coef_[0]
    
    contributions = scaled_vals * coefs
    df_imp = pd.DataFrame({
        'Faktor': input_data.columns,
        'Kontribusi': contributions,
        'Nilai_Asli': input_data.values[0]
    }).sort_values('Kontribusi', ascending=True)
    
    colors = ['#dc3545' if x < 0 else '#198754' for x in df_imp['Kontribusi']]
    
    fig = go.Figure(go.Bar(
        x=df_imp['Kontribusi'],
        y=df_imp['Faktor'],
        orientation='h',
        marker=dict(color=colors),
        text=df_imp['Kontribusi'].round(3),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Kontribusi: %{x:.4f}<br>Nilai: %{customdata}<extra></extra>',
        customdata=df_imp['Nilai_Asli']
    ))
    
    fig.update_layout(
        title="Analisis Kontribusi Faktor terhadap Keputusan Prediksi",
        xaxis_title="Besaran Pengaruh (Positif = Mendukung, Negatif = Menghambat)",
        yaxis_title="Faktor Akademik",
        height=400,
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor='white',
        font=dict(color='black', size=12)
    )
    
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="gray")
    
    return fig, df_imp

def get_feature_explanation(df_imp):
    explanations = []
    
    top_positive = df_imp[df_imp['Kontribusi'] > 0].nlargest(2, 'Kontribusi')
    top_negative = df_imp[df_imp['Kontribusi'] < 0].nsmallest(2, 'Kontribusi')
    
    if not top_positive.empty:
        for _, row in top_positive.iterrows():
            explanations.append(f"**{row['Faktor']}** (Nilai: {row['Nilai_Asli']}) memberikan kontribusi POSITIF sebesar {row['Kontribusi']:.4f} - ini adalah kekuatan utama profil Anda.")
    
    if not top_negative.empty:
        for _, row in top_negative.iterrows():
            explanations.append(f"**{row['Faktor']}** (Nilai: {row['Nilai_Asli']}) memberikan kontribusi NEGATIF sebesar {row['Kontribusi']:.4f} - ini adalah area yang perlu diperbaiki.")
    
    return explanations

st.markdown("<h1>Sistem Pendukung Keputusan untuk Seleksi Pascasarjana</h1>", unsafe_allow_html=True)
st.markdown("<p>Dashboard untuk analisis probabilitas penerimaan mahasiswa berdasarkan data historis dan machine learning.</p>", unsafe_allow_html=True)
st.divider()

st.markdown("### Data Akademik Kandidat")

candidate_name = st.text_input(
    "Nama Kandidat (Opsional)", 
    value="",
    placeholder="Masukkan nama lengkap kandidat...",
    key="name_input"
)

if candidate_name:
    st.success(f"Analisis untuk: {candidate_name}")

st.markdown("---")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**GRE Score**")
    show_help_info("GRE Score")
    gre = st.number_input(
        "Masukkan nilai GRE", 
        min_value=260, max_value=340, value=310,
        label_visibility="collapsed",
        key="gre_input"
    )
    
    st.markdown("**Rating Universitas Asal**")
    show_help_info("University Rating")
    rating = st.selectbox(
        "Pilih rating universitas", 
        options=[1, 2, 3, 4, 5],
        index=2,
        format_func=lambda x: f"{x} - {'Tidak Dikenal' if x==1 else 'Rendah' if x==2 else 'Rata-rata' if x==3 else 'Baik' if x==4 else 'Top Tier'}",
        label_visibility="collapsed",
        key="rating_input"
    )
    
    st.markdown("**Kualitas SOP (Statement of Purpose)**")
    show_help_info("SOP")
    sop = st.slider(
        "Skala 1-5", 
        1.0, 5.0, 3.0, 0.5,
        format="%0.1f",
        label_visibility="collapsed",
        key="sop_input"
    )
    
with c2:
    st.markdown("**TOEFL Score**")
    show_help_info("TOEFL Score")
    toefl = st.number_input(
        "Masukkan nilai TOEFL", 
        min_value=0, max_value=120, value=100,
        label_visibility="collapsed",
        key="toefl_input"
    )
    
    st.markdown("**IPK / CGPA (Skala 4.0)**")
    show_help_info("GPA")
    gpa = st.number_input(
        "Masukkan IPK", 
        min_value=0.0, max_value=4.0, value=3.20, step=0.01,
        format="%.2f",
        label_visibility="collapsed",
        key="gpa_input"
    )
    
    st.markdown("**Kualitas LOR (Letter of Recommendation)**")
    show_help_info("LOR")
    lor = st.slider(
        "Skala 1-5", 
        1.0, 5.0, 3.0, 0.5,
        format="%0.1f",
        label_visibility="collapsed",
        key="lor_input"
    )

st.markdown("---")
st.markdown("**Pengalaman Riset**")
show_help_info("Research")
research_opt = st.radio(
    "Status pengalaman riset", 
    options=["Tidak Ada", "Ada"],
    horizontal=True,
    label_visibility="collapsed",
    key="research_input"
)
research = 1 if research_opt == "Ada" else 0

predict_btn = st.button("Analisis Probabilitas Penerimaan", use_container_width=True)

input_df = pd.DataFrame({
    'GRE Score': [gre], 'TOEFL Score': [toefl], 'University Rating': [rating],
    'SOP': [sop], 'LOR': [lor], 'GPA': [gpa], 'Research': [research]
})

if predict_btn:
    with st.spinner("Sedang memproses data dan melakukan prediksi..."):
        time.sleep(0.5)
        pred, prob = get_prediction(input_df)
        prob_percent = prob[1] * 100
        
        st.markdown("---")
        st.markdown("### Hasil Analisis Prediksi")
        
        if candidate_name:
            st.markdown(f"<h4>Kandidat: {candidate_name}</h4>", unsafe_allow_html=True)
        
        bg_color = "#d1e7dd" if pred == 1 else "#f8d7da"
        text_color = "#0f5132" if pred == 1 else "#842029"
        status_text = "DIREKOMENDASIKAN DITERIMA" if pred == 1 else "TIDAK DIREKOMENDASIKAN"
        
        st.markdown(f"""
        <div style="background-color: {bg_color}; color: {text_color}; padding: 15px; border-radius: 5px; text-align: center; margin-bottom: 20px; border: 1px solid {text_color};">
            <h2 style="margin:0; color: {text_color} !important;">{status_text}</h2>
            <p style="margin:0; font-weight:bold; color: {text_color};">Probabilitas Penerimaan: {prob_percent:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_gauge, col_summary = st.columns([1, 1])
        
        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_percent,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence Score", 'font': {'color': "black", 'size': 16}},
                number = {'suffix': "%", 'font': {'size': 40, 'color': 'black'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
                    'bar': {'color': "#0d6efd"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': "#ffebee"},
                        {'range': [40, 60], 'color': "#fff9c4"},
                        {'range': [60, 80], 'color': "#e1f5fe"},
                        {'range': [80, 100], 'color': "#e8f5e9"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70}
                }
            ))
            fig_gauge.update_layout(
                height=250,
                margin=dict(l=20,r=20,t=40,b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "black"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col_summary:
            st.markdown("**Ringkasan Data Input:**")
            summary_data = {
                "Kriteria": ["GRE", "TOEFL", "Univ. Rating", "SOP", "LOR", "GPA", "Riset"],
                "Nilai": [gre, toefl, rating, sop, lor, f"{gpa:.2f}", "Ada" if research==1 else "Tidak"]
            }
            st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)
        
        st.markdown("---")
        st.markdown("<h4>Rekomendasi</h4>", unsafe_allow_html=True)
        recs = generate_recommendations(input_df, pred, prob)
        
        for i, rec in enumerate(recs, 1):
            st.markdown(f"**{i}.** {rec}")
        
        st.markdown("---")
        st.markdown("<h4>Analisis Feature Importance</h4>", unsafe_allow_html=True)
        st.markdown("<p>Grafik ini menunjukkan bagaimana setiap faktor berkontribusi terhadap hasil prediksi:</p>", unsafe_allow_html=True)
        
        fig_imp, df_imp = plot_feature_importance(input_df)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        with st.expander("Penjelasan Detail Kontribusi Faktor"):
            explanations = get_feature_explanation(df_imp)
            for exp in explanations:
                st.markdown(exp)
            
            st.markdown("---")
            st.markdown("**Interpretasi:**")
            st.markdown("- **Nilai Positif (Hijau)**: Faktor ini meningkatkan peluang penerimaan")
            st.markdown("- **Nilai Negatif (Merah)**: Faktor ini menurunkan peluang penerimaan")
            st.markdown("- **Besaran Nilai**: Semakin besar angkanya (positif/negatif), semakin kuat pengaruhnya")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.8rem; color: #6c757d;'>Decision Support System - 230011, 230045, 230057</p>", unsafe_allow_html=True)