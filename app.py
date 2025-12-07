import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import plotly.graph_objects as go
import plotly.express as px

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
    
    button[kind="primary"],
    button[kind="secondary"],
    .stButton > button,
    div[data-testid="stButton"] > button,
    .stDownloadButton > button {
        background-color: #495057 !important;
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
    div[data-testid="stButton"] > button:hover,
    .stDownloadButton > button:hover {
        background-color: #343a40 !important;
        color: #ffffff !important;
    }
    
    button[kind="primary"] p,
    button[kind="secondary"] p,
    .stButton > button p,
    div[data-testid="stButton"] > button p,
    .stDownloadButton > button p {
        color: #ffffff !important;
    }
    
    h1, h2, h3, h4 {
        color: #212529 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #212529 !important;
    }
    
    /* Fix untuk expander - pastikan teks tetap hitam */
    details summary {
        color: #212529 !important;
    }
    
    details[open] summary {
        color: #212529 !important;
        background-color: transparent !important;
    }
    
    details summary:hover {
        color: #212529 !important;
    }
    
    details summary p,
    details summary span,
    details summary div {
        color: #212529 !important;
    }
    
    /* Fix untuk konten di dalam expander */
    details[open] > div {
        color: #212529 !important;
    }
    
    details[open] p, details[open] span, details[open] li {
        color: #212529 !important;
    }
    
    /* Fix untuk teks di dalam summary saat expander dibuka */
    details[open] summary * {
        color: #212529 !important;
    }
    
    /* Fix untuk file uploader text */
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploadDropzone"] label,
    div[data-testid="stFileUploadDropzone"] p,
    div[data-testid="stFileUploadDropzone"] span,
    div[data-testid="stFileUploadDropzone"] small {
        color: #212529 !important;
    }
    
    /* Pastikan dataframe dan tabel tetap readable */
    .dataframe, .dataframe * {
        color: #212529 !important;
    }
    
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e9ecef;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #495057;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
        font-weight: 500;
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

def create_template_csv():
    """Membuat template CSV untuk download"""
    template_data = {
        'Nama': ['Lukas Austin', 'Devin Suryadi', 'Orlando Bloem'],
        'GRE_Score': [300, 310, 315],
        'TOEFL_Score': [95, 105, 110],
        'University_Rating': [2, 3, 4],
        'SOP': [2, 3.5, 3.0],
        'LOR': [4.0, 3.5, 2.5],
        'GPA': [3.65, 3.40, 4],
        'Research': [1, 1, 0]
    }
    return pd.DataFrame(template_data)


def validate_csv(df):
    """Validasi format CSV yang diupload"""
    required_columns = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'GPA', 'Research']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Kolom berikut tidak ditemukan: {', '.join(missing_cols)}"
    
    # Validasi range nilai
    validations = {
        'GRE_Score': (260, 340),
        'TOEFL_Score': (0, 120),
        'University_Rating': (1, 5),
        'SOP': (1, 5),
        'LOR': (1, 5),
        'GPA': (0, 4),
        'Research': (0, 1)
    }
    
    for col, (min_val, max_val) in validations.items():
        if df[col].min() < min_val or df[col].max() > max_val:
            return False, f"Nilai {col} harus berada di antara {min_val} dan {max_val}"
    
    return True, "Valid"

def batch_predict(df):
    """Melakukan prediksi batch untuk semua kandidat"""
    feature_cols = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'GPA', 'Research']
    X = df[feature_cols]
    
    # Rename kolom untuk sesuai dengan model
    X_renamed = X.rename(columns={
        'GRE_Score': 'GRE Score',
        'TOEFL_Score': 'TOEFL Score',
        'University_Rating': 'University Rating'
    })
    
    scaled_data = scaler.transform(X_renamed)
    predictions = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)[:, 1]
    
    return predictions, probabilities

def analyze_batch_results(df, predictions, probabilities):
    """Menganalisis hasil prediksi batch"""
    df_result = df.copy()
    df_result['Prediksi'] = ['Diterima' if p == 1 else 'Ditolak' for p in predictions]
    df_result['Probabilitas_Penerimaan'] = probabilities * 100
    
    # Statistik agregat
    total_candidates = len(df_result)
    accepted = sum(predictions)
    rejected = total_candidates - accepted
    avg_probability = probabilities.mean() * 100
    
    # Kategori probabilitas
    df_result['Kategori'] = pd.cut(
        df_result['Probabilitas_Penerimaan'],
        bins=[0, 40, 60, 80, 100],
        labels=['Rendah (<40%)', 'Menengah (40-60%)', 'Baik (60-80%)', 'Sangat Baik (>80%)']
    )
    
    return df_result, {
        'total': total_candidates,
        'accepted': accepted,
        'rejected': rejected,
        'acceptance_rate': (accepted/total_candidates)*100,
        'avg_probability': avg_probability
    }


def compute_feature_importance(model, feature_names):
    """Ambil feature importance (tree) atau koefisien (linear); kembalikan dalam urutan menurun."""
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coefs = np.array(model.coef_, dtype=float)
        if coefs.ndim > 1:
            coefs = np.mean(np.abs(coefs), axis=0)
        else:
            coefs = np.abs(coefs)
        importances = coefs

    if importances is None or importances.size == 0:
        return None

    # Selaraskan panjang jika model mengembalikan lebih banyak/lebih sedikit
    n = min(len(importances), len(feature_names))
    paired = list(zip(feature_names[:n], importances[:n]))
    paired.sort(key=lambda x: x[1], reverse=True)

    total = sum(v for _, v in paired) or 1.0
    data = [{
        'Fitur': name,
        'Importance': (value / total) * 100
    } for name, value in paired]

    return pd.DataFrame(data)


def plot_feature_importance(importance_df):
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Fitur',
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues',
        text=importance_df['Importance'].map(lambda v: f"{v:.1f}%"),
    )

    fig.update_layout(
        title=dict(
            text="<b>Feature Importance Model</b>",
            x=0.05,
            xanchor="left",
            font=dict(size=14, color='#212529')
        ),
        xaxis_title="Kontribusi (%)",
        yaxis_title="",
        height=420,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        margin=dict(l=80, r=40, t=60, b=60),
        font=dict(color='#212529', family='Segoe UI'),
        coloraxis_showscale=False,
        xaxis=dict(
            showgrid=True, 
            gridcolor='#e9ecef', 
            showline=True, 
            linecolor='#dee2e6', 
            rangemode='tozero',
            tickfont=dict(color='#212529')
        ),
        yaxis=dict(
            showgrid=False, 
            showline=True, 
            linecolor='#dee2e6',
            tickfont=dict(color='#212529')
        )
    )

    fig.update_traces(
        marker_line_color='#0b4870', 
        marker_line_width=1, 
        textposition='outside',
        textfont=dict(color='#212529')
    )

    return fig

def plot_probability_distribution(df_result):
    """Membuat histogram distribusi probabilitas"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df_result['Probabilitas_Penerimaan'],
        nbinsx=20,
        name='Distribusi Probabilitas',
        marker_color='#495057',
        opacity=0.8
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Distribusi Probabilitas Penerimaan</b>",
            x=0.05,
            xanchor="left",
            font=dict(size=14, color='#212529')
        ),
        xaxis_title="Probabilitas Penerimaan (%)",
        yaxis_title="Jumlah Kandidat",
        height=400,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        font=dict(color='#212529', family='Segoe UI'),
        margin=dict(l=80, r=80, t=60, b=60),
        xaxis=dict(
            showgrid=True,
            gridcolor='#e9ecef',
            showline=True,
            linecolor='#dee2e6',
            tickfont=dict(color='#212529')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e9ecef',
            showline=True,
            linecolor='#dee2e6',
            tickfont=dict(color='#212529')
        )
    )
    
    return fig

def plot_acceptance_by_category(df_result):
    """Membuat pie chart acceptance rate"""
    category_counts = df_result['Prediksi'].value_counts()
    
    colors = ['#198754' if lbl == 'Diterima' else '#dc3545' for lbl in category_counts.index]

    fig = go.Figure(data=[go.Pie(
        labels=category_counts.index,
        values=category_counts.values,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo='label+percent',
        textfont=dict(size=12, color='#212529'),
        hovertemplate='<b>%{label}</b><br>Jumlah: %{value}<br>Persentase: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text="<b>Proporsi Prediksi Penerimaan</b>",
            x=0.05,
            xanchor="left",
            font=dict(size=14, color='#212529')
        ),
        height=400,
        paper_bgcolor='white',
        font=dict(color='#212529', family='Segoe UI'),
        margin=dict(l=80, r=80, t=60, b=60),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color='#212529')
        )
    )
    
    return fig

def plot_feature_averages(df_result):
    """Membuat bar chart rata-rata fitur berdasarkan prediksi - dipisah per kategori skala"""
    feature_cols = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'GPA', 'Research']
    
    accepted = df_result[df_result['Prediksi'] == 'Diterima'][feature_cols].mean()
    rejected = df_result[df_result['Prediksi'] == 'Ditolak'][feature_cols].mean()
    
    # Kelompokkan fitur berdasarkan skala
    group1 = ['GRE_Score', 'TOEFL_Score']  # Skala besar
    group2 = ['University_Rating', 'SOP', 'LOR']  # Skala 1-5
    group3 = ['GPA', 'Research']  # Skala kecil
    
    # Chart 1: GRE & TOEFL Score
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        name='Diterima',
        x=group1,
        y=[accepted[col] for col in group1],
        marker_color='#198754',
        text=[f"{accepted[col]:.1f}" for col in group1],
        textposition='outside',
        textfont=dict(color='#212529')
    ))
    fig1.add_trace(go.Bar(
        name='Ditolak',
        x=group1,
        y=[rejected[col] for col in group1],
        marker_color='#dc3545',
        text=[f"{rejected[col]:.1f}" for col in group1],
        textposition='outside',
        textfont=dict(color='#212529')
    ))
    fig1.update_layout(
        title=dict(
            text="<b>Skor Tes Standar</b>",
            x=0.05,
            xanchor="left",
            font=dict(size=13, color='#212529')
        ),
        xaxis_title="",
        yaxis_title="Rata-rata Nilai",
        barmode='group',
        height=350,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(color='#212529', family='Segoe UI'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color='#212529')
        ),
        xaxis=dict(showgrid=False, showline=True, linecolor='#dee2e6', tickfont=dict(color='#212529')),
        yaxis=dict(showgrid=True, gridcolor='#e9ecef', showline=True, linecolor='#dee2e6', tickfont=dict(color='#212529'))
    )
    
    # Chart 2: Rating & Evaluasi (1-5)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name='Diterima',
        x=group2,
        y=[accepted[col] for col in group2],
        marker_color='#198754',
        text=[f"{accepted[col]:.2f}" for col in group2],
        textposition='outside',
        textfont=dict(color='#212529')
    ))
    fig2.add_trace(go.Bar(
        name='Ditolak',
        x=group2,
        y=[rejected[col] for col in group2],
        marker_color='#dc3545',
        text=[f"{rejected[col]:.2f}" for col in group2],
        textposition='outside',
        textfont=dict(color='#212529')
    ))
    fig2.update_layout(
        title=dict(
            text="<b>Rating & Evaluasi Dokumen</b>",
            x=0.05,
            xanchor="left",
            font=dict(size=13, color='#212529')
        ),
        xaxis_title="",
        yaxis_title="Rata-rata Nilai",
        barmode='group',
        height=350,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(color='#212529', family='Segoe UI'),
        yaxis=dict(range=[0, 5.5], showgrid=True, gridcolor='#e9ecef', showline=True, linecolor='#dee2e6', tickfont=dict(color='#212529')),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color='#212529')
        ),
        xaxis=dict(showgrid=False, showline=True, linecolor='#dee2e6', tickfont=dict(color='#212529'))
    )
    
    # Chart 3: GPA & Research
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name='Diterima',
        x=group3,
        y=[accepted[col] for col in group3],
        marker_color='#198754',
        text=[f"{accepted[col]:.2f}" for col in group3],
        textposition='outside',
        textfont=dict(color='#212529')
    ))
    fig3.add_trace(go.Bar(
        name='Ditolak',
        x=group3,
        y=[rejected[col] for col in group3],
        marker_color='#dc3545',
        text=[f"{rejected[col]:.2f}" for col in group3],
        textposition='outside',
        textfont=dict(color='#212529')
    ))
    fig3.update_layout(
        title=dict(
            text="<b>GPA & Pengalaman Riset</b>",
            x=0.05,
            xanchor="left",
            font=dict(size=13, color='#212529')
        ),
        xaxis_title="",
        yaxis_title="Rata-rata Nilai",
        barmode='group',
        height=350,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(color='#212529', family='Segoe UI'),
        yaxis=dict(range=[0, 4.5], showgrid=True, gridcolor='#e9ecef', showline=True, linecolor='#dee2e6', tickfont=dict(color='#212529')),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color='#212529')
        ),
        xaxis=dict(showgrid=False, showline=True, linecolor='#dee2e6', tickfont=dict(color='#212529'))
    )
    
    return fig1, fig2, fig3

# Header
st.markdown("<h1>Sistem Prediksi Penerimaan Pascasarjana</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #6c757d; font-size: 1.05rem; margin-top: -10px;'>Dashboard Analisis Admisi untuk Administrator Institusi Pendidikan</p>", unsafe_allow_html=True)
st.divider()

# Download template
st.markdown("<h3>Download Template CSV</h3>", unsafe_allow_html=True)

template_df = create_template_csv()
csv_template = template_df.to_csv(index=False)

col1, col2 = st.columns([1, 3])
with col1:
    st.download_button(
        label="Download Template",
        data=csv_template,
        file_name="template_kandidat.csv",
        mime="text/csv"
    )

with col2:
    with st.expander("Lihat Format Template & Panduan Pengisian"):
        st.dataframe(
            template_df,
            use_container_width=True,
            hide_index=True
        )
        st.markdown("""
        **Keterangan Kolom:**
        - **Nama**: Nama kandidat (opsional untuk identifikasi)
        - **GRE_Score**: Skor GRE (260-340)
        - **TOEFL_Score**: Skor TOEFL (0-120)
        - **University_Rating**: Rating universitas asal (1-5)
        - **SOP**: Rating Statement of Purpose (1.0-5.0)
        - **LOR**: Rating Letter of Recommendation (1.0-5.0)
        - **GPA**: IPK (0.00-4.00)
        - **Research**: Pengalaman riset (0 = Tidak, 1 = Ya)
        """)

st.markdown("---")

# Upload file
st.markdown("<h3>Upload File CSV</h3>", unsafe_allow_html=True)

# Styling upload box: dropzone gelap dengan teks putih, daftar file tetap teks gelap
st.markdown(
    """
    <style>
    /* Daftar file terunggah (tetap gelap di teks, latar default putih) */
    div[data-testid="stFileUploaderFileName"] { color: #212529 !important; }
    div[data-testid="stFileUploaderFileSize"] { color: #6c757d !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Pilih file CSV dengan data kandidat",
    type=['csv'],
    help="File CSV harus mengikuti format template yang disediakan"
)

if uploaded_file is not None:
    try:
        # Baca CSV
        df_candidates = pd.read_csv(uploaded_file)
        
        st.markdown(
            f"""
            <div style="
                background-color: #d1e7dd;
                padding: 15px 20px;
                border-radius: 6px;
                color: #0f5132;
                font-weight: 600;
                margin: 15px 0;
            ">
                File berhasil diupload: <strong>{uploaded_file.name}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(f"**Jumlah kandidat:** {len(df_candidates)}")
        
        # Validasi
        is_valid, message = validate_csv(df_candidates)
        
        if not is_valid:
            st.markdown(
                f"""
                <div style="
                    background-color: #f8d7da;
                    padding: 15px 20px;
                    border-radius: 6px;
                    color: #842029;
                    font-weight: 600;
                    margin: 15px 0;
                ">
                    Format CSV tidak valid: {message}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.stop()
        
        # Tampilkan preview data
        with st.expander("Preview Data Kandidat"):
            st.dataframe(
                df_candidates,
                use_container_width=True,
                hide_index=True,
                height=420
            )
        
        # Tombol analisis
        if st.button("Mulai Analisis Prediksi", use_container_width=True):
            with st.spinner("Sedang memproses prediksi untuk semua kandidat..."):
                time.sleep(1)
                
                # Prediksi batch - dapatkan probabilitas dan simpan ke session state
                predictions, probabilities = batch_predict(df_candidates)
                st.session_state['probabilities'] = probabilities
                st.session_state['df_candidates'] = df_candidates
                st.session_state['analysis_done'] = True
        
        # Tampilkan hasil jika analisis sudah dilakukan
        if st.session_state.get('analysis_done', False):
            probabilities = st.session_state['probabilities']
            df_candidates = st.session_state['df_candidates']
            
            st.markdown("---")
            st.markdown("<h2>Hasil Analisis</h2>", unsafe_allow_html=True)
            
            # Threshold slider untuk real-time adjustment
            st.markdown("<h4 style='color: #495057; margin: 10px 0;'>Atur Threshold Prediksi</h4>", unsafe_allow_html=True)
            threshold = st.slider(
                "Informasi untuk Probabilitas (semakin tinggi, semakin selektif)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Kandidat dengan probabilitas >= threshold akan diprediksi DITERIMA"
            )
            
            # Terapkan threshold untuk prediksi kustom
            custom_predictions = (probabilities >= threshold).astype(int)
            df_result, stats = analyze_batch_results(df_candidates, custom_predictions, probabilities)
            
            # Tampilkan info threshold aktif
            st.markdown(
                f"<p style='color: #6c757d; font-size: 0.9rem; margin: 5px 0 20px 0;'>Threshold aktif: <strong>{threshold}</strong> <br> Probabilitas >= {threshold*100:.0f}% akan Diterima</p>",
                unsafe_allow_html=True
            )
            
            # Metric cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Kandidat</div>
                    <div class="metric-value">{stats['total']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Direkomendasikan Diterima</div>
                    <div class="metric-value" style="color: #198754;">{stats['accepted']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Tidak Direkomendasikan</div>
                    <div class="metric-value" style="color: #dc3545;">{stats['rejected']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Acceptance Rate</div>
                    <div class="metric-value" style="color: #495057;">{stats['acceptance_rate']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Grafik analisis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h4 style='color: #495057; margin-bottom: 10px;'>Distribusi Probabilitas Penerimaan</h4>", unsafe_allow_html=True)
                fig_dist = plot_probability_distribution(df_result)
                st.plotly_chart(fig_dist, use_container_width=True)
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 12px; border-radius: 4px; margin-top: -10px;">
                    <p style="color: #6c757d; margin: 0; font-size: 0.88rem; line-height: 1.5;">
                        Grafik ini menunjukkan sebaran tingkat kemungkinan diterima dari seluruh kandidat. 
                        Peak (puncak) yang tinggi menunjukkan banyak kandidat dengan probabilitas serupa. 
                        Semakin ke kanan (mendekati 100%), semakin tinggi peluang penerimaan.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("<h4 style='color: #495057; margin-bottom: 10px;'>Proporsi Prediksi Penerimaan</h4>", unsafe_allow_html=True)
                fig_pie = plot_acceptance_by_category(df_result)
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 12px; border-radius: 4px; margin-top: -10px;">
                    <p style="color: #6c757d; margin: 0; font-size: 0.88rem; line-height: 1.5;">
                        Grafik menampilkan perbandingan visual antara kandidat yang direkomendasikan untuk diterima (hijau) 
                        dan yang tidak direkomendasikan (merah). Persentase menunjukkan proporsi dari total kandidat.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Feature comparison
            st.markdown("<h3>Perbandingan Fitur: Kandidat Diterima vs Ditolak</h3>", unsafe_allow_html=True)
            
            st.markdown("""
            <p style="color: #6c757d; margin: 10px 0 20px 0; font-size: 0.95rem; line-height: 1.6;">
                Grafik berikut membandingkan rata-rata nilai dari berbagai parameter antara kandidat yang diterima (hijau) 
                dan ditolak (merah). Perbedaan yang signifikan menunjukkan faktor-faktor yang paling berpengaruh terhadap keputusan penerimaan.
            </p>
            """, unsafe_allow_html=True)
            
            fig1, fig2, fig3 = plot_feature_averages(df_result)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
            with col3:
                st.plotly_chart(fig3, use_container_width=True)

            # Feature importance
            importance_feature_names = [
                'GRE Score',
                'TOEFL Score',
                'University Rating',
                'SOP',
                'LOR',
                'GPA',
                'Research'
            ]

            importance_df = compute_feature_importance(model, importance_feature_names)

            st.markdown("<h3>Feature Importance Model</h3>", unsafe_allow_html=True)
            st.markdown(
                "<p style='color: #6c757d; margin: 10px 0 20px 0; font-size: 0.95rem;'>Semakin tinggi nilai importance, semakin besar pengaruh fitur terhadap prediksi model. Gunakan daftar ini untuk memprioritaskan perbaikan data atau intervensi kandidat.</p>",
                unsafe_allow_html=True
            )

            if importance_df is not None:
                fig_imp = plot_feature_importance(importance_df)
                st.plotly_chart(fig_imp, use_container_width=True)

                top3 = importance_df.head(3)
                st.markdown("<p style='color: #495057; margin: 10px 0 5px 0; font-weight: 600;'>Tiga faktor teratas:</p>", unsafe_allow_html=True)
                for _, row in top3.iterrows():
                    item = f"- {row['Fitur']}: kontribusi {row['Importance']:.1f}%"
                    st.markdown(f"<p style='color: #495057; margin: 2px 0 2px 15px;'>{item}</p>", unsafe_allow_html=True)
            else:
                st.info("Model tidak menyediakan feature importance/koefisien sehingga bagian ini tidak dapat ditampilkan.")
            
            # Kategori probabilitas
            st.markdown("<h3>Distribusi Berdasarkan Kategori Probabilitas</h3>", unsafe_allow_html=True)
            category_summary = df_result['Kategori'].value_counts().reset_index()
            category_summary.columns = ['Kategori', 'Jumlah Kandidat']
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(
                    category_summary,
                    use_container_width=True,
                    hide_index=True
                )
            with col2:
                st.markdown("""
                **Interpretasi:**
                - **Rendah (<40%)**: Perlu perbaikan signifikan
                - **Menengah (40-60%)**: Kompetitif, perlu peningkatan
                - **Baik (60-80%)**: Kandidat kuat
                - **Sangat Baik (>80%)**: Kandidat sangat kompetitif
                """)
            
            # Tabel hasil detail
            st.markdown("<h3>Hasil Detail per Kandidat</h3>", unsafe_allow_html=True)
            
            # Format kolom untuk display
            display_df = df_result.copy()
            if 'Nama' in display_df.columns:
                cols = ['Nama', 'Prediksi', 'Probabilitas_Penerimaan', 'Kategori'] + \
                       ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'GPA', 'Research']
                display_df = display_df[[col for col in cols if col in display_df.columns]]
            
            display_df['Probabilitas_Penerimaan'] = display_df['Probabilitas_Penerimaan'].round(2)
            
            # Fungsi untuk highlight warna berdasarkan prediksi
            def highlight_prediction(row):
                if row['Prediksi'] == 'Diterima':
                    return ['background-color: #d1e7dd']*len(row)
                else:
                    return ['background-color: #f8d7da']*len(row)
            
            st.dataframe(
                display_df.style
                    .apply(highlight_prediction, axis=1)
                    .set_properties(**{
                        'color': '#4A4A4A'
                    }),
                use_container_width=True,
                hide_index=True,
                height=520
            )
            
            # Download hasil
            st.markdown("<h3>Download Hasil Analisis</h3>", unsafe_allow_html=True)
            result_csv = df_result.to_csv(index=False)
            st.download_button(
                label="Download Hasil (CSV)",
                data=result_csv,
                file_name="hasil_prediksi.csv",
                mime="text/csv"
            )
            
            # Analisis faktor
            avg_features = df_result[['GRE_Score', 'TOEFL_Score', 'GPA', 'Research']].mean()
            
            recommendations = []
            if avg_features['GRE_Score'] < 310:
                recommendations.append("- **GRE Score**: Rata-rata kandidat di bawah 310.")
            if avg_features['TOEFL_Score'] < 100:
                recommendations.append("- **TOEFL Score**: Rata-rata kandidat di bawah 100.")
            if avg_features['GPA'] < 3.2:
                recommendations.append("- **GPA**: Rata-rata IPK di bawah 3.2.")
            if avg_features['Research'] < 0.5:
                recommendations.append("- **Research Experience**: Kurang dari 50% kandidat memiliki pengalaman riset.")
            
            if recommendations:
                st.markdown("<h3>Rekomendasi Perbaikan</h3>", unsafe_allow_html=True)
                
                st.markdown("<p style='color: #6c757d; margin: 10px 0; font-size: 0.95rem;'>Area yang perlu mendapat perhatian:</p>", unsafe_allow_html=True)
                
                for rec in recommendations:
                    # Hapus ** dari string
                    clean_rec = rec.replace('**', '')
                    st.markdown(f"<p style='color: #495057; margin: 5px 0 5px 20px; font-size: 0.92rem;'>{clean_rec}</p>", unsafe_allow_html=True)
                
    except Exception as e:
        st.markdown(
            f"""
            <div style="
                background-color: #f8d7da;
                padding: 15px 20px;
                border-radius: 6px;
                color: #842029;
                font-weight: 600;
                margin: 15px 0;
            ">
                Terjadi kesalahan saat memproses file: {str(e)}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div style="
                background-color: #cff4fc;
                padding: 15px 20px;
                border-radius: 6px;
                color: #055160;
                margin: 15px 0;
            ">
                Pastikan format CSV sesuai dengan template yang disediakan.
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    st.markdown(
        """
        <div style="
            background-color: #e9ecef;
            padding: 20px 25px;
            border-radius: 6px;
            color: #495057;
            margin: 20px 0;
            font-size: 1.05rem;
        ">
            <strong>Petunjuk:</strong> Silakan upload file CSV data kandidat untuk memulai analisis prediksi penerimaan.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.8rem; color: #6c757d;'>Decision Support System - 230011, 230045, 230057</p>", unsafe_allow_html=True)