import pandas as pd
import streamlit as st
import joblib
import os
import numpy as np
from datetime import datetime, timedelta, time
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time as pytime
import base64
import textwrap
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi Harga Tiket Pesawat", page_icon="✈️", layout="wide")

# --- Fungsi Helper ---
def img_to_base64(path):
    if not os.path.exists(path):
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

AIRLINE_LOGOS = {
    "Garuda Indonesia": "assets/garuda.png",
    "Lion Air": "assets/lion.png",
    "AirAsia": "assets/airasia.png",
    "Citilink": "assets/citilink.png",
    "Batik Air": "assets/batik.png",
    "Sriwijaya Air": "assets/sriwijaya.png"
}

FLIGHT_DURATIONS = {
    ('Jakarta', 'Surabaya'): 1.5, ('Jakarta', 'Medan'): 2.2, ('Jakarta', 'Makassar'): 2.5,
    ('Jakarta', 'Denpasar'): 1.8, ('Jakarta', 'Yogyakarta'): 1.2, ('Jakarta', 'Balikpapan'): 2.0,
    ('Surabaya', 'Medan'): 3.5, ('Surabaya', 'Makassar'): 1.5, ('Surabaya', 'Denpasar'): 1.0,
    ('Surabaya', 'Yogyakarta'): 1.0, ('Surabaya', 'Balikpapan'): 1.5, ('Medan', 'Makassar'): 4.0,
    ('Medan', 'Denpasar'): 4.2, ('Medan', 'Yogyakarta'): 3.8, ('Medan', 'Balikpapan'): 3.5,
    ('Makassar', 'Denpasar'): 1.2, ('Makassar', 'Yogyakarta'): 2.0, ('Makassar', 'Balikpapan'): 1.0,
    ('Denpasar', 'Yogyakarta'): 1.2, ('Denpasar', 'Balikpapan'): 1.5, ('Yogyakarta', 'Balikpapan'): 2.2,
}

def get_duration(source, destination):
    if (source, destination) in FLIGHT_DURATIONS:
        return FLIGHT_DURATIONS[(source, destination)]
    elif (destination, source) in FLIGHT_DURATIONS:
        return FLIGHT_DURATIONS[(destination, source)]
    else:
        return 2.0

# --- 1. GENERATE DATA ---
def generate_flight_data(file_path):
    """Membuat dataset dengan fitur yang lebih lengkap"""
    with st.spinner("Generating flight data..."):
        NUM_ROWS = 3000
        AIRLINES = ['Garuda Indonesia', 'Lion Air', 'Citilink', 'Batik Air', 'AirAsia', 'Sriwijaya Air']
        CITIES = ['Jakarta', 'Surabaya', 'Medan', 'Makassar', 'Denpasar', 'Yogyakarta', 'Balikpapan']
        CLASSES = ['Economy', 'Business']

        booking_dates = [datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 730)) for _ in range(NUM_ROWS)]
        days_to_departure_list = np.random.randint(1, 90, size=NUM_ROWS)
        departure_dates = [book_date + timedelta(days=int(dtd)) for book_date, dtd in zip(booking_dates, days_to_departure_list)]

        airlines = np.random.choice(AIRLINES, size=NUM_ROWS)
        flight_classes = np.random.choice(CLASSES, size=NUM_ROWS, p=[0.80, 0.20])
        
        sources, destinations = [], []
        for _ in range(NUM_ROWS):
            pair = np.random.choice(CITIES, size=2, replace=False)
            sources.append(pair[0])
            destinations.append(pair[1])
        
        durations = [get_duration(src, dst) + np.random.uniform(-0.2, 0.2) for src, dst in zip(sources, destinations)]
        durations = [round(max(0.5, d), 1) for d in durations]

        # Logika harga realistis
        base_prices = np.random.randint(500, 1500, size=NUM_ROWS) 
        price_modifier = (90 - days_to_departure_list) / 90 
        prices = base_prices + (base_prices * price_modifier * np.random.uniform(0.2, 0.8))
        prices = np.where(flight_classes == 'Business', prices * np.random.uniform(1.8, 2.5), prices)
        
        # Tambahkan noise realistis
        prices = prices + np.random.normal(0, 100, NUM_ROWS)
        prices = np.maximum(prices, 400)  # Minimum price
        
        # Tambahkan beberapa outlier (5%)
        outlier_indices = np.random.choice(NUM_ROWS, size=int(NUM_ROWS * 0.05), replace=False)
        prices[outlier_indices] = prices[outlier_indices] * np.random.uniform(2, 3, len(outlier_indices))
        
        df = pd.DataFrame({
            'Airline': airlines, 
            'Booking_Date': booking_dates, 
            'Departure_Date': departure_dates,
            'Days_to_Departure': days_to_departure_list, 
            'Source': sources,
            'Destination': destinations, 
            'Duration': durations, 
            'Class': flight_classes, 
            'Price': prices
        })
        
        # Tambahkan missing values secara random (2%)
        for col in ['Duration', 'Days_to_Departure']:
            missing_idx = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
            df.loc[missing_idx, col] = np.nan
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
    st.success(f"✅ Dataset berhasil dibuat: {NUM_ROWS} rows")
    return df

# --- 2. DATA CLEANING ---
def clean_data(df):
    """Membersihkan data: handle missing values, duplicates, outliers"""
    st.subheader("🧹 Data Cleaning")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Awal", len(df))
    
    # Handle missing values
    missing_before = df.isnull().sum().sum()
    df['Duration'].fillna(df['Duration'].median(), inplace=True)
    df['Days_to_Departure'].fillna(df['Days_to_Departure'].median(), inplace=True)
    missing_after = df.isnull().sum().sum()
    
    with col2:
        st.metric("Missing Values", f"{missing_before} → {missing_after}")
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    
    # Handle outliers menggunakan IQR
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_count = len(df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)])
    df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]
    
    with col3:
        st.metric("Outliers Removed", outliers_count)
    
    st.success(f"✅ Data bersih: {len(df)} rows (dikurangi {duplicates} duplikat)")
    
    return df

# --- 3. DATA AUGMENTATION ---
def augment_data(df):
    """Augmentasi data dengan menambah variasi"""
    st.subheader("🔄 Data Augmentation")
    
    original_size = len(df)
    augmented_rows = []
    
    # Strategi 1: Tambahkan noise pada harga (±5%)
    for _ in range(int(original_size * 0.3)):
        row = df.sample(1).copy()
        row['Price'] = row['Price'] * np.random.uniform(0.95, 1.05)
        row['Days_to_Departure'] = row['Days_to_Departure'] + np.random.randint(-3, 4)
        row['Days_to_Departure'] = np.clip(row['Days_to_Departure'], 1, 90)
        augmented_rows.append(row)
    
    # Strategi 2: Swap source-destination untuk rute yang sama
    for _ in range(int(original_size * 0.2)):
        row = df.sample(1).copy()
        row['Source'], row['Destination'] = row['Destination'].values[0], row['Source'].values[0]
        augmented_rows.append(row)
    
    df_augmented = pd.concat([df] + augmented_rows, ignore_index=True)
    
    st.info(f"📊 Data ditambahkan: {original_size} → {len(df_augmented)} rows (+{len(df_augmented) - original_size})")
    
    return df_augmented

# --- 4. PREPROCESSING & NORMALIZATION ---
def preprocess_data(df):
    """Preprocessing: encoding, feature engineering, normalization"""
    st.subheader("⚙️ Data Processing & Normalization")
    
    df = df.copy()
    
    # Feature Engineering
    df['Departure_Date'] = pd.to_datetime(df['Departure_Date'])
    df['Month'] = df['Departure_Date'].dt.month
    df['DayOfWeek'] = df['Departure_Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Encoding categorical features
    categorical_cols = ["Airline", "Source", "Destination", "Class"]
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    # Feature selection
    feature_cols = ["Airline", "Source", "Destination", "Duration", "Class", 
                   "Days_to_Departure", "Month", "DayOfWeek", "IsWeekend"]
    X = df[feature_cols]
    y = df["Price"]
    
    # Normalization menggunakan StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    st.success(f"✅ Features: {len(feature_cols)} | Target: Price")
    
    # Tampilkan statistik
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Feature Statistics (After Scaling)**")
        st.dataframe(X_scaled.describe().round(2), height=200)
    with col2:
        st.write("**Price Distribution**")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(y, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Price')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    return X_scaled, y, encoders, scaler, feature_cols

# --- 5. TRAIN-TEST SPLIT ---
def split_data(X, y):
    """Membagi data menjadi training dan testing (80:20)"""
    st.subheader("📊 Data Splitting")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Set", f"{len(X_train)} rows")
    with col2:
        st.metric("Testing Set", f"{len(X_test)} rows")
    with col3:
        st.metric("Split Ratio", "80:20")
    
    return X_train, X_test, y_train, y_test

# --- 6. MODEL TRAINING & COMPARISON ---
def train_and_compare_models(X_train, X_test, y_train, y_test):
    """Melatih 3 model dan membandingkan performanya"""
    st.subheader("🤖 Model Training & Comparison")
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    progress_bar = st.progress(0)
    for idx, (name, model) in enumerate(models.items()):
        with st.spinner(f"Training {name}..."):
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            results[name] = {
                'model': model,
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'predictions': y_pred_test
            }
        
        progress_bar.progress((idx + 1) / len(models))
    
    # Display comparison table
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Train MAE': [results[m]['train_mae'] for m in results],
        'Test MAE': [results[m]['test_mae'] for m in results],
        'Train RMSE': [results[m]['train_rmse'] for m in results],
        'Test RMSE': [results[m]['test_rmse'] for m in results],
        'Train R²': [results[m]['train_r2'] for m in results],
        'Test R²': [results[m]['test_r2'] for m in results]
    })
    
    st.write("**📈 Performance Comparison**")
    st.dataframe(comparison_df.style.highlight_max(subset=['Train R²', 'Test R²'], color='lightgreen')
                                    .highlight_min(subset=['Test MAE', 'Test RMSE'], color='lightgreen')
                                    .format(precision=2))
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ['test_mae', 'test_rmse', 'test_r2']
    titles = ['Mean Absolute Error (Lower is Better)', 'Root Mean Squared Error (Lower is Better)', 'R² Score (Higher is Better)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        values = [results[m][metric] for m in results]
        axes[idx].bar(results.keys(), values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[idx].set_title(title, fontsize=10)
        axes[idx].set_ylabel(metric.upper())
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Best model selection
    best_model_name = min(results, key=lambda x: results[x]['test_mae'])
    st.success(f"🏆 **Best Model: {best_model_name}** (Lowest Test MAE: {results[best_model_name]['test_mae']:.2f})")
    
    return results, best_model_name

# --- SAVE MODELS ---
def save_artifacts(results, best_model_name, encoders, scaler, feature_cols, model_dir='model'):
    """Menyimpan semua model dan artifacts"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save best model
    best_model = results[best_model_name]['model']
    joblib.dump(best_model, f'{model_dir}/best_model.joblib')
    joblib.dump(encoders, f'{model_dir}/encoders.joblib')
    joblib.dump(scaler, f'{model_dir}/scaler.joblib')
    
    # Save all models
    for name, data in results.items():
        safe_name = name.replace(' ', '_').lower()
        joblib.dump(data['model'], f'{model_dir}/{safe_name}_model.joblib')
    
    # Save metadata
    metadata = {
        'best_model': best_model_name,
        'feature_cols': feature_cols,
        'models': list(results.keys())
    }
    joblib.dump(metadata, f'{model_dir}/metadata.joblib')
    
    st.success(f"✅ All models saved in '{model_dir}/' directory")

# --- LOAD MODELS ---
@st.cache_resource
def load_artifacts(model_dir='model'):
    """Load model dan artifacts"""
    best_model = joblib.load(f'{model_dir}/best_model.joblib')
    encoders = joblib.load(f'{model_dir}/encoders.joblib')
    scaler = joblib.load(f'{model_dir}/scaler.joblib')
    metadata = joblib.load(f'{model_dir}/metadata.joblib')
    return best_model, encoders, scaler, metadata

# --- PREDICTION APP ---
def run_prediction_app():
    """Aplikasi prediksi dengan UI E-Ticket style"""
    try:
        model, encoders, scaler, metadata = load_artifacts()
        
        CITY_DATA = {
            'Jakarta': {'code': 'CGK', 'name': 'Soekarno Hatta Int.'},
            'Surabaya': {'code': 'SUB', 'name': 'Juanda Int.'},
            'Medan': {'code': 'KNO', 'name': 'Kualanamu Int.'},
            'Makassar': {'code': 'UPG', 'name': 'Sultan Hasanuddin Int.'},
            'Denpasar': {'code': 'DPS', 'name': 'Ngurah Rai Int.'},
            'Yogyakarta': {'code': 'YIA', 'name': 'Yogyakarta Int.'},
            'Balikpapan': {'code': 'BPN', 'name': 'Sepinggan Int.'}
        }
        
        st.title("✈️ Prediksi Harga Tiket Pesawat")
        st.write("Masukkan detail pemesanan dan penerbangan untuk mendapatkan estimasi harga.")
        
        # Input Form
        st.subheader("Detail Perjalanan")
        trip_type = st.radio("Tipe Perjalanan", ["Sekali Jalan", "Pulang-Pergi"], horizontal=True)
        
        col1, col2 = st.columns(2)
        with col1:
            booking_date = st.date_input("Tanggal Pemesanan", value=datetime.now())
        with col2:
            departure_date = st.date_input("Tanggal Keberangkatan", value=datetime.now() + timedelta(days=7))
        
        st.subheader("Detail Penerbangan")
        airline = st.selectbox("Pilih Maskapai", sorted(encoders["Airline"].classes_))
        flight_class = st.selectbox("Pilih Kelas Penerbangan", ["Economy Class", "Business Class"])
        source = st.selectbox("Pilih Kota Asal", sorted(encoders["Source"].classes_))
        destination = st.selectbox("Pilih Kota Tujuan", sorted(encoders["Destination"].classes_))
        
        duration = get_duration(source, destination)
        st.info(f"Estimasi durasi penerbangan untuk rute ini adalah **{duration} jam**.")
        
        st.subheader("Detail Penumpang")
        passengers = st.number_input("Jumlah Penumpang", min_value=1, max_value=10, value=1, step=1)
        
        passenger_details = []
        if passengers > 0:
            for i in range(passengers):
                cols = st.columns([3, 2])
                with cols[0]:
                    name = st.text_input(f"Nama Penumpang {i + 1}", key=f"pax_name_{i}")
                with cols[1]:
                    gender = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"], key=f"pax_gender_{i}", horizontal=True, label_visibility="collapsed")
                passenger_details.append({'name': name, 'gender': gender})
        
        if st.button("Prediksi Harga", type="primary", use_container_width=True):
            days_to_departure = (departure_date - booking_date).days
            
            if source == destination:
                st.warning("Kota Asal dan Kota Tujuan tidak boleh sama.")
            elif days_to_departure < 0:
                st.error("Tanggal Keberangkatan tidak boleh sebelum Tanggal Pemesanan.")
            else:
                # Prepare features
                airline_enc = encoders["Airline"].transform([airline])[0]
                source_enc = encoders["Source"].transform([source])[0]
                destination_enc = encoders["Destination"].transform([destination])[0]
                class_raw = flight_class.split(' ')[0]
                class_enc = encoders["Class"].transform([class_raw])[0]
                month = departure_date.month
                day_of_week = departure_date.weekday()
                is_weekend = 1 if day_of_week >= 5 else 0
                
                sample = pd.DataFrame({
                    "Airline": [airline_enc], 
                    "Source": [source_enc], 
                    "Destination": [destination_enc],
                    "Duration": [duration], 
                    "Class": [class_enc], 
                    "Days_to_Departure": [days_to_departure],
                    "Month": [month],
                    "DayOfWeek": [day_of_week],
                    "IsWeekend": [is_weekend]
                })
                
                # Scale features
                sample_scaled = scaler.transform(sample)
                
                # Predict
                single_ticket_price = model.predict(sample_scaled)[0] * 1000
                round_trip_multiplier = 1.9 if trip_type == "Pulang-Pergi" else 1.0
                total_price = single_ticket_price * passengers * round_trip_multiplier
                harga_rupiah = f"Rp {format(int(total_price), ',').replace(',', '.')}"
                
                # Generate ticket details
                source_info = CITY_DATA.get(source)
                destination_info = CITY_DATA.get(destination)
                flight_number = f"{airline.split(' ')[0][:2].upper()}-{np.random.randint(100, 999)}"
                pnr_code = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'), 6))
                airline_logo_path = AIRLINE_LOGOS.get(airline, "assets/default.png")
                airline_logo = img_to_base64(airline_logo_path)
                traveloka_logo = img_to_base64("assets/traveloka.png")
                departure_time = time(np.random.randint(5,14), np.random.choice([0, 15, 30, 45]))
                arrival_datetime = datetime.combine(departure_date, departure_time) + timedelta(hours=duration)
                arrival_time = arrival_datetime.time()
                
                passengers_html_list = ""
                for i, detail in enumerate(passenger_details):
                    name = detail["name"].strip()
                    gender = detail["gender"]
                    title = "Tn." if gender == "Laki-laki" else "Ny."
                    display_name = name if name else f"Penumpang {i + 1}"
                    formatted_name = f"{title} {display_name} (Dewasa)"
                    ticket_number = f"126-2141358{np.random.randint(100, 999)}"
                    passengers_html_list += f"""
                    <div class="passenger-row">
                        <span>{i+1}.</span>
                        <span>{formatted_name}</span>
                        <span>{source_info['code']}-{destination_info['code']}</span>
                        <span>7 KG Bagasi Kabin</span>
                        <span>{ticket_number}</span>
                    </div>
                    """
                
                # Display E-Ticket
                st.markdown("---")
                st.subheader("🧾 Estimasi E-Tiket Anda")
                
                FULL_HTML = textwrap.dedent(f"""
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
                .ticket-box {{
                    font-family: 'Poppins', sans-serif; background: white; border-radius: 15px;
                    border: 1px solid #e0e0e0; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                    color: #333; overflow: hidden; max-width: 700px; margin: 0 auto;
                }}
                .ticket-header {{
                    padding: 15px 25px; display: flex; justify-content: space-between; align-items: center;
                    border-bottom: 1px solid #eee;
                }}
                .ticket-header img {{ height: 25px; }}
                .ticket-header h2 {{ margin: 0; font-size: 20px; font-weight: 600; color: #555; }}
                .ticket-body {{ padding: 25px; }}
                .flight-summary {{ border-bottom: 1px dashed #ccc; padding-bottom: 20px; margin-bottom: 20px; }}
                .flight-summary p {{ margin: 0; font-size: 14px; font-weight: 500; color: #003366; }}
                .airline-info {{ display: flex; align-items: center; margin-top: 10px; }}
                .airline-info img {{ width: 35px; height: 35px; margin-right: 10px; border-radius: 5px; }}
                .airline-info div strong {{ font-size: 18px; font-weight: 600; display: block; }}
                .airline-info div span {{ font-size: 14px; color: #777; }}
                .flight-path-details {{ display: flex; justify-content: space-between; align-items: center; text-align: left; }}
                .flight-time-info .time {{ font-size: 24px; font-weight: 700; margin: 0; }}
                .flight-time-info .date {{ font-size: 14px; margin: 0 0 5px 0; color: #555; }}
                .flight-time-info .city {{ font-size: 16px; font-weight: 600; margin: 0; }}
                .flight-time-info .airport {{ font-size: 12px; color: #888; margin: 0; }}
                .path-icon svg {{ width: 30px; height: 30px; color: #00529b; }}
                .booking-codes {{
                    display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
                    background-color: #f9f9f9; padding: 15px 25px; border-top: 1px solid #eee;
                    border-bottom: 1px solid #eee; margin: 25px -25px;
                }}
                .booking-codes div span {{ font-size: 12px; color: #888; display: block; }}
                .booking-codes div strong {{ font-size: 16px; font-weight: 600; }}
                .passenger-section h3 {{ font-size: 18px; font-weight: 600; margin-bottom: 15px; border-bottom: 2px solid #003366; padding-bottom: 5px; display: inline-block; }}
                .passenger-table .table-header, .passenger-row {{
                    display: grid; grid-template-columns: 30px 2fr 1fr 1.5fr 1.5fr; gap: 10px;
                    padding: 8px 0; font-size: 14px; border-bottom: 1px solid #f0f0f0; align-items: center;
                }}
                .passenger-table .table-header {{ font-weight: 600; color: #888; font-size: 12px; }}
                .price-footer {{ background-color: #f5f7fa; padding: 20px 25px; text-align: right; }}
                .price-footer span {{ font-size: 14px; color: #555; }}
                .price-footer h3 {{ margin: 5px 0 0; font-size: 26px; font-weight: 700; color: #d32f2f; }}
                </style>
                <div class="ticket-box">
                    <div class="ticket-header">
                        <img src="data:image/png;base64,{traveloka_logo}" height="30">
                        <h2>E-Ticket Prediksi</h2>
                    </div>
                    <div class="ticket-body">
                        <div class="flight-summary">
                            <p>Departure Flight / Penerbangan Pergi</p>
                            <div class="airline-info">
                                <img src="data:image/png;base64,{airline_logo}" style="width:35px;height:35px;border-radius:5px;">
                                <div>
                                    <strong>{airline}</strong>
                                    <span>{flight_number} • {flight_class}</span>
                                </div>
                            </div>
                        </div>
                        <div class="flight-path-details">
                            <div class="flight-time-info">
                                <p class="time">{departure_time.strftime('%H:%M')}</p>
                                <p class="date">{departure_date.strftime('%a, %d %b %Y')}</p>
                                <p class="city">{source} ({source_info['code']})</p>
                                <p class="airport">{source_info['name']}</p>
                            </div>
                            <div class="path-icon">
                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
                            </div>
                            <div class="flight-time-info" style="text-align: right;">
                                <p class="time">{arrival_time.strftime('%H:%M')}</p>
                                <p class="date">{arrival_datetime.strftime('%a, %d %b %Y')}</p>
                                <p class="city">{destination} ({destination_info['code']})</p>
                                <p class="airport">{destination_info['name']}</p>
                            </div>
                        </div>
                    </div>
                    <div class="booking-codes">
                        <div><span>ID Pesanan (Prediksi)</span><strong>{np.random.randint(100000000, 999999999)}</strong></div>
                        <div><span>Kode Booking Maskapai (PNR)</span><strong>{pnr_code}</strong></div>
                    </div>
                    <div class="ticket-body" style="padding-top: 0;">
                        <div class="passenger-section">
                            <h3>Passenger Details / Detail Penumpang</h3>
                            <div class="passenger-table">
                                <div class="table-header">
                                    <span>No.</span>
                                    <span>Penumpang</span>
                                    <span>Rute</span>
                                    <span>Fasilitas</span>
                                    <span>No. Tiket</span>
                                </div>
                                {passengers_html_list}
                            </div>
                        </div>
                    </div>
                    <div class="price-footer">
                        <span>Total Estimasi Harga ({trip_type})</span>
                        <h3>{harga_rupiah}</h3>
                    </div>
                </div>
                """)
                components.html(FULL_HTML, height=900, scrolling=True)
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.warning("Silakan rebuild model terlebih dahulu")

# --- MAIN APP ---
def main():
    st.sidebar.title("🎛️ Control Panel")
    mode = st.sidebar.radio("Mode", ["🔮 Prediksi", "🔧 Training & Analysis"])
    
    DATA_PATH = 'data/flight_price.csv'
    MODEL_DIR = 'model'
    
    if mode == "🔮 Prediksi":
        if not os.path.exists(f'{MODEL_DIR}/best_model.joblib'):
            st.error("⚠️ Model belum tersedia. Silakan lakukan training terlebih dahulu!")
            if st.button("🔧 Go to Training Mode"):
                st.rerun()
        else:
            run_prediction_app()
    
    else:  # Training mode
        st.title("🔧 Model Training & Analysis")
        
        if st.sidebar.button("🔄 Generate New Dataset", type="primary"):
            df = generate_flight_data(DATA_PATH)
            st.session_state['df_raw'] = df
        
        if os.path.exists(DATA_PATH):
            if 'df_raw' not in st.session_state:
                st.session_state['df_raw'] = pd.read_csv(DATA_PATH)
            
            df_raw = st.session_state['df_raw']
            
            with st.expander("📄 Raw Data Preview", expanded=False):
                st.dataframe(df_raw.head(20))
            
            # Pipeline
            df_clean = clean_data(df_raw.copy())
            df_augmented = augment_data(df_clean)
            X_scaled, y, encoders, scaler, feature_cols = preprocess_data(df_augmented)
            X_train, X_test, y_train, y_test = split_data(X_scaled, y)
            
            if st.button("🚀 Start Training", type="primary"):
                results, best_model_name = train_and_compare_models(X_train, X_test, y_train, y_test)
                save_artifacts(results, best_model_name, encoders, scaler, feature_cols, MODEL_DIR)
                st.success("✅ Training selesai! Silakan ke mode Prediksi.")
        else:
            st.warning("⚠️ Dataset belum ada. Klik 'Generate New Dataset' untuk memulai.")

if __name__ == "__main__":
    main()