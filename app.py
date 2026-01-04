import pandas as pd
import streamlit as st
import joblib
import os
import numpy as np
from datetime import datetime, timedelta, time
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import time as pytime
import base64
import textwrap
import streamlit.components.v1 as components



# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi Harga Tiket Pesawat", page_icon="✈️", layout="centered")

# --- Bagian 1: Logika Durasi & Fungsi Setup Otomatis ---


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


# Kamus untuk menyimpan estimasi durasi penerbangan (dalam jam)
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
    """Mendapatkan durasi dari kamus, menangani rute bolak-balik."""
    if (source, destination) in FLIGHT_DURATIONS:
        return FLIGHT_DURATIONS[(source, destination)]
    elif (destination, source) in FLIGHT_DURATIONS:
        return FLIGHT_DURATIONS[(destination, source)]
    else:
        return 2.0 # Durasi default jika rute tidak ditemukan

def generate_flight_data(file_path):
    """Membuat dataset dummy yang lebih realistis dengan fitur baru."""
    with st.spinner("Mengkalibrasi ulang data harga, mohon tunggu..."):
        pytime.sleep(1)
        
        NUM_ROWS = 2500
        AIRLINES = ['Garuda Indonesia', 'Lion Air', 'Citilink', 'Batik Air', 'AirAsia', 'Sriwijaya Air']
        CITIES = ['Jakarta', 'Surabaya', 'Medan', 'Makassar', 'Denpasar', 'Yogyakarta', 'Balikpapan']
        CLASSES = ['Economy', 'Business']

        booking_dates = [datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 364)) for _ in range(NUM_ROWS)]
        days_to_departure_list = np.random.randint(1, 90, size=NUM_ROWS)
        departure_dates = [book_date + timedelta(days=int(dtd)) for book_date, dtd in zip(booking_dates, days_to_departure_list)]

        airlines = np.random.choice(AIRLINES, size=NUM_ROWS)
        flight_classes = np.random.choice(CLASSES, size=NUM_ROWS, p=[0.85, 0.15])
        
        sources, destinations = [], []
        for _ in range(NUM_ROWS):
            pair = np.random.choice(CITIES, size=2, replace=False)
            sources.append(pair[0])
            destinations.append(pair[1])
        
        durations = [get_duration(src, dst) + np.random.uniform(-0.2, 0.2) for src, dst in zip(sources, destinations)]
        durations = [round(max(0.5, d), 1) for d in durations]

        # --- LOGIKA HARGA BARU YANG LEBIH REALISTIS ---
        base_prices = np.random.randint(400, 1200, size=NUM_ROWS) 
        price_modifier = (90 - days_to_departure_list) / 90 
        prices = base_prices + (base_prices * price_modifier * np.random.uniform(0.1, 0.6))
        prices = np.where(flight_classes == 'Business', prices * np.random.uniform(1.6, 2.2), prices)
        
        df = pd.DataFrame({
            'Airline': airlines, 'Booking_Date': booking_dates, 'Departure_Date': departure_dates,
            'Days_to_Departure': days_to_departure_list, 'Source': sources,
            'Destination': destinations, 'Duration': durations, 'Class': flight_classes, 'Price': prices
        })
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
    st.success(f"Dataset baru berhasil dibuat di '{file_path}'")
    pytime.sleep(1)

def train_and_save_model(data_path, model_path, encoders_path):
    """Melatih model dari data dan menyimpannya."""
    with st.spinner("Melatih ulang model AI dengan data baru..."):
        pytime.sleep(1)
        df = pd.read_csv(data_path)
        
        df['Departure_Date'] = pd.to_datetime(df['Departure_Date'])
        df['Month'] = df['Departure_Date'].dt.month
        
        categorical_cols = ["Airline", "Source", "Destination", "Class"]
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            
        X = df[["Airline", "Source", "Destination", "Duration", "Class", "Days_to_Departure", "Month"]]
        y = df["Price"]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(encoders, encoders_path)
    st.success("Model baru berhasil dilatih dan disimpan!")
    pytime.sleep(1)

# --- Bagian 2: Fungsi Utama Aplikasi ---

def run_app():
    """Menjalankan antarmuka utama aplikasi prediksi."""
    try:
        model, encoders = load_model_and_encoders()
        
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

        # Input dari Pengguna
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


        if st.button("Prediksi Harga"):
            days_to_departure = (departure_date - booking_date).days
            
            if source == destination:
                st.warning("Kota Asal dan Kota Tujuan tidak boleh sama.")
            elif days_to_departure < 0:
                st.error("Tanggal Keberangkatan tidak boleh sebelum Tanggal Pemesanan.")
            else:
                airline_enc = encoders["Airline"].transform([airline])[0]
                source_enc = encoders["Source"].transform([source])[0]
                destination_enc = encoders["Destination"].transform([destination])[0]
                class_raw = flight_class.split(' ')[0]
                class_enc = encoders["Class"].transform([class_raw])[0]
                month = departure_date.month

                sample = pd.DataFrame({
                    "Airline": [airline_enc], "Source": [source_enc], "Destination": [destination_enc],
                    "Duration": [duration], "Class": [class_enc], "Days_to_Departure": [days_to_departure],
                    "Month": [month]
                })

                single_ticket_price = model.predict(sample)[0] * 1000
                round_trip_multiplier = 1.9 if trip_type == "Pulang-Pergi" else 1.0
                total_price = single_ticket_price * passengers * round_trip_multiplier
                harga_rupiah = f"Rp {format(int(total_price), ',').replace(',', '.')}"
                
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

                # --- Menampilkan E-Tiket (PERBAIKAN RENDERING) ---
                st.markdown("---")
                st.subheader("🧾 Estimasi E-Tiket Anda")
                
                # Menggabungkan CSS dan HTML menjadi satu blok untuk rendering yang lebih stabil
                FULL_HTML = textwrap.dedent(f"""
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
                .ticket-box {{
                    font-family: 'Poppins', sans-serif; background: white; border-radius: 15px;
                    border: 1px solid #e0e0e0; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                    color: #333; overflow: hidden;
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
                                <img src="data:image/png;base64,{airline_logo}"
                            style="width:35px;height:35px;border-radius:5px;">
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
                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
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
        st.error(f"Terjadi kesalahan saat memuat aplikasi: {e}")
        st.warning("Mencoba membersihkan file lama dan memulai ulang...")
        if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        if os.path.exists(ENCODERS_PATH): os.remove(ENCODERS_PATH)
        if os.path.exists(DATA_PATH): os.remove(DATA_PATH)
        pytime.sleep(1)
        st.rerun()

# --- Bagian 3: Logika Inisialisasi ---

DATA_PATH = 'data/flight_price.csv'
MODEL_PATH = 'model/flight_price_model.joblib'
ENCODERS_PATH = 'model/encoders.joblib'

@st.cache_resource
def load_model_and_encoders():
    """Memuat model dan encoders yang sudah ada."""
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, encoders

# --- Logika Inisialisasi yang Diperbarui ---
needs_rebuild = True
if os.path.exists(DATA_PATH) and os.path.exists(MODEL_PATH):
    try:
        df_check = pd.read_csv(DATA_PATH)
        # Kriteria utama: Cek apakah file data sudah memiliki kolom 'Days_to_Departure'
        # dan harga maksimumnya masuk akal (di bawah 5000, karena data lama harganya sangat tinggi)
        if 'Days_to_Departure' in df_check.columns and df_check['Price'].max() < 5000:
            needs_rebuild = False
    except Exception:
        needs_rebuild = True
else:
    needs_rebuild = True

if needs_rebuild:
    st.info("Model atau data usang terdeteksi. Mempersiapkan aplikasi...")
    if os.path.exists(DATA_PATH): os.remove(DATA_PATH)
    if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
    if os.path.exists(ENCODERS_PATH): os.remove(ENCODERS_PATH)
    generate_flight_data(DATA_PATH)
    train_and_save_model(DATA_PATH, MODEL_PATH, ENCODERS_PATH)
    st.success("Setup selesai! Memuat aplikasi...")
    pytime.sleep(2)
    st.rerun()
else:
    run_app()

