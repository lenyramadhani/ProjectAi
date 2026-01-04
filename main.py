import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Baca dataset
df = pd.read_csv("data/flight_price.csv")

# 2. Encode kolom kategori
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month

categorical_cols = ["Airline", "Source", "Destination"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 3. Pilih fitur (X) dan target (y)
X = df[["Airline", "Source", "Destination", "Month", "Duration"]]
y = df["Price"]

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluasi
y_pred = model.predict(X_test)

# Skala harga biar realistis (x1000)
y_pred = y_pred * 1000
y_test = y_test * 1000

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Model Evaluation ===")
print(f"Mean Absolute Error (MAE): Rp {mae:,.0f}")
print(f"R2 Score: {r2:.2f}")

# 7. Input user
print("\n=== Prediksi Harga Tiket (Input User) ===")
airline = input("Masukkan Maskapai: ")
source = input("Masukkan Kota Asal: ")
destination = input("Masukkan Kota Tujuan: ")
month = int(input("Masukkan Bulan Keberangkatan (1-12): "))
duration = float(input("Masukkan Durasi Penerbangan (jam): "))

# Encode input user
try:
    airline_enc = encoders["Airline"].transform([airline])[0]
    source_enc = encoders["Source"].transform([source])[0]
    destination_enc = encoders["Destination"].transform([destination])[0]
except ValueError:
    print("\nError: Input tidak sesuai dataset!")
    exit()

sample = pd.DataFrame({
    "Airline": [airline_enc],
    "Source": [source_enc],
    "Destination": [destination_enc],
    "Month": [month],
    "Duration": [duration]
})

# Prediksi
pred_price = model.predict(sample)[0]
pred_price = pred_price * 1000  # ubah skala ke jutaan

harga_rupiah = f"Rp {format(int(pred_price), ',').replace(',', '.')}"
print(f"\nPrediksi Harga Tiket: {harga_rupiah} IDR")
