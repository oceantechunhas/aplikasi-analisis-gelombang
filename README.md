# Aplikasi Analisis Gelombang ver0

Versi ini sudah disiapkan untuk **deploy via GitHub + Streamlit Community Cloud**.

## Isi repo
- `app.py` : aplikasi web utama
- `requirements.txt` : dependency Python
- `contoh_format_data_gelombang.txt` : contoh format input

## Cara jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Langkah upload ke GitHub
1. Buat repo baru di GitHub, misalnya `aplikasi-analisis-gelombang`
2. Upload file:
   - `app.py`
   - `requirements.txt`
   - `contoh_format_data_gelombang.txt`
   - `README.md`
3. Commit ke branch `main`

## Langkah deploy ke Streamlit Community Cloud
1. Login ke Streamlit Community Cloud
2. Hubungkan akun GitHub
3. Klik **Create app**
4. Pilih repository GitHub Anda
5. Pilih branch: `main`
6. Main file path: `app.py`
7. Klik **Deploy**

## Catatan
- GitHub menyimpan source code
- Streamlit Community Cloud menjalankan app Python
- GitHub Pages tidak menjalankan app Python; GitHub Pages hanya cocok untuk halaman statis
