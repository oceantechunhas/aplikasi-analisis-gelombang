import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gumbel_r, weibull_min, genextreme

APP_TITLE = "Aplikasi Analisis Gelombang ver0"
FOOTER_TEXT = "copyright OceanerFTUH | special thanks to : Magister Civil Eng UH"


# =========================
# Helpers
# =========================
def read_wave_file(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8", errors="ignore")

    df = None
    last_error = None
    for sep in ["\t", ";", ",", r"\s+"]:
        try:
            tmp = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
            if tmp.shape[1] >= 4:
                df = tmp
                break
        except Exception as e:
            last_error = e

    if df is None or df.shape[1] < 4:
        raise ValueError(f"Tidak bisa membaca file. Error terakhir: {last_error}")

    col_map = {}
    for c in df.columns:
        c_low = str(c).strip().lower()
        if "time" in c_low or "tanggal" in c_low or "waktu" in c_low or "jam" in c_low:
            col_map[c] = "time"
        elif "periode" in c_low or "period" in c_low:
            col_map[c] = "period"
        elif "arah" in c_low or "direction" in c_low:
            col_map[c] = "direction"
        elif "tinggi" in c_low or "height" in c_low or c_low == "hs":
            col_map[c] = "hs"

    df = df.rename(columns=col_map)
    required = ["time", "period", "direction", "hs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak lengkap. Diperlukan: {required}. Terbaca: {list(df.columns)}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    df["direction"] = pd.to_numeric(df["direction"], errors="coerce") % 360.0
    df["hs"] = pd.to_numeric(df["hs"], errors="coerce")

    df = df.dropna(subset=["time", "direction", "hs"]).copy()
    df = df[df["hs"] >= 0].copy()

    if df.empty:
        raise ValueError("Data kosong setelah pembersihan.")

    return df


def record_length_years(df: pd.DataFrame) -> float:
    days = (df["time"].max() - df["time"].min()).total_seconds() / (24 * 3600)
    return max(days / 365.25, 0.01)


def trust_note(n_years: float) -> str:
    trusted = max(1, int(np.floor(2 * n_years)))
    caution = max(trusted + 1, int(np.floor(3 * n_years)))
    return (
        f"Panjang record efektif sekitar {n_years:.2f} tahun. "
        f"Sebagai rule-of-thumb, periode ulang hingga sekitar {trusted} tahun masih lebih layak dibaca. "
        f"Di atas itu hasil tetap diproses tetapi makin ekstrapolatif; sekitar > {caution} tahun "
        f"sebaiknya dibaca dengan kehati-hatian tinggi."
    )


def direction_sector(direction_series: pd.Series, bin_size: float = 22.5):
    centers = np.arange(0, 360, bin_size)
    idx = np.floor(((direction_series.values + bin_size / 2) % 360) / bin_size).astype(int)
    return centers[idx]


def deg_to_compass(deg: float) -> str:
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((deg + 22.5) % 360) // 45)
    return labels[idx]


def dominant_direction_text(df: pd.DataFrame) -> str:
    sectors = direction_sector(df["direction"], 22.5)
    counts = pd.Series(sectors).value_counts()
    if counts.empty:
        return "-"
    deg = float(counts.idxmax())
    return f"{deg:.1f}° ({deg_to_compass(deg)})"


def parse_hs_bins(raw: str):
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) < 2:
        raise ValueError("Kelas Hs harus minimal dua batas.")
    vals = sorted(vals)
    vals[-1] = np.inf
    return vals


def frequency_table(df: pd.DataFrame, bin_size: float, hs_bins):
    dir_labels = np.arange(0, 360, bin_size)
    df2 = df.copy()
    df2["dir_sector"] = direction_sector(df2["direction"], bin_size)

    hs_labels = []
    for i in range(len(hs_bins) - 1):
        a, b = hs_bins[i], hs_bins[i + 1]
        hs_labels.append(f">= {a:.1f}" if np.isinf(b) else f"{a:.1f}–{b:.1f}")

    df2["hs_class"] = pd.cut(df2["hs"], bins=hs_bins, labels=hs_labels, include_lowest=True, right=False)
    table = pd.crosstab(df2["dir_sector"], df2["hs_class"])
    table = table.reindex(index=dir_labels, columns=hs_labels, fill_value=0)
    return table / len(df2) * 100.0, dir_labels


def energy_series(df: pd.DataFrame, bin_size: float):
    dir_labels = np.arange(0, 360, bin_size)
    df2 = df.copy()
    df2["dir_sector"] = direction_sector(df2["direction"], bin_size)
    energy = df2.groupby("dir_sector")["hs"].apply(lambda x: np.sum(x ** 2))
    return energy.reindex(dir_labels, fill_value=0.0), dir_labels


def make_wave_rose_figure(df: pd.DataFrame, mode: str, bin_size: float, hs_bins):
    colors = ["#d9f0ff", "#92c5de", "#4393c3", "#2166ac", "#053061", "#021b2d"]
    if mode == "both":
        fig = plt.figure(figsize=(12, 6), dpi=180)
        axes = [fig.add_subplot(1, 2, 1, projection="polar"), fig.add_subplot(1, 2, 2, projection="polar")]
    else:
        fig = plt.figure(figsize=(8, 7), dpi=180)
        axes = [fig.add_subplot(111, projection="polar")]

    if mode in ["frequency", "both"]:
        ax = axes[0]
        table_pct, dirs = frequency_table(df, bin_size, hs_bins)
        angles = np.deg2rad(dirs)
        width = np.deg2rad(bin_size * 0.9)
        bottom = np.zeros(len(table_pct))
        for i, col in enumerate(table_pct.columns):
            vals = table_pct[col].values
            ax.bar(angles, vals, width=width, bottom=bottom, color=colors[i % len(colors)],
                   edgecolor="black", linewidth=0.5, label=col)
            bottom += vals
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.arange(0, 360, 45), labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        ax.set_title("Frequency Wave Rose", pad=16, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(title="Hs (m)", loc="upper right", bbox_to_anchor=(1.22, 1.10))

    if mode == "energy":
        ax = axes[0]
        energy, dirs = energy_series(df, bin_size)
        angles = np.deg2rad(dirs)
        width = np.deg2rad(bin_size * 0.9)
        ax.bar(angles, energy.values, width=width, color="#2c7fb8", edgecolor="black")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.arange(0, 360, 45), labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        ax.set_title("Energy Wave Rose (Hs²)", pad=16, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.7)

    if mode == "both":
        ax = axes[1]
        energy, dirs = energy_series(df, bin_size)
        angles = np.deg2rad(dirs)
        width = np.deg2rad(bin_size * 0.9)
        ax.bar(angles, energy.values, width=width, color="#2c7fb8", edgecolor="black")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.arange(0, 360, 45), labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        ax.set_title("Energy Wave Rose (Hs²)", pad=16, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.7)

    fig.suptitle("Mawar Gelombang\nArah Gelombang (Dari Arah)", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.015, FOOTER_TEXT, ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def annual_maxima(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["year"] = work["time"].dt.year
    annual = work.groupby("year").agg({"hs": "max", "period": "max"}).rename(
        columns={"hs": "annual_max_hs", "period": "annual_max_period"}
    )
    return annual.reset_index()


def fit_return_levels(x, return_periods, method_name):
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        raise ValueError("Annual maxima minimal 3 tahun untuk fitting distribusi.")

    p = 1.0 - 1.0 / np.asarray(return_periods, dtype=float)

    if method_name == "Gumbel (Fisher-Tippett I)":
        params = gumbel_r.fit(x)
        rl = gumbel_r.ppf(p, *params)
        param_text = f"loc={params[0]:.4f}, scale={params[1]:.4f}"
    elif method_name == "Weibull":
        try:
            params = weibull_min.fit(x, floc=0)
        except Exception:
            params = weibull_min.fit(x)
        rl = weibull_min.ppf(p, *params)
        param_text = f"shape={params[0]:.4f}, loc={params[1]:.4f}, scale={params[2]:.4f}"
    elif method_name == "GEV":
        params = genextreme.fit(x)
        rl = genextreme.ppf(p, *params)
        param_text = f"shape={params[0]:.4f}, loc={params[1]:.4f}, scale={params[2]:.4f}"
    else:
        raise ValueError("Metode tidak dikenali.")

    return rl, param_text


def make_extreme_plot(annual: pd.DataFrame, result: pd.DataFrame, method: str):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    ax.plot(annual["year"], annual["annual_max_hs"], marker="o", label="Annual max Hs")
    ax.plot(result["Return Period (yr)"], result["Hs Return Level (m)"], marker="s", linestyle="--", label="Hs return level")
    ax2 = ax.twinx()
    ax2.plot(annual["year"], annual["annual_max_period"], marker="^", linestyle="-.", color="tab:green", label="Annual max period")
    ax.set_xlabel("Tahun / Return period")
    ax.set_ylabel("Hs (m)")
    ax2.set_ylabel("Period (s)")
    ax.set_title(f"Analisis Ekstrem Gelombang\nMetode: {method}")
    ax.grid(True, linestyle="--", alpha=0.7)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.text(0.5, 0.01, FOOTER_TEXT, ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def report_text(df: pd.DataFrame, result: pd.DataFrame, method: str) -> str:
    n_years = record_length_years(df)
    max_row = result.loc[result["Return Period (yr)"].idxmax()]
    return (
        f"{APP_TITLE}\n\n"
        "Laporan Ringkas Analisis Gelombang\n\n"
        "RESUME DATA\n"
        f"Jumlah data: {len(df):,} baris\n"
        f"Rentang data: {df['time'].min()} s/d {df['time'].max()}\n"
        f"Panjang data efektif: {n_years:.2f} tahun\n"
        f"Hs min / max / mean: {df['hs'].min():.3f} / {df['hs'].max():.3f} / {df['hs'].mean():.3f} m\n"
        f"Periode min / max / mean: {df['period'].min():.3f} / {df['period'].max():.3f} / {df['period'].mean():.3f} s\n"
        f"Arah dominan: {dominant_direction_text(df)}\n\n"
        "DESKRIPSI ANALISIS\n"
        f"Analisis ekstrem dilakukan dengan metode {method} berbasis annual maxima.\n\n"
        "KESIMPULAN\n"
        f"Pada periode ulang {int(max_row['Return Period (yr)'])} tahun, Hs proyeksi sekitar "
        f"{max_row['Hs Return Level (m)']:.2f} m dan periode proyeksi sekitar "
        f"{max_row['Period Return Level (s)']:.2f} s.\n\n"
        "CATATAN\n"
        f"{trust_note(n_years)}\n\n"
        f"{FOOTER_TEXT}\n"
    )


def generate_report_pdf(df: pd.DataFrame, wave_fig, result: pd.DataFrame, method: str) -> bytes:
    annual = annual_maxima(df)
    txt = report_text(df, result, method)

    out = io.BytesIO()
    with PdfPages(out) as pdf:
        # page 1
        fig1 = plt.figure(figsize=(8.27, 11.69))
        ax1 = fig1.add_axes([0, 0, 1, 1])
        ax1.axis("off")
        ax1.text(0.5, 0.965, APP_TITLE, ha="center", va="top", fontsize=18, fontweight="bold")
        ax1.text(0.5, 0.938, "Laporan Ringkas Analisis Gelombang", ha="center", va="top", fontsize=11)
        ax1.text(0.07, 0.88, textwrap.fill(txt, 100), ha="left", va="top", fontsize=9.5, linespacing=1.45)
        ax1.text(0.5, 0.025, FOOTER_TEXT, ha="center", va="bottom", fontsize=9)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        # page 2
        fig2 = plt.figure(figsize=(8.27, 11.69))
        ax2 = fig2.add_axes([0, 0, 1, 1])
        ax2.axis("off")
        ax2.text(0.5, 0.965, "Gambar Mawar Gelombang", ha="center", va="top", fontsize=16, fontweight="bold")

        img_buf = io.BytesIO()
        wave_fig.savefig(img_buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
        img_buf.seek(0)
        img = plt.imread(img_buf)
        ax_img = fig2.add_axes([0.08, 0.23, 0.84, 0.62])
        ax_img.imshow(img)
        ax_img.set_aspect("equal")
        ax_img.axis("off")

        ax2.text(0.5, 0.025, FOOTER_TEXT, ha="center", va="bottom", fontsize=9)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

    out.seek(0)
    return out.getvalue()


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(FOOTER_TEXT)

with st.sidebar:
    st.header("Upload dan Pengaturan")
    uploaded = st.file_uploader("Upload data gelombang", type=["txt", "csv", "tsv"])
    mode = st.selectbox("Jenis grafik", ["frequency", "energy", "both"], index=0)
    bin_size = st.number_input("Lebar sektor arah (deg)", min_value=5.0, max_value=90.0, value=22.5, step=2.5)
    hs_bins_raw = st.text_input("Kelas Hs (m)", value="0,0.5,1,1.5,2,999")
    method = st.selectbox("Metode ekstrem", ["Gumbel (Fisher-Tippett I)", "Weibull", "GEV"])
    rp_raw = st.text_input("Return period (tahun)", value="5,10,25,50")

st.download_button(
    "Download contoh format data",
    data=(
        "time\tPERIODE\tARAH GELOMBANG\tTINGGI GELOMBANG\n"
        "2022-11-01T03:00:00.000Z\t2.619999886\t55.88000107\t0.079999998\n"
        "2022-11-01T06:00:00.000Z\t2.609999895\t57.66000366\t0.07\n"
        "2022-11-01T09:00:00.000Z\t2.589999914\t58.47000122\t0.059999999\n"
    ),
    file_name="contoh_format_data_gelombang.txt",
    mime="text/plain"
)

if uploaded is not None:
    try:
        df = read_wave_file(uploaded)
    except Exception as e:
        st.error(str(e))
        st.stop()

    n_years = record_length_years(df)
    st.error(trust_note(n_years))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah data", f"{len(df):,}")
    c2.metric("Hs rata-rata (m)", f"{df['hs'].mean():.3f}")
    c3.metric("Hs maksimum (m)", f"{df['hs'].max():.3f}")
    c4.metric("Arah dominan", dominant_direction_text(df))

    st.subheader("Preview data")
    st.dataframe(df.head(20), use_container_width=True)

    try:
        hs_bins = parse_hs_bins(hs_bins_raw)
        return_periods = sorted([float(x.strip()) for x in rp_raw.split(",") if x.strip()])
    except Exception as e:
        st.error(f"Input tidak valid: {e}")
        st.stop()

    st.subheader("Mawar Gelombang")
    wave_fig = make_wave_rose_figure(df, mode, float(bin_size), hs_bins)
    st.pyplot(wave_fig, use_container_width=True)

    st.subheader("Analisis Ekstrem")
    annual = annual_maxima(df)
    hs_rl, hs_param = fit_return_levels(annual["annual_max_hs"], return_periods, method)
    tp_rl, tp_param = fit_return_levels(annual["annual_max_period"], return_periods, method)
    result = pd.DataFrame({
        "Return Period (yr)": return_periods,
        "Hs Return Level (m)": hs_rl,
        "Period Return Level (s)": tp_rl
    })

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Annual maxima**")
        st.dataframe(
            annual.rename(columns={
                "year": "Tahun",
                "annual_max_hs": "Hs Maks Tahunan (m)",
                "annual_max_period": "Periode Maks Tahunan (s)"
            }),
            use_container_width=True
        )
    with col_b:
        st.markdown("**Return level**")
        st.dataframe(result.round(3), use_container_width=True)

    st.caption(f"Parameter fit Hs: {hs_param}")
    st.caption(f"Parameter fit Period: {tp_param}")

    ext_fig = make_extreme_plot(annual, result, method)
    st.pyplot(ext_fig, use_container_width=True)

    pdf_bytes = generate_report_pdf(df, wave_fig, result, method)
    st.download_button(
        "Download laporan PDF",
        data=pdf_bytes,
        file_name="laporan_analisis_gelombang.pdf",
        mime="application/pdf"
    )

    st.download_button(
        "Download teks laporan untuk Word",
        data=report_text(df, result, method),
        file_name="laporan_analisis_gelombang.txt",
        mime="text/plain"
    )
else:
    st.info("Upload file data gelombang untuk memulai analisis.")
