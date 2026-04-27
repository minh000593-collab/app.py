import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import soundfile as sf

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Debate Coach AI", layout="centered")

st.title("🎤 Debate Coach AI")
st.write("Phân tích kỹ năng nói từ file ghi âm (.wav)")

st.divider()

# ======================
# UPLOAD
# ======================
audio_file = st.file_uploader("Upload file ghi âm (.wav)", type=["wav"])

# ======================
# LOAD AUDIO SAFE
# ======================
def load_audio_safe(file):
    try:
        file.seek(0)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        y, sr = sf.read(tmp_path)

        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        return y, sr, None

    except Exception as e:
        return None, None, str(e)

# ======================
# FEATURE EXTRACTION
# ======================
def extract_features(y, sr):
    duration = len(y) / sr

    energy = np.abs(y)
    energy_mean = np.mean(energy) + 1e-6

    activity = np.sum(energy > energy_mean)
    activity_rate = activity / max(duration, 1e-6)

    silence_ratio = np.sum(energy < 0.01) / len(energy)

    return activity_rate, silence_ratio, duration

# ======================
# RUBRIC
# ======================
def classify_speed(rate):
    if 80 <= rate <= 140:
        return "Tốt", 2
    elif 60 <= rate < 80 or 140 < rate <= 180:
        return "Trung bình", 1
    else:
        return "Cần cải thiện", 0


def classify_silence(ratio):
    if ratio <= 0.25:
        return "Tốt", 2
    elif ratio <= 0.4:
        return "Trung bình", 1
    else:
        return "Cần cải thiện", 0


def classify_filler():
    filler_count = 5  # placeholder

    if filler_count <= 3:
        return "Tốt", 2
    elif filler_count <= 7:
        return "Trung bình", 1
    else:
        return "Cần cải thiện", 0


def overall_label(total):
    if total >= 5:
        return "Tốt"
    elif total >= 3:
        return "Trung bình"
    else:
        return "Cần cải thiện"

# ======================
# FEEDBACK
# ======================
def generate_feedback(speed, silence, filler):
    fb = []

    if speed == "Tốt":
        fb.append("✔ Tốc độ nói ổn định.")
    elif speed == "Trung bình":
        fb.append("⚠ Tốc độ chưa đều.")
    else:
        fb.append("❌ Cần cải thiện tốc độ nói.")

    if silence == "Tốt":
        fb.append("✔ Nhịp nghỉ hợp lý.")
    elif silence == "Trung bình":
        fb.append("⚠ Có vài khoảng dừng dài.")
    else:
        fb.append("❌ Quá nhiều khoảng lặng.")

    if filler == "Tốt":
        fb.append("✔ Ít từ đệm.")
    elif filler == "Trung bình":
        fb.append("⚠ Có từ đệm.")
    else:
        fb.append("❌ Từ đệm nhiều.")

    fb.append("💡 Gợi ý: luyện nói chậm hơn và thay từ đệm bằng pause tự nhiên.")

    return fb

# ======================
# MAIN
# ======================
if audio_file is not None:

    st.success("Upload thành công!")

    y, sr, err = load_audio_safe(audio_file)

    if err:
        st.error("Lỗi đọc file")
        st.code(err)

    else:

        # ======================
        # 1. WAVEFORM FIRST (IMPORTANT)
        # ======================
        st.subheader("📈 Waveform")

        fig, ax = plt.subplots()
        ax.plot(y)
        ax.set_title("Audio waveform")
        st.pyplot(fig)

        st.divider()

        # ======================
        # 2. ANALYSIS
        # ======================
        with st.spinner("Đang phân tích..."):

            activity_rate, silence_ratio, duration = extract_features(y, sr)

            speed_label, speed_score = classify_speed(activity_rate)
            silence_label, silence_score = classify_silence(silence_ratio)
            filler_label, filler_score = classify_filler()

            total = speed_score + silence_score + filler_score
            overall = overall_label(total)

        # ======================
        # 3. RESULTS
        # ======================
        st.subheader("📊 Kết quả")

        st.write(f"⏱ Thời lượng: {duration:.2f}s")
        st.write(f"🧠 Activity rate: {activity_rate:.2f}")
        st.write(f"🔇 Khoảng lặng: {silence_ratio:.2f}")

        st.subheader("🏁 Chấm điểm")

        st.write("🗣 Tốc độ:", speed_label)
        st.write("🔇 Khoảng lặng:", silence_label)
        st.write("💬 Từ đệm:", filler_label)

        st.write("🎯 Overall:", overall)

        # ======================
        # 4. FEEDBACK
        # ======================
        st.subheader("💡 Feedback")

        for f in generate_feedback(speed_label, silence_label, filler_label):
            st.write("•", f)

else:
    st.info("Upload file WAV để bắt đầu 🎤")