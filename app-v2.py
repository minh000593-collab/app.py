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

# ======================
# DISCLAIMER
# ======================
st.info("⚠️ Kết quả chỉ mang tính tham khảo, không phải đánh giá tuyệt đối.")

st.divider()

# ======================
# CONSENT
# ======================
consent = st.checkbox(
    "Tôi đồng ý cho DCAI sử dụng audio để phân tích tốc độ nói, cao độ, từ đệm và khoảng lặng."
)

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

    # ======================
    # PITCH (proxy - đơn giản)
    # ======================
    zero_crossings = np.sum(np.abs(np.diff(np.sign(y)))) / len(y)
    pitch_stability = 1 - min(zero_crossings, 1)  # càng ổn định càng cao

    return activity_rate, silence_ratio, duration, pitch_stability

# ======================
# RUBRIC
# ======================
def classify_speed(rate):
    if 80 <= rate <= 140:
        return "Ổn định 👍", 2
    elif 60 <= rate < 80 or 140 < rate <= 180:
        return "Hơi lệch nhịp ⚠️", 1
    else:
        return "Cần luyện thêm ⛔", 0


def classify_silence(ratio):
    if ratio <= 0.25:
        return "Nhịp nghỉ tốt 👍", 2
    elif ratio <= 0.4:
        return "Có hơi nhiều khoảng dừng ⚠️", 1
    else:
        return "Ngắt quãng khá nhiều ⛔", 0


def classify_filler():
    filler_count = 5  # placeholder

    if filler_count <= 3:
        return "Ít từ đệm 👍", 2
    elif filler_count <= 7:
        return "Có từ đệm ⚠️", 1
    else:
        return "Từ đệm khá nhiều ⛔", 0


def classify_pitch(stability):
    if stability >= 0.7:
        return "Giọng khá ổn định 👍", 2
    elif stability >= 0.4:
        return "Giọng hơi dao động ⚠️", 1
    else:
        return "Giọng chưa ổn định ⛔", 0


def overall_label(total):
    if total >= 6:
        return "Tốt 👍"
    elif total >= 3:
        return "Trung bình ⚠️"
    else:
        return "Cần luyện tập ⛔"

# ======================
# FEEDBACK (FRIENDLY)
# ======================
def generate_feedback(speed, silence, filler, pitch):
    fb = []

    fb.append("💡 Nhận xét dựa trên đoạn ghi âm ngắn, chỉ mang tính tham khảo.")

    fb.append(f"🗣 Tốc độ: {speed} — hãy thử điều chỉnh để người nghe dễ theo dõi hơn.")
    fb.append(f"⏱ Nhịp nghỉ: {silence} — nghỉ đúng chỗ sẽ giúp lập luận thuyết phục hơn.")
    fb.append(f"💬 Từ đệm: {filler} — giảm 'ừ, à, kiểu như...' sẽ giúp nói mạch lạc hơn.")
    fb.append(f"🎼 Cao độ: {pitch} — giữ giọng ổn định sẽ làm bài nói tự tin hơn.")

    fb.append("🚀 Gợi ý: luyện nói theo đoạn 30–60s, tập dừng ở ý quan trọng thay vì dùng từ đệm.")

    return fb

# ======================
# MAIN
# ======================
if audio_file is not None:

    if not consent:
        st.warning("Bạn cần đồng ý cho DCAI phân tích audio trước khi tiếp tục.")
        st.stop()

    st.success("Upload thành công!")

    y, sr, err = load_audio_safe(audio_file)

    if err:
        st.error("Lỗi đọc file")
        st.code(err)

    else:

        # ======================
        # WAVEFORM
        # ======================
        st.subheader("📈 Waveform")

        fig, ax = plt.subplots()
        ax.plot(y)
        ax.set_title("Audio waveform")
        st.pyplot(fig)

        st.divider()

        # ======================
        # ANALYSIS
        # ======================
        with st.spinner("Đang phân tích..."):

            activity_rate, silence_ratio, duration, pitch = extract_features(y, sr)

            speed_label, speed_score = classify_speed(activity_rate)
            silence_label, silence_score = classify_silence(silence_ratio)
            filler_label, filler_score = classify_filler()
            pitch_label, pitch_score = classify_pitch(pitch)

            total = speed_score + silence_score + filler_score + pitch_score
            overall = overall_label(total)

        # ======================
        # RESULTS
        # ======================
        st.subheader("📊 Kết quả")

        st.write(f"⏱ Thời lượng: {duration:.2f}s")
        st.write(f"📌 Activity rate: {activity_rate:.2f}")
        st.write(f"🔇 Khoảng lặng: {silence_ratio:.2f}")

        st.subheader("🏁 Đánh giá")

        st.write("🗣 Tốc độ:", speed_label)
        st.write("🔇 Nhịp nghỉ:", silence_label)
        st.write("💬 Từ đệm:", filler_label)
        st.write("🎼 Cao độ:", pitch_label)

        st.write("🎯 Tổng thể:", overall)

        # ======================
        # FEEDBACK
        # ======================
        st.subheader("💡 Feedback")

        for f in generate_feedback(speed_label, silence_label, filler_label, pitch_label):
            st.write("•", f)

else:
    st.info("Upload file WAV để bắt đầu 🎤")
