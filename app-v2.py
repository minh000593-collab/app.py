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

st.info("⚠️ Kết quả chỉ mang tính tham khảo")

st.divider()

# ======================
# SESSION STATE
# ======================
if "scores" not in st.session_state:
    st.session_state.scores = []

# ======================
# CONSENT
# ======================
consent = st.checkbox("Đồng ý cho AI phân tích audio")

# ======================
# UPLOAD
# ======================
audio_file = st.file_uploader("Upload .wav", type=["wav"])

# ======================
# LOAD AUDIO
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
# FEATURES
# ======================
def extract_features(y, sr):
    duration = len(y) / sr

    energy = np.abs(y)
    energy_mean = np.mean(energy) + 1e-6

    activity = np.sum(energy > energy_mean)
    activity_rate = activity / max(duration, 1e-6)

    silence_ratio = np.sum(energy < 0.01) / len(energy)

    zero_crossings = np.sum(np.abs(np.diff(np.sign(y)))) / len(y)
    pitch = 1 - min(zero_crossings, 1)

    return activity_rate, silence_ratio, duration, pitch

# ======================
# CLASSIFY
# ======================
def classify_speed(rate):
    if 80 <= rate <= 140:
        return "ổn 👍", 2
    elif 60 <= rate < 80 or 140 < rate <= 180:
        return "hơi lệch ⚠️", 1
    else:
        return "cần luyện ⛔", 0


def classify_silence(ratio):
    if ratio <= 0.25:
        return "tốt 👍", 2
    elif ratio <= 0.4:
        return "hơi ngắt ⚠️", 1
    else:
        return "nhiều ngắt ⛔", 0


def classify_filler():
    filler = 5

    if filler <= 3:
        return "ít 👍", 2
    elif filler <= 7:
        return "có ⚠️", 1
    else:
        return "nhiều ⛔", 0


def classify_pitch(p):
    if p >= 0.7:
        return "ổn 👍", 2
    elif p >= 0.4:
        return "dao động ⚠️", 1
    else:
        return "chưa ổn ⛔", 0

# ======================
# SCORE
# ======================
def score(total):
    return round(total / 8 * 10, 1)

# ======================
# FEEDBACK (GỌN + NATURAL)
# ======================
def feedback(speed, silence, filler, pitch):
    return [
        f"🗣 Tốc độ: {speed}",
        f"⏱ Nhịp nghỉ: {silence}",
        f"💬 Từ đệm: {filler}",
        f"🎤 Cao độ: {pitch}",
        "💡 Tip: ưu tiên dừng nhẹ thay vì dùng từ đệm"
    ]

# ======================
# MAIN
# ======================
if audio_file is not None:

    if not consent:
        st.warning("Cần đồng ý trước khi phân tích")
        st.stop()

    st.success("Đã nhận file")

    y, sr, err = load_audio_safe(audio_file)

    if err:
        st.error("Lỗi file")
        st.code(err)

    else:

        # waveform
        st.subheader("Waveform")
        fig, ax = plt.subplots()
        ax.plot(y)
        st.pyplot(fig)

        st.divider()

        # analysis
        a, s, d, p = extract_features(y, sr)

        sp, sc1 = classify_speed(a)
        sl, sc2 = classify_silence(s)
        fu, sc3 = classify_filler()
        pi, sc4 = classify_pitch(p)

        total = sc1 + sc2 + sc3 + sc4
        final_score = score(total)

        st.session_state.scores.append(final_score)

        # results
        st.subheader("Kết quả")

        st.write("⏱", round(d, 2), "giây")
        st.write("🎯 Điểm:", final_score, "/10")

        st.write("🗣", sp)
        st.write("⏱", sl)
        st.write("💬", fu)
        st.write("🎤", pi)

        # feedback
        st.subheader("Gợi ý")
        for f in feedback(sp, sl, fu, pi):
            st.write(f)

        # progress (>= 2 files)
        if len(st.session_state.scores) >= 2:
            st.subheader("Tiến bộ")

            fig2, ax2 = plt.subplots()
            ax2.plot(st.session_state.scores)
            ax2.set_title("Score theo thời gian")
            ax2.set_xlabel("Lần")
            ax2.set_ylabel("Điểm /10")

            st.pyplot(fig2)

else:
    st.info("Upload file WAV để bắt đầu")
