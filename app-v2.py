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
st.write("Phân tích kỹ năng nói qua file ghi âm")

st.info("Kết quả chỉ mang tính tham khảo và nhằm mục đích luyện tập")

st.divider()

# ======================
# SESSION STATE (PROGRESS TRACKING)
# ======================
if "scores" not in st.session_state:
    st.session_state.scores = []

# ======================
# CONSENT
# ======================
consent = st.checkbox(
    "Mình đồng ý cho hệ thống phân tích audio để hỗ trợ cải thiện kỹ năng nói"
)

# ======================
# UPLOAD
# ======================
audio_file = st.file_uploader("Upload file .wav", type=["wav"])

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
    pitch_stability = 1 - min(zero_crossings, 1)

    return activity_rate, silence_ratio, duration, pitch_stability

# ======================
# CLASSIFIERS
# ======================
def classify_speed(rate):
    if 80 <= rate <= 140:
        return "ổn định", 2
    elif 60 <= rate < 80 or 140 < rate <= 180:
        return "hơi chưa đều", 1
    else:
        return "cần luyện thêm", 0


def classify_silence(ratio):
    if ratio <= 0.25:
        return "nhịp nghỉ hợp lý", 2
    elif ratio <= 0.4:
        return "đôi lúc hơi ngắt quãng", 1
    else:
        return "ngắt nghỉ khá nhiều", 0


def classify_filler():
    filler_count = 5

    if filler_count <= 3:
        return "ít dùng từ đệm", 2
    elif filler_count <= 7:
        return "có xuất hiện từ đệm", 1
    else:
        return "dùng từ đệm khá nhiều", 0


def classify_pitch(stability):
    if stability >= 0.7:
        return "giọng khá ổn định", 2
    elif stability >= 0.4:
        return "giọng hơi dao động", 1
    else:
        return "giọng còn chưa ổn định", 0

# ======================
# SCORE (OUT OF 10)
# ======================
def compute_score(total):
    return round((total / 8) * 10, 1)

# ======================
# FRIENDLY FEEDBACK
# ======================
def generate_feedback(speed, silence, filler, pitch):
    return [
        "Mình đã xem qua đoạn nói của bạn và thấy có vài điểm thú vị có thể cải thiện nhẹ.",
        f"Về tốc độ nói thì hiện tại đang ở mức {speed}. Bạn có thể thử nói chậm hơn một chút ở những ý quan trọng để người nghe dễ theo dõi hơn.",
        f"Nhịp nghỉ của bạn đang {silence}. Nếu thêm vài khoảng dừng tự nhiên, phần lập luận sẽ rõ ràng hơn.",
        f"Phần từ đệm thì {filler}. Khi giảm bớt những từ như ừ, à, thì bài nói sẽ mượt hơn đáng kể.",
        f"Giọng nói có độ ổn định {pitch}. Bạn đang đi đúng hướng rồi, chỉ cần luyện thêm để giữ nhịp đều hơn.",
        "Nếu luyện đều mỗi ngày với đoạn 30 đến 60 giây, bạn sẽ thấy cải thiện rất nhanh."
    ]

# ======================
# MAIN
# ======================
if audio_file is not None:

    if not consent:
        st.warning("Bạn cần đồng ý trước khi hệ thống phân tích audio")
        st.stop()

    st.success("Upload thành công")

    y, sr, err = load_audio_safe(audio_file)

    if err:
        st.error("Không đọc được file")
        st.code(err)

    else:

        # ======================
        # WAVEFORM
        # ======================
        st.subheader("Waveform")

        fig, ax = plt.subplots()
        ax.plot(y)
        ax.set_title("Audio signal")
        st.pyplot(fig)

        st.divider()

        # ======================
        # ANALYSIS
        # ======================
        activity_rate, silence_ratio, duration, pitch = extract_features(y, sr)

        speed_label, speed_score = classify_speed(activity_rate)
        silence_label, silence_score = classify_silence(silence_ratio)
        filler_label, filler_score = classify_filler()
        pitch_label, pitch_score = classify_pitch(pitch)

        total = speed_score + silence_score + filler_score + pitch_score
        score_10 = compute_score(total)

        # save progress
        st.session_state.scores.append(score_10)

        # ======================
        # RESULTS
        # ======================
        st.subheader("Kết quả")

        st.write("Thời lượng:", round(duration, 2), "giây")

        st.write("Tổng điểm:", score_10, "/10")

        st.write("Tốc độ:", speed_label)
        st.write("Nhịp nghỉ:", silence_label)
        st.write("Từ đệm:", filler_label)
        st.write("Cao độ:", pitch_label)

        # ======================
        # FEEDBACK
        # ======================
        st.subheader("Gợi ý cải thiện")

        for f in generate_feedback(speed_label, silence_label, filler_label, pitch_label):
            st.write(f)

        # ======================
        # PROGRESS CHART
        # ======================
        st.subheader("Tiến bộ của bạn")

        if len(st.session_state.scores) > 1:

            fig2, ax2 = plt.subplots()
            ax2.plot(st.session_state.scores)
            ax2.set_title("Điểm theo thời gian luyện tập")
            ax2.set_xlabel("Lần luyện")
            ax2.set_ylabel("Điểm /10")

            st.pyplot(fig2)

        else:
            st.write("Hãy upload thêm vài file để thấy tiến bộ rõ hơn")

else:
    st.info("Upload file WAV để bắt đầu")
