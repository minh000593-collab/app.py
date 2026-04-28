import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import soundfile as sf
import whisper
import librosa
import re

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Debate Coach AI", layout="centered")

st.title("🎤 Debate Coach AI")
st.write("Phân tích kỹ năng nói từ file ghi âm (.wav)")

st.info("⚠️ Kết quả chỉ mang tính tham khảo, không phải đánh giá tuyệt đối.")
st.divider()

# ======================
# SESSION STATE
# ======================
if "scores" not in st.session_state:
    st.session_state.scores = []

# ======================
# CONSENT
# ======================
consent = st.checkbox(
    "Tôi đồng ý cho Debate Coach AI sử dụng audio để phân tích tốc độ nói, cao độ, và số lượng từ đệm trong bài nói tranh biện của mình."
)

audio_file = st.file_uploader("Upload file ghi âm (.wav)", type=["wav"])

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# ======================
# LOAD AUDIO SAFE
# ======================
def load_audio_safe(file):
    file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.read())
        path = tmp.name

    y, sr = sf.read(path)

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    return y, sr, path

# ======================
# TRANSCRIBE (AI)
# ======================
def transcribe(path):
    result = model.transcribe(path)
    return result["text"]

# ======================
# FEATURE EXTRACTION
# ======================
def extract_features(y, sr):
    duration = len(y) / sr

    energy = np.abs(y)
    energy_mean = np.mean(energy) + 1e-6

    activity = np.sum(energy > energy_mean)
    activity_rate = activity / max(duration, 1e-6)

    zero_crossings = np.sum(np.abs(np.diff(np.sign(y)))) / len(y)
    pitch_stability = 1 - min(zero_crossings, 1)

    return activity_rate, duration, pitch_stability

# ======================
# RUBRIC (GIỮ NGUYÊN TEXT)
# ======================
def classify_speed(rate):
    if 80 <= rate <= 140:
        return "Tốc độ ổn định 👍", 2
    elif rate > 140:
        return "Bạn đang nói hơi nhanh ⚠️, hãy thử chậm lại một chút để người nghe theo kịp nhé.", 1
    else:
        return "Bạn đang nói hơi chậm ⚠️, thử tăng nhịp lên để bài nói tự nhiên hơn nhé.", 1


def classify_filler(text):
    filler_words = ["um", "uh", "like", "à", "ừ", "kiểu", "ờ"]
    filler_count = sum(text.lower().count(w) for w in filler_words)

    if filler_count <= 3:
        return (
            "Bạn sử dụng khá ít từ đệm 👍",
            2,
            "Giữ phong độ này nhé, rất tốt cho sự trôi chảy khi nói."
        )

    elif filler_count <= 7:
        return (
            "Bạn có sử dụng từ đệm ⚠️ (như 'ừ', 'à', 'kiểu như')",
            1,
            "Hãy thử dừng 1–2 giây thay vì dùng từ đệm khi cần suy nghĩ."
        )

    else:
        return (
            "Bạn sử dụng từ đệm khá nhiều ⛔",
            0,
            "Luyện nói chậm lại và thay từ đệm bằng khoảng dừng ngắn."
        )


def classify_pitch(stability):
    if stability >= 0.7:
        return (
            "Giọng của bạn khá ổn định 👍 Giữ tốc độ nói này là tốt rồi. Bạn có thể thử nhấn nhá nhiều hơn để tăng tính biểu cảm nhé.",
            2,
            "Giữ phong độ này và thử thêm nhấn nhá để bài nói thuyết phục hơn."
        )

    elif stability >= 0.4:
        return (
            "Giọng của bạn hơi dao động ⚠️",
            1,
            "Hãy thử nói chậm lại một chút và giữ hơi đều hơn giữa các câu."
        )

    else:
        return (
            "Giọng của bạn chưa ổn định ⛔",
            0,
            "Hãy luyện nói chậm, rõ từng ý và kiểm soát hơi thở tốt hơn."
        )


def overall_label(total):
    if total >= 6:
        return "Kĩ năng nói của bạn ở mức tốt 👍"
    elif total >= 3:
        return "Kĩ năng nói của bạn ở mức trung bình ⚠️"
    else:
        return "Kĩ năng nói của bạn cần luyện tập thêm ⛔"

# ======================
# MAIN
# ======================
if audio_file is not None:

    if not consent:
        st.warning("Bạn cần đồng ý trước khi phân tích audio.")
        st.stop()

    st.success("Upload thành công!")

    y, sr, path = load_audio_safe(audio_file)
    duration = len(y) / sr

    st.subheader("📈 Waveform")
    fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)

    with st.spinner("Đang phân tích..."):

        text = transcribe(path)

        activity_rate, duration, pitch = extract_features(y, sr)

        speed_label, speed_score = classify_speed(activity_rate)
        filler_label, filler_score, filler_suggestion = classify_filler(text)
        pitch_label, pitch_score, pitch_suggestion = classify_pitch(pitch)

        total = speed_score + filler_score + pitch_score

        score_10 = (total / 6) * 10

        overall = overall_label(total)

        st.session_state.scores.append(score_10)

    # ======================
    # RESULTS (GIỮ NGUYÊN TEXT)
    # ======================
    st.subheader("📊 Kết quả phân tích bài nói của bạn")
    st.write(f"⏱ Thời lượng: {duration:.2f}s")
    st.write(f"🎯 Điểm: {score_10:.2f}/10")

    st.subheader("🏁 Các tiêu chí đánh giá")
    st.write("🗣 Tốc độ:", speed_label)
    st.write("💬 Từ đệm:", filler_label)
    st.write("🎼 Cao độ:", pitch_label)
    st.write("🎯 Tổng thể:", overall)

    # ======================
    # PROGRESS (UNCHANGED)
    # ======================
    if len(st.session_state.scores) >= 2:

        st.subheader("📈 Sự tiến bộ của bạn")

        x = np.arange(1, len(st.session_state.scores) + 1, dtype=int)

        fig2, ax2 = plt.subplots()
        ax2.plot(x, st.session_state.scores, marker="o")

        ax2.set_title("Progress score ( /10 )")
        ax2.set_xlabel("Lần upload")
        ax2.set_ylabel("Điểm")

        ax2.set_xticks(x)

        st.pyplot(fig2)

        if st.session_state.scores[-1] > st.session_state.scores[-2]:
            st.success("🔥 Bạn đang tiến bộ rõ rệt. Hãy giữ nhịp độ này và tiếp tục luyện tập!")
        elif st.session_state.scores[-1] == st.session_state.scores[-2]:
            st.info("🙂 Kĩ năng nói của bạn rất ổn định.")
        else:
            st.warning("💡 Hãy duy trì luyện tập đều đặn nhé!")