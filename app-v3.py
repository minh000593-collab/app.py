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

st.info("⚠️ Kết quả chỉ mang tính tham khảo, không phải đánh giá tuyệt đối.")
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
    "Tôi đồng ý cho Debate Coach AI sử dụng audio để phân tích tốc độ nói, cao độ, và số lượng từ đệm trong bài nói tranh biện của mình."
)

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

    zero_crossings = np.sum(np.abs(np.diff(np.sign(y)))) / len(y)
    pitch_stability = 1 - min(zero_crossings, 1)

    return activity_rate, duration, pitch_stability

# ======================
# RUBRIC
# ======================
def classify_speed(rate):
    if 80 <= rate <= 140:
        return "Tốc độ ổn định 👍", 2
    elif rate > 140:
        return "Bạn đang nói hơi nhanh ⚠️, hãy thử chậm lại một chút để người nghe theo kịp nhé.", 1
    else:
        return "Bạn đang nói hơi chậm ⚠️, thử tăng nhịp lên để bài nói tự nhiên hơn nhé.", 1

def classify_filler():
    filler_count = 5
    if filler_count <= 3:
        return "Bạn sử dụng khá ít từ đệm 👍", 2
    elif filler_count <= 7:
    return (
        "Bạn có sử dụng từ đệm ⚠️ (như 'ừ', 'à', 'kiểu như')",
        1,
        "Hãy thử dừng 1–2 giây thay vì dùng từ đệm khi cần suy nghĩ nha."
    )

else:
    return (
        "Bạn sử dụng từ đệm khá nhiều ⛔ (ảnh hưởng độ trôi chảy của bài nói)",
        0,
        "Hãy luyện nói chậm lại và tập thay từ đệm bằng các khoảng dừng ngắn nha."

def classify_pitch(stability):
    if stability >= 0.7:
        return "Giọng của bạn khá ổn định 👍 Giữ tốc độ nói này là tốt rồi. Bạn có thể thử nhấn nhá nhiều hơn để tăng tính biểu cảm nhé.", 2
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
            "Hãy thử nói chậm lại một chút và giữ hơi đều hơn giữa các câu."
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

    y, sr, err = load_audio_safe(audio_file)

    if err:
        st.error(err)
    else:

        st.subheader("📈 Waveform")
        fig, ax = plt.subplots()
        ax.plot(y)
        st.pyplot(fig)

        with st.spinner("Đang phân tích..."):

            activity_rate, silence_ratio, duration, pitch = extract_features(y, sr)

            speed_label, speed_score = classify_speed(activity_rate)
            silence_label, silence_score = classify_silence(silence_ratio)
            filler_label, filler_score = classify_filler()
            pitch_label, pitch_score = classify_pitch(pitch)

            total = speed_score + silence_score + filler_score + pitch_score

            # scale về /10
            score_10 = (total / 8) * 10

            overall = overall_label(total)

            # ======================
            # STORE PROGRESS
            # ======================
            st.session_state.scores.append(score_10)

        # ======================
        # RESULTS
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
        # PROGRESS SECTION (NEW)
        # ======================
        if len(st.session_state.scores) >= 2:

            st.subheader("📈 Sự tiến bộ của bạn")

            fig2, ax2 = plt.subplots()
            ax2.plot(range(1, len(st.session_state.scores) + 1),
                     st.session_state.scores, marker="o")
            ax2.set_title("Progress score ( /10 )")
            ax2.set_xlabel("Lần upload")
            ax2.set_ylabel("Điểm")

            st.pyplot(fig2)

            # nhận xét xu hướng
            if st.session_state.scores[-1] > st.session_state.scores[-2]:
                st.success("🔥 Bạn đang tiến bộ rõ rệt. Hãy giữ nhịp độ này và tiếp tục luyện tập!")
            elif st.session_state.scores[-1] == st.session_state.scores[-2]:
                st.info("🙂 Kĩ năng nói của bạn rất ổn định.")
            else:
                st.warning("💡 Hãy duy trì luyện tập đều đặn nhé!")

else:
    st.info("Upload file WAV để bắt đầu 🎤")
