import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import soundfile as sf

# ======================
# CONFIG (UI ONLY)
# ======================
st.set_page_config(page_title="Debate Coach AI", layout="centered")

st.markdown("""
<div style="text-align:center;">
    <h1>🎤 Debate Coach AI</h1>
</div>
""", unsafe_allow_html=True)

st.write("Phân tích kỹ năng nói từ file ghi âm (.wav)")

st.info("⚠️ Kết quả chỉ mang tính tham khảo, không phải đánh giá tuyệt đối.")
st.divider()

# ======================
# SESSION STATE
# ======================
if "scores" not in st.session_state:
    st.session_state.scores = []

# ======================
# INPUT SECTION (UI WRAP)
# ======================
st.subheader("📥 Upload")

consent = st.checkbox(
    "Tôi đồng ý cho Debate Coach AI sử dụng audio để phân tích tốc độ nói, cao độ, và số lượng từ đệm trong bài nói tranh biện của mình."
)

audio_file = st.file_uploader("Upload file ghi âm (.wav)", type=["wav"])

st.divider()

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
# FEATURE EXTRACTION (NO SILENCE)
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

        # ======================
        # WAVEFORM CARD (UI ONLY)
        # ======================
        st.subheader("📈 Waveform")

        fig, ax = plt.subplots()
        ax.plot(y)
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig)

        # ======================
        # ANALYSIS
        # ======================
        with st.spinner("Đang phân tích..."):

            activity_rate, duration, pitch = extract_features(y, sr)

            speed_label, speed_score = classify_speed(activity_rate)
            filler_label, filler_score, filler_suggestion = classify_filler()
            pitch_label, pitch_score, pitch_suggestion = classify_pitch(pitch)

            total = speed_score + filler_score + pitch_score

            score_10 = (total / 6) * 10

            overall = overall_label(total)

            st.session_state.scores.append(score_10)

        # ======================
        # RESULTS (UI UPGRADE ONLY)
        # ======================
        st.subheader("📊 Kết quả phân tích bài nói của bạn")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("⏱ Thời lượng", f"{duration:.2f}s")
            st.metric("🎯 Điểm", f"{score_10:.2f}/10")

        with col2:
            st.metric("🗣 Tốc độ", speed_label)
            st.metric("💬 Từ đệm", filler_label)

        st.metric("🎼 Cao độ", pitch_label)
        st.success(f"🏁 Tổng thể: {overall}")

        st.divider()

        # ======================
        # PROGRESS (UI ONLY)
        # ======================
        if len(st.session_state.scores) >= 2:

            st.subheader("📈 Sự tiến bộ của bạn")

            x = np.arange(1, len(st.session_state.scores) + 1, dtype=int)

            fig2, ax2 = plt.subplots()
            ax2.plot(x, st.session_state.scores, marker="o")

            ax2.set_xticks(x)

            st.pyplot(fig2)

            if st.session_state.scores[-1] > st.session_state.scores[-2]:
                st.success("🔥 Bạn đang tiến bộ rõ rệt. Hãy giữ nhịp độ này và tiếp tục luyện tập!")
            elif st.session_state.scores[-1] == st.session_state.scores[-2]:
                st.info("🙂 Kĩ năng nói của bạn rất ổn định.")
            else:
                st.warning("💡 Hãy duy trì luyện tập đều đặn nhé!")
