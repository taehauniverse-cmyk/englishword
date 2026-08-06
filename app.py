import streamlit as st
import pandas as pd
import random
import datetime
import json
import io
import streamlit.components.v1 as components
from gtts import gTTS

# --- 페이지 설정 ---
st.set_page_config(page_title="15일 완성 3000 단어장", page_icon="⚡", layout="centered")

# --- 원어민 발음 (gTTS) ---
@st.cache_data
def get_audio_bytes(text):
    try:
        tts = gTTS(text=text, lang='en')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except:
        return None

# --- 데이터 로드 ---
@st.cache_data
def load_vocab_data():
    try:
        return pd.read_excel("English_3000_Words.xlsx")
    except Exception:
        st.error("⚠️ 'English_3000_Words.xlsx' 파일을 찾을 수 없습니다.")
        return None

df = load_vocab_data()

# --- 사이드바: 실시간 접속 시간 타이머 (이탈 시 자동 정지) ---
with st.sidebar:
    st.subheader("⏱️ 순수 학습 시간")
    timer_html = """
    <div style="
        background: #0f172a;
        color: #38bdf8;
        padding: 12px;
        border-radius: 12px;
        font-size: 14px;
        font-weight: bold;
        text-align: center;
        border: 1px solid #38bdf8;
        font-family: system-ui, -apple-system, sans-serif;
    ">
        현재 세션 접속<br>
        <span id="timer-display" style="font-size: 22px; color: #ffffff;">00:00:00</span>
    </div>

    <script>
    (function() {
        let activeSeconds = parseInt(localStorage.getItem('user_active_seconds') || '0');
        let isVisible = !document.hidden;

        document.addEventListener('visibilitychange', function() {
            isVisible = !document.hidden;
        });

        setInterval(function() {
            if (isVisible) {
                activeSeconds++;
                localStorage.setItem('user_active_seconds', activeSeconds);
                
                let hrs = String(Math.floor(activeSeconds / 3600)).padStart(2, '0');
                let mins = String(Math.floor((activeSeconds % 3600) / 60)).padStart(2, '0');
                let secs = String(activeSeconds % 60).padStart(2, '0');
                
                let display = document.getElementById('timer-display');
                if (display) {
                    display.innerText = hrs + ":" + mins + ":" + secs;
                }
            }
        }, 1000);
    })();
    </script>
    """
    components.html(timer_html, height=85)

# --- 사용자 상태 초기화 ---
if "user_data" not in st.session_state:
    st.session_state.user_data = {
        "start_date": str(datetime.date.today()),
        "today_completed": 0,
        "words": {}
    }

if df is not None and not st.session_state.user_data["words"]:
    today_str = str(datetime.date.today())
    for _, row in df.iterrows():
        st.session_state.user_data["words"][row["영단어"]] = {
            "box": 1,
            "next_review": today_str,
            "wrong_cnt": 0
        }

if "current_batch" not in st.session_state:
    st.session_state.current_batch = []
if "batch_index" not in st.session_state:
    st.session_state.batch_index = 0
if "q_type" not in st.session_state:
    st.session_state.q_type = "ENG_TO_KOR"
if "q_options" not in st.session_state:
    st.session_state.q_options = []

# --- 사이드바 통계 및 백업 ---
st.sidebar.title("📊 15일 플랜 달성률")

if df is not None:
    total_words = len(df)
    mastered = sum(1 for v in st.session_state.user_data["words"].values() if v["box"] >= 4)
    pct = (mastered / total_words) * 100
    
    st.sidebar.metric("15일 전체 목표 달성률", f"{pct:.1f}%", f"{mastered} / {total_words} 단어")
    st.sidebar.progress(pct / 100)
    
    today_done = st.session_state.user_data.get("today_completed", 0)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 오늘 하루 목표치")
    st.sidebar.write(f"목표: **200단어** / 완료: **{today_done}단어**")
    st.sidebar.progress(min(1.0, today_done / 200))

st.sidebar.markdown("---")
st.sidebar.subheader("💾 진도 데이터 백업")
user_json = json.dumps(st.session_state.user_data, ensure_ascii=False, indent=2)
st.sidebar.download_button("📥 백업 저장", user_json, f"vocab_15days_{datetime.date.today()}.json", "application/json")

uploaded_file = st.sidebar.file_uploader("📤 백업 복원", type=["json"])
if uploaded_file is not None:
    st.session_state.user_data = json.load(uploaded_file)
    st.sidebar.success("복원되었습니다!")

# --- 메인 인터페이스 ---
st.title("⚡ 15일 완성 3,000 영단어")

if df is None:
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔥 오늘의 15단어 세션", "⚡ 10분 복습 퀴즈", "🗂️ 오답 노트 & 검색"])

def setup_question(target_word, mode):
    correct_row = df[df["영단어"] == target_word].iloc[0]
    if mode == "ENG_TO_KOR":
        correct_ans = correct_row["뜻"]
        others = df[df["뜻"] != correct_ans]["뜻"].sample(3).tolist()
    else:
        correct_ans = target_word
        others = df[df["영단어"] != target_word]["영단어"].sample(3).tolist()
    
    options = [correct_ans] + others
    random.shuffle(options)
    st.session_state.q_options = options
    st.session_state.q_correct = correct_ans

# ==========================================
# TAB 1: 15단어 세션 (예문 + 한글 해석)
# ==========================================
with tab1:
    today_str = str(datetime.date.today())
    
    if not st.session_state.current_batch:
        due_words = [
            w for w, data in st.session_state.user_data["words"].items()
            if data["next_review"] <= today_str and data["box"] < 5
        ]
        if due_words:
            st.session_state.current_batch = due_words[:15]
            st.session_state.batch_index = 0
            st.session_state.q_type = random.choice(["ENG_TO_KOR", "KOR_TO_ENG"])
            setup_question(st.session_state.current_batch[0], st.session_state.q_type)

    if st.session_state.current_batch:
        idx = st.session_state.batch_index
        if idx < len(st.session_state.current_batch):
            curr_word = st.session_state.current_batch[idx]
            word_info = df[df["영단어"] == curr_word].iloc[0]
            
            st.caption(f"15단어 세션 진행 중: {idx + 1} / {len(st.session_state.current_batch)}")
            st.progress((idx + 1) / len(st.session_state.current_batch))
            
            audio_bytes = get_audio_bytes(curr_word)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")

            # 예문 및 한글 해석 함께 표시
            ex_trans = word_info.get("예문_번역", "")
            st.info(f"💡 **예문:** {word_info['예문']}\n\n💬 **해석:** {ex_trans}")
            
            if st.session_state.q_type == "ENG_TO_KOR":
                st.markdown(f"### 🔤 단어: **{curr_word}** `{word_info['발음기호']}`")
                st.write("👉 올바른 **한글 뜻**을 선택하세요:")
            else:
                st.markdown(f"### 🇰🇷 뜻: **{word_info['뜻']}** (`{word_info['품사']}`)")
                st.write("👉 알맞은 **영어 단어**를 선택하세요:")
                
            cols = st.columns(2)
            for i, option in enumerate(st.session_state.q_options):
                col = cols[i % 2]
                if col.button(option, key=f"opt_{idx}_{i}", use_container_width=True):
                    if option == st.session_state.q_correct:
                        st.success("🎉 정답입니다!")
                        curr_box = st.session_state.user_data["words"][curr_word]["box"]
                        st.session_state.user_data["words"][curr_word]["box"] = min(5, curr_box + 1)
                        st.session_state.user_data["words"][curr_word]["next_review"] = str(datetime.date.today() + datetime.timedelta(days=curr_box * 2))
                        st.session_state.user_data["today_completed"] = st.session_state.user_data.get("today_completed", 0) + 1
                        
                        st.session_state.batch_index += 1
                        if st.session_state.batch_index < len(st.session_state.current_batch):
                            next_w = st.session_state.current_batch[st.session_state.batch_index]
                            st.session_state.q_type = random.choice(["ENG_TO_KOR", "KOR_TO_ENG"])
                            setup_question(next_w, st.session_state.q_type)
                        st.rerun()
                    else:
                        st.error(f"❌ 틀렸습니다! (정답: {st.session_state.q_correct}) -> 오답노트에 자동 추가됩니다.")
                        st.session_state.user_data["words"][curr_word]["box"] = 1
                        st.session_state.user_data["words"][curr_word]["wrong_cnt"] += 1
                        st.session_state.current_batch.append(curr_word)
                        
                        st.session_state.batch_index += 1
                        if st.session_state.batch_index < len(st.session_state.current_batch):
                            next_w = st.session_state.current_batch[st.session_state.batch_index]
                            st.session_state.q_type = random.choice(["ENG_TO_KOR", "KOR_TO_ENG"])
                            setup_question(next_w, st.session_state.q_type)
                        st.rerun()
        else:
            st.balloons()
            st.success("🎉 15단어 미니 세션을 모두 마쳤습니다!")
            if st.button("다음 15단어 세션 시작하기", use_container_width=True):
                st.session_state.current_batch = []
                st.rerun()
    else:
        st.info("🎈 오늘 배정된 복습 단어를 모두 마쳤습니다!")

# ==========================================
# TAB 2: 10분 스피드 복습 퀴즈
# ==========================================
with tab2:
    st.subheader("⚡ 10분 스피드 복습 퀴즈")
    st.caption("주요 단어를 복습하고 툭툭 풀어봅니다.")
    
    if "rev_word" not in st.session_state or st.button("새 복습 문제 불러오기"):
        review_candidates = [
            w for w, data in st.session_state.user_data["words"].items()
            if data["box"] <= 2 or data["wrong_cnt"] > 0
        ]
        if not review_candidates:
            review_candidates = list(st.session_state.user_data["words"].keys())
            
        st.session_state.rev_word = random.choice(review_candidates)
        r_info = df[df["영단어"] == st.session_state.rev_word].iloc[0]
        
        others = df[df["뜻"] != r_info["뜻"]]["뜻"].sample(3).tolist()
        opts = [r_info["뜻"]] + others
        random.shuffle(opts)
        st.session_state.rev_opts = opts
        st.session_state.rev_correct = r_info["뜻"]

    r_target = st.session_state.rev_word
    r_row = df[df["영단어"] == r_target].iloc[0]
    
    st.markdown(f"### 🔤 **{r_target}** `{r_row['발음기호']}`")
    st.info(f"💡 예문: {r_row['예문']}\n\n💬 해석: {r_row.get('예문_번역', '')}")
    
    selected = st.radio("알맞은 뜻을 고르세요:", st.session_state.rev_opts, key="rev_radio")
    
    if st.button("정답 확인", use_container_width=True):
        if selected == st.session_state.rev_correct:
            st.success("🎉 정답입니다!")
            st.session_state.user_data["words"][r_target]["box"] = min(5, st.session_state.user_data["words"][r_target]["box"] + 1)
        else:
            st.error(f"❌ 틀렸습니다! 정답: **{st.session_state.rev_correct}**")
            st.session_state.user_data["words"][r_target]["box"] = 1
            st.session_state.user_data["words"][r_target]["wrong_cnt"] += 1

# ==========================================
# TAB 3: 오답 노트
# ==========================================
with tab3:
    st.subheader("🗂️ 오답 노트")
    
    wrong_sorted = sorted(
        st.session_state.user_data["words"].items(),
        key=lambda x: x[1]["wrong_cnt"],
        reverse=True
    )
    
    wrong_list = []
    for word, stats in wrong_sorted:
        if stats["wrong_cnt"] > 0 or stats["box"] == 1:
            row = df[df["영단어"] == word].iloc[0]
            wrong_list.append({
                "영단어": word,
                "뜻": row["뜻"],
                "품사": row["품사"],
                "암기단계": f"Box {stats['box']}",
                "틀린 횟수": stats["wrong_cnt"],
                "다음 복습일": stats["next_review"]
            })
            
    if wrong_list:
        st.dataframe(pd.DataFrame(wrong_list), use_container_width=True)
    else:
        st.success("🎉 현재 오답 노트가 비어있습니다!")
