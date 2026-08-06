import streamlit as st
import pandas as pd
import random
import datetime
import json
import io
from gtts import gTTS

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="3000 단어 10일 완성", page_icon="⚡", layout="centered")

# --- 원어민 발음 생성 함수 (gTTS) ---
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

# --- 데이터로드 함수 ---
@st.cache_data
def load_vocab_data():
    try:
        df = pd.read_excel("English_3000_Words.xlsx")
        return df
    except Exception:
        st.error("⚠️ 'English_3000_Words.xlsx' 파일을 찾을 수 없습니다. 앱 폴더에 엑셀 파일을 넣어주세요.")
        return None

df = load_vocab_data()

# --- 상태 초기화 ---
if "user_data" not in st.session_state:
    st.session_state.user_data = {
        "streak": 1,
        "last_login": str(datetime.date.today()),
        "words": {} # word: {"box": 1, "next_review": str, "wrong_cnt": 0}
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
if "mode" not in st.session_state:
    st.session_state.mode = "flashcard" # flashcard or quiz
if "show_meaning" not in st.session_state:
    st.session_state.show_meaning = False

# --- 사이드바: 통계 및 저장/복원 ---
st.sidebar.title("📊 학습 관리자")

if df is not None:
    total_cnt = len(df)
    mastered_cnt = sum(1 for v in st.session_state.user_data["words"].values() if v["box"] >= 4)
    progress_pct = (mastered_cnt / total_cnt) * 100
    
    st.sidebar.metric("총 완맹 암기율", f"{progress_pct:.1f}%", f"{mastered_cnt}/{total_cnt} 단어")
    st.sidebar.progress(progress_pct / 100)

st.sidebar.markdown("---")
st.sidebar.subheader("💾 진도 보존 (데이터 백업)")

# JSON 데이터 다운로드
user_data_json = json.dumps(st.session_state.user_data, ensure_ascii=False, indent=2)
st.sidebar.download_button(
    label="📥 백업 파일 다운로드",
    data=user_data_json,
    file_name=f"vocab_backup_{datetime.date.today()}.json",
    mime="application/json"
)

# JSON 데이터 업로드
uploaded_file = st.sidebar.file_uploader("📤 백업 파일 복원하기", type=["json"])
if uploaded_file is not None:
    loaded_data = json.load(uploaded_file)
    st.session_state.user_data = loaded_data
    st.sidebar.success("성공적으로 복원되었습니다!")

# --- 메인 화면 ---
st.title("⚡ 10일 완성 3,000 영단어")

if df is None:
    st.stop()

# 탭 구성
tab1, tab2, tab3 = st.tabs(["🔥 오늘의 15단어 세션", "🎯 4지선다 퀴즈", "🗂️ 전체/오답 목록"])

# ==========================================
# TAB 1: 오늘의 15단어 마이크로 세션
# ==========================================
with tab1:
    today_str = str(datetime.date.today())
    
    # 세션이 비어있으면 15개 추출
    if not st.session_state.current_batch:
        due_words = [
            w for w, data in st.session_state.user_data["words"].items()
            if data["next_review"] <= today_str and data["box"] < 5
        ]
        if due_words:
            # 복습 우선, 최대 15개
            st.session_state.current_batch = due_words[:15]
            st.session_state.batch_index = 0
            st.session_state.show_meaning = False

    if st.session_state.current_batch:
        idx = st.session_state.batch_index
        if idx < len(st.session_state.current_batch):
            curr_word = st.session_state.current_batch[idx]
            word_info = df[df["영단어"] == curr_word].iloc[0]
            
            st.caption(f"미니 세션 진행률: {idx + 1} / {len(st.session_state.current_batch)}")
            st.progress((idx + 1) / len(st.session_state.current_batch))
            
            # 단어 카드 출력
            st.markdown(f"### 🔤 **{word_info['영단어']}**")
            st.write(f"품사: `{word_info['품사']}` | 발음: `{word_info['발음기호']}`")
            
            # 발음 듣기 버튼
            audio_bytes = get_audio_bytes(word_info['영단어'])
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")

            st.markdown("---")
            
            if not st.session_state.show_meaning:
                if st.button("👁️ 뜻 확인하기", use_container_width=True):
                    st.session_state.show_meaning = True
                    st.rerun()
            else:
                st.success(f"**뜻:** {word_info['뜻']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("⭕ 알고 있음 (다음 단계)", use_container_width=True):
                        # Box 단계 상승 및 다음 복습일 계산
                        curr_box = st.session_state.user_data["words"][curr_word]["box"]
                        new_box = min(5, curr_box + 1)
                        next_days = [0, 1, 2, 4, 7, 15][new_box]
                        next_date = str(datetime.date.today() + datetime.timedelta(days=next_days))
                        
                        st.session_state.user_data["words"][curr_word]["box"] = new_box
                        st.session_state.user_data["words"][curr_word]["next_review"] = next_date
                        
                        st.session_state.batch_index += 1
                        st.session_state.show_meaning = False
                        st.rerun()
                with col2:
                    if st.button("❌ 헷갈림/틀림 (재배치)", use_container_width=True):
                        # 틀리면 Box 1로 초기화 및 해당 세션 뒤쪽에 재출제 추가
                        st.session_state.user_data["words"][curr_word]["box"] = 1
                        st.session_state.user_data["words"][curr_word]["wrong_cnt"] += 1
                        st.session_state.current_batch.append(curr_word) # 세션 뒤에 추가
                        
                        st.session_state.batch_index += 1
                        st.session_state.show_meaning = False
                        st.rerun()
        else:
            st.balloons()
            st.success("🎉 축하합니다! 이번 15단어 마이크로 세션을 완료했습니다.")
            if st.button("다음 15단어 시작하기", use_container_width=True):
                st.session_state.current_batch = []
                st.rerun()
    else:
        st.info("🎈 오늘 복습할 단어를 모두 완료했습니다! '4지선다 퀴즈' 탭에서 실력을 점검해 보세요.")

# ==========================================
# TAB 2: 4지선다 객관식 퀴즈 (Active Recall)
# ==========================================
with tab2:
    st.subheader("🎯 능동적 회상 4지선다 퀴즈")
    
    # 퀴즈용 단어 1개 추출
    if "quiz_word" not in st.session_state or st.button("새 퀴즈 불러오기"):
        all_words = list(st.session_state.user_data["words"].keys())
        st.session_state.quiz_word = random.choice(all_words)
        
        correct_info = df[df["영단어"] == st.session_state.quiz_word].iloc[0]
        correct_meaning = correct_info["뜻"]
        
        # 오답 보기 3개 추출
        other_meanings = df[df["뜻"] != correct_meaning]["뜻"].sample(3).tolist()
        options = [correct_meaning] + other_meanings
        random.shuffle(options)
        
        st.session_state.quiz_options = options
        st.session_state.quiz_correct = correct_meaning
        st.session_state.quiz_answered = False

    q_word = st.session_state.quiz_word
    q_info = df[df["영단어"] == q_word].iloc[0]
    
    st.markdown(f"### 단어: **{q_word}**  `{q_info['발음기호']}`")
    
    user_choice = st.radio("올바른 뜻을 선택하세요:", st.session_state.quiz_options)
    
    if st.button("정답 확인", use_container_width=True):
        st.session_state.quiz_answered = True
        if user_choice == st.session_state.quiz_correct:
            st.success("🎉 정답입니다!")
            # Box 단계 약간 상승
            curr_box = st.session_state.user_data["words"][q_word]["box"]
            st.session_state.user_data["words"][q_word]["box"] = min(5, curr_box + 1)
        else:
            st.error(f"❌ 틀렸습니다. 정답은 **'{st.session_state.quiz_correct}'** 입니다.")
            st.session_state.user_data["words"][q_word]["box"] = 1

# ==========================================
# TAB 3: 오답 노트 및 검색
# ==========================================
with tab3:
    st.subheader("🗂️ 단어 검색 및 오답 관리")
    
    search_term = st.text_input("단어 또는 뜻 검색", "")
    
    # 틀린 횟수가 많은 순으로 단어 정렬
    sorted_words = sorted(
        st.session_state.user_data["words"].items(),
        key=lambda x: x[1]["wrong_cnt"],
        reverse=True
    )
    
    list_data = []
    for word, stats in sorted_words:
        row = df[df["영단어"] == word].iloc[0]
        if search_term.lower() in word.lower() or search_term in row["뜻"]:
            list_data.append({
                "영단어": word,
                "뜻": row["뜻"],
                "품사": row["품사"],
                "암기단계": f"Box {stats['box']}",
                "틀린 횟수": stats["wrong_cnt"]
            })
            
    st.dataframe(pd.DataFrame(list_data), use_container_width=True)
