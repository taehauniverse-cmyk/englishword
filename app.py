import streamlit as st
import pandas as pd
import re
from collections import Counter
import random
import nltk
import os
import json
import time
import io
import eng_to_ipa as ipa
from gtts import gTTS

# --- 1. NLTK 데이터 초기화 ---
@st.cache_resource
def init_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

init_nltk()

# --- 2. 원어민 발음 음성 생성 (gTTS 캐싱) ---
@st.cache_data
def get_audio_bytes(text_to_speak):
    if not text_to_speak or text_to_speak in ["예문 없음", "대본 예문 없음", "-", ""]:
        return None
    try:
        tts = gTTS(text=text_to_speak, lang='en', tld='com')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        return None

# --- 3. 학습 시간 측정 로직 (5분 이내 조작 누적) ---
now_time = time.time()
today_date_str = time.strftime("%Y-%m-%d")

if 'study_date' not in st.session_state or st.session_state.study_date != today_date_str:
    st.session_state.study_date = today_date_str
    st.session_state.today_study_sec = 0
    st.session_state.last_ping_time = now_time
else:
    time_diff = now_time - st.session_state.get('last_ping_time', now_time)
    if time_diff < 300:  # 5분 이내 조작만 학습 시간 카운트
        st.session_state.today_study_sec += time_diff
    st.session_state.last_ping_time = now_time

total_sec = int(st.session_state.today_study_sec)
hrs = total_sec // 3600
mins = (total_sec % 3600) // 60
secs = total_sec % 60
study_time_display = f"{hrs}시간 {mins}분 {secs}초" if hrs > 0 else f"{mins}분 {secs}초"

# --- 4. 기본 3,000 단어장 데이터 로드 및 전처리 ---
VOCAB_FILE = "English_3000_Words.xlsx"
USER_DATA_FILE = "user_progress_3000.json"

@st.cache_data
def load_base_vocab():
    if os.path.exists(VOCAB_FILE):
        df = pd.read_excel(VOCAB_FILE)
        
        # 컬럼 이름 맞춤 처리
        rename_map = {
            "영단어": "영어 단어",
            "뜻": "대표 뜻",
            "예문_번역": "예문 번역"
        }
        df.rename(columns=rename_map, inplace=True)
        
        # 필수 컬럼 보장
        for col in ["발음기호", "품사", "예문", "예문 번역"]:
            if col not in df.columns:
                df[col] = "-" if col == "발음기호" else ("기타" if col == "품사" else "예문 없음")
                
        for col, default_val in [("숙달도", 1), ("틀린 횟수", 0), ("맞춘 횟수", 0)]:
            if col not in df.columns:
                df[col] = default_val
                
        return df
    else:
        st.error(f"⚠️ '{VOCAB_FILE}' 파일을 찾을 수 없습니다. 저장소에 엑셀 파일을 올려주세요.")
        return None

df_base = load_base_vocab()

# --- 5. 사용자 진도 데이터 저장/복원 함수 ---
def load_user_progress():
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_user_progress(progress_data):
    with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)

if "user_progress" not in st.session_state:
    st.session_state.user_progress = load_user_progress()

if df_base is not None:
    current_df = df_base.copy()
    for idx, row in current_df.iterrows():
        word = str(row["영어 단어"]).strip()
        if word in st.session_state.user_progress:
            saved_stats = st.session_state.user_progress[word]
            current_df.at[idx, "숙달도"] = saved_stats.get("숙달도", 1)
            current_df.at[idx, "맞춘 횟수"] = saved_stats.get("맞춘 횟수", 0)
            current_df.at[idx, "틀린 횟수"] = saved_stats.get("틀린 횟수", 0)

def sync_and_save_word(word, level, correct_inc=0, wrong_inc=0):
    real_idx = current_df[current_df['영어 단어'] == word].index[0]
    
    new_mastery = int(current_df.at[real_idx, '숙달도']) if level is None else level
    new_correct = int(current_df.at[real_idx, '맞춘 횟수']) + correct_inc
    new_wrong = int(current_df.at[real_idx, '틀린 횟수']) + wrong_inc
    
    current_df.at[real_idx, '숙달도'] = new_mastery
    current_df.at[real_idx, '맞춘 횟수'] = new_correct
    current_df.at[real_idx, '틀린 횟수'] = new_wrong
    
    st.session_state.user_progress[word] = {
        "숙달도": new_mastery,
        "맞춘 횟수": new_correct,
        "틀린 횟수": new_wrong
    }
    save_user_progress(st.session_state.user_progress)

# --- 6. 텍스트 강조 및 빈칸 처리 헬퍼 함수 ---
def highlight_target_word(sentence, target_word):
    if not sentence or sentence in ["예문 없음", "대본 예문 없음", "-"]:
        return "등록된 예문이 없습니다."
    pattern = re.compile(r'\b(' + re.escape(target_word) + r')\b', re.IGNORECASE)
    return pattern.sub(r'**\1**', sentence)

def make_blank_sentence(sentence, target_word):
    pattern = re.compile(r'\b' + re.escape(target_word) + r'\b', re.IGNORECASE)
    return pattern.sub("✏️ [ ______ ]", sentence)

# --- 7. UI 기본 설정 ---
st.set_page_config(page_title="3000 단어 마스터 프로그램", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
    h1 { font-size: 1.25rem !important; margin-bottom: 0.3rem !important; }
    h2 { font-size: 1.1rem !important; margin-bottom: 0.3rem !important; }
    h3 { font-size: 1.0rem !important; margin-bottom: 0.3rem !important; }
    .stAlert { padding: 0.4rem 0.8rem !important; margin-bottom: 0.4rem !important; }
    blockquote { margin: 0.3rem 0rem !important; padding: 0.4rem 0.8rem !important; font-size: 0.9rem !important; }
    audio { height: 35px !important; margin-bottom: 0.3rem !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("### ⚡ 3,000 영단어 마스터 프로그램")

if df_base is None:
    st.stop()

# --- 8. 사이드바 학습 보드 & 데이터 관리 ---
with st.sidebar:
    st.header("📈 학습 통계 보드")
    
    total_saved = len(current_df)
    total_correct = pd.to_numeric(current_df['맞춘 횟수'], errors='coerce').fillna(0).sum()
    total_wrong = pd.to_numeric(current_df['틀린 횟수'], errors='coerce').fillna(0).sum()
    total_attempts = total_correct + total_wrong
    acc_rate = (total_correct / total_attempts * 100) if total_attempts > 0 else 0.0

    current_df['숙달도'] = pd.to_numeric(current_df['숙달도'], errors='coerce').fillna(1).astype(int)
    mastery_sum = (current_df['숙달도'] - 1).sum()
    max_mastery = total_saved * 4
    learning_completion_rate = (mastery_sum / max_mastery * 100) if total_saved > 0 else 0.0

    st.metric("🎓 전체 종합 완수율", f"{learning_completion_rate:.1f} %")
    st.progress(learning_completion_rate / 100)
    
    st.metric("⏱️ 오늘 학습 시간", study_time_display)
    st.metric("📚 총 수록 단어", f"{total_saved:,} 개")
    st.metric("🎯 총 퀴즈 풀이", f"{int(total_attempts):,} 회")
    st.metric("📊 평균 정답률", f"{acc_rate:.1f} %")
    
    st.divider()
    st.subheader("💾 진도 데이터 수동 백업")
    json_progress = json.dumps(st.session_state.user_progress, ensure_ascii=False, indent=2)
    st.download_button("📥 내 진도 백업 파일 다운로드", json_progress, f"vocab_progress_{today_date_str}.json", "application/json")
    
    uploaded_json = st.file_uploader("📤 백업 파일 복원", type=["json"])
    if uploaded_json is not None:
        try:
            st.session_state.user_progress = json.load(uploaded_json)
            save_user_progress(st.session_state.user_progress)
            st.success("진도 데이터가 성공적으로 복원되었습니다!")
            st.rerun()
        except Exception:
            st.error("올바른 백업 JSON 파일이 아닙니다.")

# --- 9. 메인 학습 탭 구성 ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📚 단어장 사전", 
    "🧠 4지선다 퀴즈", 
    "❌ 오답노트", 
    "⚡ 반응속도 복습",
    "💬 예문 암기"
])

# --- TAB 1: 3000 단어장 사전 ---
with tab1:
    st.caption("✏️ **팁:** 표에서 뜻을 클릭해 직접 수정한 뒤 '저장하기' 버튼을 누르면 저장됩니다.")
    display_df = current_df.copy()
    display_df['숙달도'] = pd.to_numeric(display_df['숙달도'], errors='coerce').fillna(1).astype(int)
    display_df['숙달 상태'] = display_df['숙달도'].apply(lambda x: "⭐"*x)
    
    search = st.text_input("영단어 또는 뜻 검색", "")
    if search:
        display_df = display_df[
            display_df['영어 단어'].astype(str).str.contains(search.lower()) |
            display_df['대표 뜻'].astype(str).str.contains(search)
        ]

    target_cols = [c for c in ["영어 단어", "발음기호", "대표 뜻", "품사", "예문", "예문 번역", "숙달 상태", "맞춘 횟수", "틀린 횟수"] if c in display_df.columns]

    edited_df = st.data_editor(display_df[target_cols], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 수정사항 저장하기", key="save_vocab_btn"):
        for col in ["대표 뜻", "예문", "예문 번역"]:
            if col in edited_df.columns:
                current_df[col] = edited_df[col]
        st.success("성공적으로 저장되었습니다!")

# --- TAB 2: 4지선다 퀴즈 ---
with tab2:
    if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
    if 'q_score' not in st.session_state: st.session_state.q_score = 0
    if 'q_data' not in st.session_state or st.button("🎲 퀴즈 순서 새로 섞기", key="q_reshuffle_btn"):
        st.session_state.q_data = current_df.sample(frac=1).reset_index(drop=True)
        st.session_state.q_idx = 0
        st.session_state.q_score = 0
        st.session_state.q_answered = False
        st.session_state.pop('q_curr_opts', None)

    q_df = st.session_state.q_data
    total_q = len(q_df)

    if st.session_state.q_idx < total_q:
        row = q_df.iloc[st.session_state.q_idx]
        target_word = str(row['영어 단어']).strip()
        correct_ans = str(row.get('대표 뜻', '-')).strip()
        phonetic = row.get('발음기호', '-')
        example_sent = row.get('예문', '')
        example_trans = row.get('예문 번역', '')
        
        st.progress(st.session_state.q_idx / total_q)
        st.info(f"### **{target_word}** `[{row.get('품사', '')}]` `{phonetic}`")
        
        word_audio = get_audio_bytes(target_word)
        if word_audio: st.audio(word_audio, format="audio/mp3")
        
        st.markdown(f"> 💡 **예문:** \"{highlight_target_word(example_sent, target_word)}\"")
        if example_trans and example_trans != "예문 없음":
            st.markdown(f"> 💬 **해석:** \"{example_trans}\"")

        if 'q_curr_opts' not in st.session_state or st.session_state.get('q_opts_word') != target_word:
            all_meanings = list(set(current_df['대표 뜻'].astype(str).str.strip()) - {correct_ans})
            distractors = random.sample(all_meanings, min(3, len(all_meanings)))
            options = distractors + [correct_ans]
            random.shuffle(options)
            options.append("❓ 모르겠음")
            
            st.session_state.q_curr_opts = options
            st.session_state.q_opts_word = target_word

        options = st.session_state.q_curr_opts

        if not st.session_state.get('q_answered', False):
            with st.form(key=f"gen_quiz_form_{st.session_state.q_idx}"):
                user_ans = st.radio("올바른 뜻을 선택하세요:", options)
                submit = st.form_submit_button("정답 제출")
                if submit:
                    st.session_state.q_user_ans = user_ans.strip()
                    st.session_state.q_answered = True
                    
                    if user_ans.strip() == correct_ans:
                        st.session_state.q_score += 1
                        sync_and_save_word(target_word, level=None, correct_inc=1, wrong_inc=0)
                    else:
                        real_lvl = current_df[current_df['영어 단어'] == target_word]['숙달도'].values[0]
                        new_lvl = max(1, real_lvl - 1)
                        sync_and_save_word(target_word, level=new_lvl, correct_inc=0, wrong_inc=1)
                    
                    st.rerun()
        else:
            u_ans = st.session_state.get('q_user_ans', '')
            if u_ans == correct_ans:
                st.success(f"🎉 정답입니다! (**{correct_ans}**)")
            elif u_ans == "❓ 모르겠음":
                st.warning(f"💡 정답은 **'{correct_ans}'** 입니다!")
            else:
                st.error(f"❌ 틀렸습니다! 내가 선택한 답: '{u_ans}' ➔ **정답: '{correct_ans}'**")
            
            if st.button("다음 문제 ➡️", key="q_next_btn"):
                st.session_state.q_idx += 1
                st.session_state.q_answered = False
                st.session_state.pop('q_curr_opts', None)
                st.rerun()
    else:
        st.balloons()
        st.success(f"🏆 전체 단어 퀴즈 완료! 점수: {st.session_state.q_score} / {total_q}")

# --- TAB 3: 오답노트 ---
with tab3:
    current_df['틀린 횟수'] = pd.to_numeric(current_df['틀린 횟수'], errors='coerce').fillna(0).astype(int)
    wrong_df = current_df[current_df['틀린 횟수'] > 0].sort_values(by='틀린 횟수', ascending=False)
    
    if wrong_df.empty:
        st.success("👏 현재 틀린 단어가 없습니다! 완벽합니다.")
    else:
        st.caption(f"총 **{len(wrong_df)}개**의 틀린 단어가 기록되어 있습니다.")
        target_cols_w = [c for c in ["영어 단어", "발음기호", "대표 뜻", "품사", "틀린 횟수"] if c in wrong_df.columns]
        st.dataframe(wrong_df[target_cols_w], use_container_width=True)
        
        st.divider()
        if len(wrong_df) >= 2:
            if 'w_idx' not in st.session_state: st.session_state.w_idx = 0
            
            if st.session_state.w_idx < len(wrong_df):
                w_row = wrong_df.iloc[st.session_state.w_idx]
                target_w = str(w_row['영어 단어']).strip()
                correct_w = str(w_row.get('대표 뜻', '-')).strip()
                phonetic_w = w_row.get('발음기호', '-')
                example_w = w_row.get('예문', '')
                example_trans_w = w_row.get('예문 번역', '')
                
                st.info(f"### **{target_w}** `{phonetic_w}` (누적 틀린 횟수: {w_row['틀린 횟수']}회)")
                w_audio = get_audio_bytes(target_w)
                if w_audio: st.audio(w_audio, format="audio/mp3")

                st.markdown(f"> 💡 **예문:** \"{highlight_target_word(example_w, target_w)}\"")
                if example_trans_w and example_trans_w != "예문 없음":
                    st.markdown(f"> 💬 **해석:** \"{example_trans_w}\"")
                
                if 'w_curr_opts' not in st.session_state or st.session_state.get('w_opts_word') != target_w:
                    all_meanings = list(set(current_df['대표 뜻'].astype(str).str.strip()) - {correct_w})
                    distractors = random.sample(all_meanings, min(3, len(all_meanings)))
                    opts = distractors + [correct_w]
                    random.shuffle(opts)
                    opts.append("❓ 모르겠음")
                    
                    st.session_state.w_curr_opts = opts
                    st.session_state.w_opts_word = target_w

                opts = st.session_state.w_curr_opts

                if not st.session_state.get('w_answered', False):
                    with st.form(key=f"wrong_quiz_form_{st.session_state.w_idx}"):
                        user_ans = st.radio("올바른 뜻 선택:", opts)
                        if st.form_submit_button("제출"):
                            st.session_state.w_user_ans = user_ans.strip()
                            st.session_state.w_answered = True
                            
                            if user_ans.strip() == correct_w:
                                sync_and_save_word(target_w, level=None, correct_inc=1, wrong_inc=0)
                            else:
                                sync_and_save_word(target_w, level=None, correct_inc=0, wrong_inc=1)
                            st.rerun()
                else:
                    u_ans = st.session_state.get('w_user_ans', '')
                    if u_ans == correct_w:
                        st.success(f"🎉 정답입니다! (**{correct_w}**)")
                    elif u_ans == "❓ 모르겠음":
                        st.warning(f"💡 정답: **'{correct_w}'**")
                    else:
                        st.error(f"❌ 틀렸습니다! **정답: '{correct_w}'**")
                    
                    if st.button("다음 오답 문제 ➡️", key="w_next_btn"):
                        st.session_state.w_idx += 1
                        st.session_state.w_answered = False
                        st.session_state.pop('w_curr_opts', None)
                        st.rerun()
            else:
                st.success("오답 집중 복습 완료!")
                if st.button("다시 처음부터 복습", key="w_restart_btn"):
                    st.session_state.w_idx = 0
                    st.session_state.w_answered = False
                    st.session_state.pop('w_curr_opts', None)
                    st.rerun()

# --- TAB 4: 반응속도 복습 ---
with tab4:
    st.caption("⚡ 4초 이내 빠른 정답 = 숙달도 상승⬆️ | 모르겠음/오답 = 숙달도 초기화⬇️")

    current_df['숙달도'] = pd.to_numeric(current_df['숙달도'], errors='coerce').fillna(1).astype(int)
    weights = (6 - current_df['숙달도']) ** 2
    
    if 'target_sample' not in st.session_state:
        st.session_state.target_sample = current_df.sample(1, weights=weights).iloc[0]
        st.session_state.rev_start_time = time.time()

    sample = st.session_state.target_sample
    target_word = str(sample['영어 단어']).strip()
    correct_ans = str(sample.get('대표 뜻', '-')).strip()
    phonetic_s = sample.get('발음기호', '-')
    example_s = sample.get('예문', '')
    example_trans_s = sample.get('예문 번역', '')
    curr_level = int(sample['숙달도'])

    st.markdown(f"### 단어: **{target_word}** `{phonetic_s}` (`{'⭐'*curr_level}`)")
    rev_audio = get_audio_bytes(target_word)
    if rev_audio: st.audio(rev_audio, format="audio/mp3")

    st.markdown(f"> 💡 **예문:** \"{highlight_target_word(example_s, target_word)}\"")
    if example_trans_s and example_trans_s != "예문 없음":
        st.markdown(f"> 💬 **해석:** \"{example_trans_s}\"")

    if 'rev_curr_opts' not in st.session_state or st.session_state.get('rev_opts_word') != target_word:
        all_meanings = list(set(current_df['대표 뜻'].astype(str).str.strip()) - {correct_ans})
        distractors = random.sample(all_meanings, min(3, len(all_meanings)))
        options = distractors + [correct_ans]
        random.shuffle(options)
        options.append("❓ 모르겠음")
        
        st.session_state.rev_curr_opts = options
        st.session_state.rev_opts_word = target_word

    options = st.session_state.rev_curr_opts

    if not st.session_state.get('rev_answered', False):
        with st.form(key=f"adaptive_form_{target_word}"):
            user_ans = st.radio("뜻 선택:", options)
            submit = st.form_submit_button("제출 및 정답 확인")

            if submit:
                elapsed_time = time.time() - st.session_state.rev_start_time
                st.session_state.rev_user_ans = user_ans.strip()
                st.session_state.rev_elapsed = elapsed_time
                st.session_state.rev_answered = True

                if user_ans.strip() == correct_ans:
                    new_lvl = min(5, curr_level + 1) if elapsed_time <= 4.0 else curr_level
                    sync_and_save_word(target_word, level=new_lvl, correct_inc=1, wrong_inc=0)
                else:
                    sync_and_save_word(target_word, level=1, correct_inc=0, wrong_inc=1)

                st.rerun()
    else:
        u_ans = st.session_state.get('rev_user_ans', '')
        el_time = st.session_state.get('rev_elapsed', 0.0)
        
        if u_ans == correct_ans:
            if el_time <= 4.0:
                st.success(f"⚡ 빠른 정답! ({el_time:.1f}초) 숙달도 상승⬆️ (**{correct_ans}**)")
            else:
                st.info(f"👍 정답입니다! ({el_time:.1f}초) (**{correct_ans}**)")
        elif u_ans == "❓ 모르겠음":
            st.warning(f"💡 정답: **{correct_ans}** -> 숙달도 초기화⬇️")
        else:
            st.error(f"❌ 틀렸습니다! 선택: '{u_ans}' ➔ **정답: '{correct_ans}'** -> 숙달도 초기화⬇️")

        if st.button("다음 카드 ➡️", key="rev_next_btn"):
            st.session_state.target_sample = current_df.sample(1, weights=(6 - current_df['숙달도'])**2).iloc[0]
            st.session_state.rev_start_time = time.time()
            st.session_state.rev_answered = False
            st.session_state.pop('rev_curr_opts', None)
            st.rerun()

# --- TAB 5: 예문 암기 (3000 단어장 엑셀 기반) ---
with tab5:
    sentence_df = current_df[
        (current_df['예문'] != '예문 없음') & 
        (current_df['예문'].notna())
    ].reset_index(drop=True)
    
    if sentence_df.empty:
        st.info("등록된 예문이 없습니다.")
    else:
        mode = st.radio("모드 선택:", ["✍️ 빈칸 퀴즈", "🎴 예문 카드"], horizontal=True, key="sent_mode_radio")
        
        if mode == "✍️ 빈칸 퀴즈":
            if 'sent_idx' not in st.session_state: st.session_state.sent_idx = 0
            total_sent = len(sentence_df)
            
            if st.button("🎲 예문 무작위 섞기", key="sent_reshuffle_btn"):
                st.session_state.sentence_quiz_df = sentence_df.sample(frac=1).reset_index(drop=True)
                st.session_state.sent_idx = 0
                st.session_state.sent_answered = False
                st.rerun()

            quiz_sent_df = st.session_state.get('sentence_quiz_df', sentence_df)
            
            if st.session_state.sent_idx < total_sent:
                s_row = quiz_sent_df.iloc[st.session_state.sent_idx]
                target_w = str(s_row['영어 단어']).strip()
                full_sent = s_row['예문']
                trans_sent = s_row['예문 번역']
                blanked_sent = make_blank_sentence(full_sent, target_w)
                
                st.progress((st.session_state.sent_idx) / total_sent)
                st.caption(f"예문 {st.session_state.sent_idx + 1} / {total_sent}")
                
                st.info(f"💡 **해석:** {trans_sent}")
                st.warning(f"📝 **문장:** {blanked_sent}")
                
                if not st.session_state.get('sent_answered', False):
                    with st.form(key=f"sent_quiz_form_{st.session_state.sent_idx}"):
                        user_input = st.text_input("✏️ 빈칸 영단어 입력:", "")
                        sub_col1, sub_col2 = st.columns(2)
                        with sub_col1: sub_btn = st.form_submit_button("정답 제출")
                        with sub_col2: dont_know_btn = st.form_submit_button("❓ 모르겠음")
                        
                        if sub_btn or dont_know_btn:
                            st.session_state.sent_user_input = user_input.strip().lower()
                            st.session_state.sent_is_dont_know = dont_know_btn
                            st.session_state.sent_answered = True
                            st.rerun()
                else:
                    is_dk = st.session_state.get('sent_is_dont_know', False)
                    u_in = st.session_state.get('sent_user_input', '')
                    c_target = target_w.lower()
                    
                    if is_dk:
                        st.warning(f"💡 정답 단어는 **'{target_w}'** 입니다.")
                    elif u_in == c_target:
                        st.success(f"🎉 정답입니다! 단어: **{target_w}**")
                    else:
                        st.error(f"❌ 틀렸습니다. 입력한 답: '{u_in}' ➔ **정답: '{target_w}'**")
                        
                    st.markdown(f"> 👉 **전체 문장:** {highlight_target_word(full_sent, target_w)}")
                    
                    sent_audio = get_audio_bytes(full_sent)
                    if sent_audio:
                        st.caption("🔊 원어민 전체 문장 발음 듣기:")
                        st.audio(sent_audio, format="audio/mp3")
                    
                    if st.button("다음 예문 ➡️", key="sent_next_btn"):
                        st.session_state.sent_idx += 1
                        st.session_state.sent_answered = False
                        st.rerun()
            else:
                st.balloons()
                st.success("🏆 모든 예문 학습 완료!")
                if st.button("다시 하기", key="sent_restart_btn"):
                    st.session_state.sent_idx = 0
                    st.session_state.sent_answered = False
                    st.rerun()

        else:
            for i, row in sentence_df.iterrows():
                meaning = row.get('대표 뜻', '')
                
                with st.expander(f"{i+1}. 💡 {row['예문 번역']}"):
                    st.markdown(f"**영어:** {highlight_target_word(row['예문'], row['영어 단어'])}")
                    st.caption(f"단어: **{row['영어 단어']}** `{row['발음기호']}` | 뜻: **{meaning}**")
                    
                    card_audio = get_audio_bytes(row['예문'])
                    if card_audio:
                        st.audio(card_audio, format="audio/mp3")
