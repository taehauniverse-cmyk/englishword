import streamlit as st
import pandas as pd
import re
from collections import Counter
import random
import nltk
from pypdf import PdfReader
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
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

init_nltk()

from nltk.corpus import stopwords
from nltk.tag import pos_tag
from deep_translator import GoogleTranslator

# --- 2. 원어민 발음 음성 생성 (gTTS 캐싱) ---
@st.cache_data
def get_audio_bytes(text_to_speak):
    if not text_to_speak or text_to_speak in ["대본 예문 없음", "예문 없음", "-", ""]:
        return None
    try:
        tts = gTTS(text=text_to_speak, lang='en', tld='com')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        return None

# --- 3. 오늘의 학습 시간 측정 로직 (5분 이내 조작 누적) ---
now_time = time.time()
today_date_str = time.strftime("%Y-%m-%d")

if 'study_date' not in st.session_state or st.session_state.study_date != today_date_str:
    st.session_state.study_date = today_date_str
    st.session_state.today_study_sec = 0
    st.session_state.last_ping_time = now_time
else:
    time_diff = now_time - st.session_state.get('last_ping_time', now_time)
    if time_diff < 300:  # 5분 이내 조작 시에만 학습 시간 카운트
        st.session_state.today_study_sec += time_diff
    st.session_state.last_ping_time = now_time

total_sec = int(st.session_state.today_study_sec)
hrs = total_sec // 3600
mins = (total_sec % 3600) // 60
secs = total_sec % 60
study_time_display = f"{hrs}시간 {mins}분 {secs}초" if hrs > 0 else f"{mins}분 {secs}초"

# --- 4. 데이터 폴더 및 메타데이터 관리 ---
DATA_DIR = "script_data"
META_FILE = os.path.join(DATA_DIR, "scripts_meta.json")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def load_meta():
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_meta(meta):
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def get_phonetic_symbol(word):
    try:
        res = ipa.convert(word)
        if res.endswith('*'):
            return "-"
        return f"/{res}/"
    except Exception:
        return "-"

def load_script_vocab(script_id):
    filepath = os.path.join(DATA_DIR, f"{script_id}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        updated = False

        if "한글 뜻" in df.columns and "대본 문맥 뜻" not in df.columns:
            df["대본 문맥 뜻"] = df["한글 뜻"]
            df["사전 대표 뜻"] = df["한글 뜻"]
            updated = True
        
        for col in ["대본 문맥 뜻", "사전 대표 뜻", "발음기호", "대본 예문", "대본 예문 번역"]:
            if col not in df.columns:
                df[col] = "-" if "뜻" in col or "발음" in col else "대본 예문 없음"
                updated = True

        for col, default_val in [("숙달도", 1), ("틀린 횟수", 0), ("맞춘 횟수", 0)]:
            if col not in df.columns:
                df[col] = default_val
                updated = True

        missing_ipa_mask = (df["발음기호"] == "-") | (df["발음기호"].isna())
        if missing_ipa_mask.any():
            df.loc[missing_ipa_mask, "발음기호"] = df.loc[missing_ipa_mask, "영어 단어"].apply(get_phonetic_symbol)
            updated = True

        if updated:
            save_script_vocab(script_id, df)

        return df
    return pd.DataFrame(columns=[
        "영어 단어", "발음기호", "대본 문맥 뜻", "사전 대표 뜻", "품사", "누적 빈도수", 
        "대본 예문", "대본 예문 번역", "숙달도", "틀린 횟수", "맞춘 횟수"
    ])

def save_script_vocab(script_id, df):
    filepath = os.path.join(DATA_DIR, f"{script_id}.csv")
    df.to_csv(filepath, index=False, encoding='utf-8-sig')

def map_pos_to_korean(tag):
    if tag.startswith('NN'): return '명사'
    elif tag.startswith('VB'): return '동사'
    elif tag.startswith('JJ'): return '형용사'
    elif tag.startswith('RB'): return '부사'
    else: return '기타'

def analyze_script_full(text):
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    word_counts = Counter(filtered_words)
    
    raw_sentences = re.split(r'\n+|\.|\!|\?', text)
    sentences = [s.strip().replace('\n', ' ') for s in raw_sentences if len(s.strip()) > 10]
    return word_counts, sentences

def find_script_sentence(word, sentences):
    pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
    for sent in sentences:
        if pattern.search(sent):
            clean_sent = ' '.join(sent.split())
            if len(clean_sent) > 130:
                clean_sent = clean_sent[:130] + "..."
            return clean_sent
    return "대본 예문 없음"

def highlight_target_word(sentence, target_word):
    if not sentence or sentence == "대본 예문 없음":
        return "등록된 대본 예문이 없습니다."
    pattern = re.compile(r'\b(' + re.escape(target_word) + r')\b', re.IGNORECASE)
    return pattern.sub(r'**\1**', sentence)

def make_blank_sentence(sentence, target_word):
    pattern = re.compile(r'\b' + re.escape(target_word) + r'\b', re.IGNORECASE)
    return pattern.sub("✏️ [ ______ ]", sentence)

def process_new_script(text, top_n, script_title):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("1/4. 대본 분석 중... (5%)")
    progress_bar.progress(0.05)
    
    word_counts, sentences = analyze_script_full(text)
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not sorted_words:
        progress_bar.empty()
        status_text.empty()
        return None, 0

    extracted_words = [w for w, c in sorted_words]
    pos_tags = dict(pos_tag(extracted_words))
    translator = GoogleTranslator(source='en', target='ko')
    
    status_text.text("2/4. 사전 대표 뜻 추출 중... (15%)")
    progress_bar.progress(0.15)
    
    batch_size = 50
    dict_meanings = []
    total_batches = (len(extracted_words) + batch_size - 1) // batch_size
    
    for i in range(0, len(extracted_words), batch_size):
        batch = extracted_words[i:i + batch_size]
        try:
            dict_meanings.extend(translator.translate_batch(batch))
        except Exception:
            for w in batch:
                try: dict_meanings.append(translator.translate(w))
                except Exception: dict_meanings.append("뜻 불러오기 실패")
        
        pct = 0.15 + ((i // batch_size) + 1 / total_batches) * 0.25
        progress_bar.progress(min(0.40, pct))

    status_text.text("3/4. 문장 및 발음기호 추출 중... (45%)")
    progress_bar.progress(0.45)
    
    examples = [find_script_sentence(w, sentences) for w in extracted_words]
    example_translations = []
    example_batch_size = 10
    total_ex_batches = (len(examples) + example_batch_size - 1) // example_batch_size
    
    for i in range(0, len(examples), example_batch_size):
        batch_sents = examples[i:i + example_batch_size]
        try:
            example_translations.extend(translator.translate_batch(batch_sents))
        except Exception:
            for sent in batch_sents:
                if sent == "대본 예문 없음":
                    example_translations.append("예문 번역 없음")
                else:
                    try: example_translations.append(translator.translate(sent))
                    except Exception: example_translations.append("예문 번역 실패")
        
        pct = 0.45 + ((i // example_batch_size) + 1 / total_ex_batches) * 0.50
        progress_bar.progress(min(0.95, pct))

    rows = []
    for (word, count), dict_meaning, example, ex_trans in zip(sorted_words, dict_meanings, examples, example_translations):
        pos = map_pos_to_korean(pos_tags.get(word, ''))
        phonetic = get_phonetic_symbol(word)
        
        rows.append({
            "영어 단어": word,
            "발음기호": phonetic,
            "대본 문맥 뜻": dict_meaning,
            "사전 대표 뜻": dict_meaning,
            "품사": pos,
            "누적 빈도수": count,
            "대본 예문": example,
            "대본 예문 번역": ex_trans,
            "숙달도": 1,
            "틀린 횟수": 0,
            "맞춘 횟수": 0
        })
        
    df = pd.DataFrame(rows)
    script_id = f"script_{int(time.time())}"
    save_script_vocab(script_id, df)
    
    meta = load_meta()
    meta[script_id] = {
        "title": script_title,
        "total_words": len(word_counts),
        "extracted_words": len(df),
        "created_at": time.strftime("%Y-%m-%d %H:%M")
    }
    save_meta(meta)
    
    progress_bar.progress(1.0)
    status_text.text("🎉 생성 및 서버 자동 저장 완료!")
    time.sleep(0.8)
    progress_bar.empty()
    status_text.empty()
    return script_id, len(word_counts)

# --- 5. UI 기본 레이아웃 & 모바일 스타일 적용 ---
st.set_page_config(page_title="대본 영단어장", layout="wide")

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

st.markdown("### 🎬 대본 영단어장 & 대사 암기기")

meta = load_meta()

# --- 6. 사이드바 보관함 및 학습 보드 ---
with st.sidebar:
    st.header("📂 대본 보관함")
    
    options = {"➕ 새 대본 등록하기": "NEW"}
    for sid, info in meta.items():
        options[f"🎬 {info['title']} ({info['extracted_words']}단어)"] = sid
    options["📥 보조: 백업 파일(.csv) 복원"] = "RESTORE"

    selected_option = st.selectbox("공부할 대본 선택", list(options.keys()))
    selected_script_id = options[selected_option]

    if 'current_script' not in st.session_state or st.session_state.current_script != selected_script_id:
        st.session_state.current_script = selected_script_id
        for k in list(st.session_state.keys()):
            if k.startswith(('q_', 'w_', 'rev_', 'sent_')):
                st.session_state.pop(k, None)

    current_df = pd.DataFrame()

    if selected_script_id == "NEW":
        st.subheader("🆕 새 대본 추가")
        script_title_input = st.text_input("대본 이름", placeholder="예: 어벤져스, 미드 오피스 S1")
        uploaded_file = st.file_uploader("텍스트(.txt) 또는 PDF(.pdf)", type=["txt", "pdf"])
        
        script_text = ""
        total_unique = 0
        if uploaded_file:
            if uploaded_file.name.endswith('.pdf'):
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    txt = page.extract_text()
                    if txt: script_text += txt + "\n"
            else:
                script_text = uploaded_file.read().decode("utf-8", errors="ignore")
                
            word_counts, _ = analyze_script_full(script_text)
            total_unique = len(word_counts)
            st.info(f"💡 추출 가능 총 단어: **{total_unique:,}개**")

        top_n = st.number_input(
            "추출 단어 수", 
            min_value=10, 
            max_value=max(total_unique, 5000) if total_unique > 0 else 5000, 
            value=min(200, total_unique) if total_unique > 0 else 200,
            step=50
        )
        
        if st.button("🚀 단어장 생성 및 서버 저장"):
            if not script_title_input.strip():
                st.error("대본 이름을 입력해 주세요!")
            elif not script_text:
                st.error("대본 파일을 올려주세요!")
            else:
                new_id, _ = process_new_script(script_text, top_n, script_title_input.strip())
                if new_id:
                    st.success("서버 보관함에 저장되었습니다!")
                    st.rerun()

    elif selected_script_id == "RESTORE":
        st.subheader("📥 백업 파일로 복원")
        backup_file = st.file_uploader("백업 CSV 선택", type=["csv"])
        restore_title = st.text_input("복원할 대본 이름 입력", placeholder="예: 어벤져스 복구본")
        
        if backup_file and restore_title.strip():
            if st.button("🚀 서버로 데이터 복구하기"):
                try:
                    res_df = pd.read_csv(backup_file)
                    new_sid = f"script_{int(time.time())}"
                    save_script_vocab(new_sid, res_df)
                    
                    meta[new_sid] = {
                        "title": restore_title.strip(),
                        "total_words": len(res_df),
                        "extracted_words": len(res_df),
                        "created_at": time.strftime("%Y-%m-%d %H:%M")
                    }
                    save_meta(meta)
                    st.success("성공적으로 복구되었습니다!")
                    st.rerun()
                except Exception:
                    st.error("올바른 백업 파일이 아닙니다.")

    else:
        current_df = load_script_vocab(selected_script_id)
        script_info = meta[selected_script_id]
        
        st.divider()
        st.header("📈 학습 보드")
        st.caption(f"📌 대본: **{script_info['title']}**")
        
        total_saved = len(current_df)
        total_correct = pd.to_numeric(current_df['맞춘 횟수'], errors='coerce').fillna(0).sum() if total_saved > 0 else 0
        total_wrong = pd.to_numeric(current_df['틀린 횟수'], errors='coerce').fillna(0).sum() if total_saved > 0 else 0
        total_attempts = total_correct + total_wrong
        acc_rate = (total_correct / total_attempts * 100) if total_attempts > 0 else 0.0

        current_df['숙달도'] = pd.to_numeric(current_df['숙달도'], errors='coerce').fillna(1).astype(int)
        mastery_sum = (current_df['숙달도'] - 1).sum() if total_saved > 0 else 0
        max_mastery = total_saved * 4 if total_saved > 0 else 1
        learning_completion_rate = (mastery_sum / max_mastery * 100) if total_saved > 0 else 0.0

        st.metric("🎓 종합 학습 완수율", f"{learning_completion_rate:.1f} %")
        st.progress(learning_completion_rate / 100)
        
        st.metric("⏱️ 오늘의 학습 시간", study_time_display)
        st.metric("📚 추출 단어", f"{total_saved:,} 개")
        st.metric("🎯 퀴즈 풀이", f"{int(total_attempts):,} 회")
        st.metric("📊 퀴즈 정답률", f"{acc_rate:.1f} %")
        
        st.divider()
        csv_backup = current_df.to_csv(index=False).encode('utf-8-sig')
        file_clean = re.sub(r'[^\w\s-]', '', script_info['title']).strip()
        st.download_button("💾 백업 파일 다운로드", csv_backup, f"{file_clean}_backup.csv", "text/csv")
        
        st.divider()
        if st.button("🗑️ 이 대본 서버에서 삭제"):
            csv_path = os.path.join(DATA_DIR, f"{selected_script_id}.csv")
            if os.path.exists(csv_path): os.remove(csv_path)
            if selected_script_id in meta:
                del meta[selected_script_id]
                save_meta(meta)
            st.warning("삭제되었습니다.")
            st.rerun()

# --- 7. 메인 화면 탭 인터페이스 ---
if selected_script_id in ["NEW", "RESTORE"]:
    st.info("👈 사이드바 메뉴에서 대본을 선택하거나 새 대본을 등록해 주세요.")
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 단어장", 
        "🧠 4지선다 퀴즈", 
        "❌ 오답노트", 
        "⚡ 반응속도 복습",
        "💬 문장 암기"
    ])

    # --- TAB 1: 단어장 편집 ---
    with tab1:
        st.caption("✏️ **팁:** 표에서 뜻을 직접 클릭하여 수정할 수 있습니다.")
        display_df = current_df.copy()
        display_df['숙달도'] = pd.to_numeric(display_df['숙달도'], errors='coerce').fillna(1).astype(int)
        display_df['숙달 상태'] = display_df['숙달도'].apply(lambda x: "⭐"*x)
        
        search = st.text_input("단어 검색", "")
        if search:
            display_df = display_df[display_df['영어 단어'].astype(str).str.contains(search.lower())]

        target_cols = [c for c in ["영어 단어", "발음기호", "대본 문맥 뜻", "사전 대표 뜻", "품사", "누적 빈도수", "대본 예문", "대본 예문 번역", "숙달 상태", "맞춘 횟수", "틀린 횟수"] if c in display_df.columns]

        edited_df = st.data_editor(display_df[target_cols], use_container_width=True, num_rows="dynamic")
        
        if st.button("💾 수정사항 서버에 저장하기", key="save_vocab_btn"):
            for col in ["대본 문맥 뜻", "사전 대표 뜻", "대본 예문", "대본 예문 번역"]:
                if col in edited_df.columns:
                    current_df[col] = edited_df[col]
            save_script_vocab(selected_script_id, current_df)
            st.success("성공적으로 저장되었습니다!")
            st.rerun()

    # --- TAB 2: 4지선다 퀴즈 ---
    with tab2:
        if len(current_df) < 4:
            st.warning("단어가 최소 4개 이상 필요합니다.")
        else:
            if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
            if 'q_score' not in st.session_state: st.session_state.q_score = 0
            if 'q_data' not in st.session_state or st.button("🎲 퀴즈 새로 섞기", key="q_reshuffle_btn"):
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
                correct_ans = str(row.get('대본 문맥 뜻', row.get('사전 대표 뜻', '-'))).strip()
                dict_ans = str(row.get('사전 대표 뜻', correct_ans)).strip()
                phonetic = row.get('발음기호', '-')
                example_sent = row.get('대본 예문', '')
                example_trans = row.get('대본 예문 번역', '')
                
                st.progress(st.session_state.q_idx / total_q)
                st.info(f"### **{target_word}** `[{row.get('품사', '')}]` `{phonetic}`")
                
                word_audio = get_audio_bytes(target_word)
                if word_audio: st.audio(word_audio, format="audio/mp3")
                
                st.markdown(f"> 🎬 **예문:** \"{highlight_target_word(example_sent, target_word)}\"")
                if example_trans and example_trans != "예문 번역 없음":
                    st.markdown(f"> 💡 **번역:** \"{example_trans}\"")

                if 'q_curr_opts' not in st.session_state or st.session_state.get('q_opts_word') != target_word:
                    mean_col = '대본 문맥 뜻' if '대본 문맥 뜻' in current_df.columns else '사전 대표 뜻'
                    all_meanings = list(set(current_df[mean_col].astype(str).str.strip()) - {correct_ans})
                    distractors = random.sample(all_meanings, min(3, len(all_meanings)))
                    options = distractors + [correct_ans]
                    random.shuffle(options)
                    options.append("❓ 모르겠음")
                    
                    st.session_state.q_curr_opts = options
                    st.session_state.q_opts_word = target_word

                options = st.session_state.q_curr_opts

                if not st.session_state.get('q_answered', False):
                    with st.form(key=f"gen_quiz_form_{st.session_state.q_idx}"):
                        user_ans = st.radio("올바른 대본 문맥 뜻 선택:", options)
                        submit = st.form_submit_button("정답 제출")
                        if submit:
                            real_idx = current_df[current_df['영어 단어'] == target_word].index[0]
                            st.session_state.q_user_ans = user_ans.strip()
                            st.session_state.q_answered = True
                            
                            if user_ans.strip() == correct_ans:
                                st.session_state.q_score += 1
                                current_df.at[real_idx, '맞춘 횟수'] = int(current_df.at[real_idx, '맞춘 횟수']) + 1
                            else:
                                current_df.at[real_idx, '틀린 횟수'] = int(current_df.at[real_idx, '틀린 횟수']) + 1
                                if int(current_df.at[real_idx, '숙달도']) > 1:
                                    current_df.at[real_idx, '숙달도'] = int(current_df.at[real_idx, '숙달도']) - 1
                            
                            save_script_vocab(selected_script_id, current_df)
                            st.rerun()
                else:
                    u_ans = st.session_state.get('q_user_ans', '')
                    if u_ans == correct_ans:
                        st.success(f"🎉 정답입니다! (**{correct_ans}**)")
                    elif u_ans == "❓ 모르겠음":
                        st.warning(f"💡 정답은 **'{correct_ans}'** 입니다!")
                    else:
                        st.error(f"❌ 틀렸습니다! 내가 선택한 답: '{u_ans}' ➔ **정답: '{correct_ans}'**")
                    
                    st.info(f"📖 **사전 대표 뜻 참고:** {dict_ans}")
                    
                    if st.button("다음 문제 ➡️", key="q_next_btn"):
                        st.session_state.q_idx += 1
                        st.session_state.q_answered = False
                        st.session_state.pop('q_curr_opts', None)
                        st.rerun()
            else:
                st.balloons()
                st.success(f"🏆 퀴즈 완료! 점수: {st.session_state.q_score} / {total_q}")

    # --- TAB 3: 오답노트 ---
    with tab3:
        current_df['틀린 횟수'] = pd.to_numeric(current_df['틀린 횟수'], errors='coerce').fillna(0).astype(int)
        wrong_df = current_df[current_df['틀린 횟수'] > 0].sort_values(by='틀린 횟수', ascending=False)
        
        if wrong_df.empty:
            st.success("👏 틀린 단어가 없습니다!")
        else:
            st.caption(f"총 **{len(wrong_df)}개**의 오답 단어가 있습니다.")
            target_cols_w = [c for c in ["영어 단어", "발음기호", "대본 문맥 뜻", "사전 대표 뜻", "품사", "틀린 횟수"] if c in wrong_df.columns]
            st.dataframe(wrong_df[target_cols_w], use_container_width=True)
            
            st.divider()
            if len(wrong_df) >= 2:
                if 'w_idx' not in st.session_state: st.session_state.w_idx = 0
                
                if st.session_state.w_idx < len(wrong_df):
                    w_row = wrong_df.iloc[st.session_state.w_idx]
                    target_w = str(w_row['영어 단어']).strip()
                    correct_w = str(w_row.get('대본 문맥 뜻', w_row.get('사전 대표 뜻', '-'))).strip()
                    dict_w = str(w_row.get('사전 대표 뜻', correct_w)).strip()
                    phonetic_w = w_row.get('발음기호', '-')
                    example_w = w_row.get('대본 예문', '')
                    example_trans_w = w_row.get('대본 예문 번역', '')
                    
                    st.info(f"### **{target_w}** `{phonetic_w}` (틀린 횟수: {w_row['틀린 횟수']}회)")
                    w_audio = get_audio_bytes(target_w)
                    if w_audio: st.audio(w_audio, format="audio/mp3")

                    st.markdown(f"> 🎬 **예문:** \"{highlight_target_word(example_w, target_w)}\"")
                    if example_trans_w and example_trans_w != "예문 번역 없음":
                        st.markdown(f"> 💡 **번역:** \"{example_trans_w}\"")
                    
                    if 'w_curr_opts' not in st.session_state or st.session_state.get('w_opts_word') != target_w:
                        mean_col = '대본 문맥 뜻' if '대본 문맥 뜻' in current_df.columns else '사전 대표 뜻'
                        all_meanings = list(set(current_df[mean_col].astype(str).str.strip()) - {correct_w})
                        distractors = random.sample(all_meanings, min(3, len(all_meanings)))
                        opts = distractors + [correct_w]
                        random.shuffle(opts)
                        opts.append("❓ 모르겠음")
                        
                        st.session_state.w_curr_opts = opts
                        st.session_state.w_opts_word = target_w

                    opts = st.session_state.w_curr_opts

                    if not st.session_state.get('w_answered', False):
                        with st.form(key=f"wrong_quiz_form_{st.session_state.w_idx}"):
                            user_ans = st.radio("뜻 선택:", opts)
                            if st.form_submit_button("제출"):
                                real_idx = current_df[current_df['영어 단어'] == target_w].index[0]
                                st.session_state.w_user_ans = user_ans.strip()
                                st.session_state.w_answered = True
                                
                                if user_ans.strip() == correct_w:
                                    current_df.at[real_idx, '맞춘 횟수'] = int(current_df.at[real_idx, '맞춘 횟수']) + 1
                                else:
                                    current_df.at[real_idx, '틀린 횟수'] = int(current_df.at[real_idx, '틀린 횟수']) + 1
                                save_script_vocab(selected_script_id, current_df)
                                st.rerun()
                    else:
                        u_ans = st.session_state.get('w_user_ans', '')
                        if u_ans == correct_w:
                            st.success(f"🎉 정답입니다! (**{correct_w}**)")
                        elif u_ans == "❓ 모르겠음":
                            st.warning(f"💡 정답: **'{correct_w}'**")
                        else:
                            st.error(f"❌ 틀렸습니다! **정답: '{correct_w}'**")
                        
                        st.info(f"📖 **사전 대표 뜻 참고:** {dict_w}")
                        
                        if st.button("다음 문제 ➡️", key="w_next_btn"):
                            st.session_state.w_idx += 1
                            st.session_state.w_answered = False
                            st.session_state.pop('w_curr_opts', None)
                            st.rerun()
                else:
                    st.success("오답 집중 퀴즈 완료!")
                    if st.button("다시 시작", key="w_restart_btn"):
                        st.session_state.w_idx = 0
                        st.session_state.w_answered = False
                        st.session_state.pop('w_curr_opts', None)
                        st.rerun()

    # --- TAB 4: 반응속도 복습 ---
    with tab4:
        st.caption("⚡ 4초 이내 빠른 정답 = 숙달도 상승⬆️ | 모르겠음 및 오답 = 숙달도 초기화⬇️")

        if len(current_df) < 4:
            st.warning("단어가 최소 4개 이상 필요합니다.")
        else:
            current_df['숙달도'] = pd.to_numeric(current_df['숙달도'], errors='coerce').fillna(1).astype(int)
            weights = (6 - current_df['숙달도']) ** 2
            
            if 'target_sample' not in st.session_state:
                st.session_state.target_sample = current_df.sample(1, weights=weights).iloc[0]
                st.session_state.rev_start_time = time.time()

            sample = st.session_state.target_sample
            target_word = str(sample['영어 단어']).strip()
            correct_ans = str(sample.get('대본 문맥 뜻', sample.get('사전 대표 뜻', '-'))).strip()
            dict_ans = str(sample.get('사전 대표 뜻', correct_ans)).strip()
            phonetic_s = sample.get('발음기호', '-')
            example_s = sample.get('대본 예문', '')
            example_trans_s = sample.get('대본 예문 번역', '')
            curr_level = int(sample['숙달도'])

            st.markdown(f"### 단어: **{target_word}** `{phonetic_s}` (`{'⭐'*curr_level}`)")
            rev_audio = get_audio_bytes(target_word)
            if rev_audio: st.audio(rev_audio, format="audio/mp3")

            st.markdown(f"> 🎬 **예문:** \"{highlight_target_word(example_s, target_word)}\"")
            if example_trans_s and example_trans_s != "예문 번역 없음":
                st.markdown(f"> 💡 **번역:** \"{example_trans_s}\"")

            if 'rev_curr_opts' not in st.session_state or st.session_state.get('rev_opts_word') != target_word:
                mean_col = '대본 문맥 뜻' if '대본 문맥 뜻' in current_df.columns else '사전 대표 뜻'
                all_meanings = list(set(current_df[mean_col].astype(str).str.strip()) - {correct_ans})
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
                        real_idx = current_df[current_df['영어 단어'] == target_word].index[0]
                        st.session_state.rev_user_ans = user_ans.strip()
                        st.session_state.rev_elapsed = elapsed_time
                        st.session_state.rev_answered = True

                        if user_ans.strip() == correct_ans:
                            if elapsed_time <= 4.0:
                                if int(current_df.at[real_idx, '숙달도']) < 5:
                                    current_df.at[real_idx, '숙달도'] = int(current_df.at[real_idx, '숙달도']) + 1
                            current_df.at[real_idx, '맞춘 횟수'] = int(current_df.at[real_idx, '맞춘 횟수']) + 1
                        else:
                            current_df.at[real_idx, '틀린 횟수'] = int(current_df.at[real_idx, '틀린 횟수']) + 1
                            current_df.at[real_idx, '숙달도'] = 1

                        save_script_vocab(selected_script_id, current_df)
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

                st.info(f"📖 **사전 대표 뜻 참고:** {dict_ans}")

                if st.button("다음 문제 ➡️", key="rev_next_btn"):
                    st.session_state.target_sample = current_df.sample(1, weights=(6 - current_df['숙달도'])**2).iloc[0]
                    st.session_state.rev_start_time = time.time()
                    st.session_state.rev_answered = False
                    st.session_state.pop('rev_curr_opts', None)
                    st.rerun()

    # --- TAB 5: 대본 문장 암기 ---
    with tab5:
        sentence_df = current_df[
            (current_df['대본 예문'] != '대본 예문 없음') & 
            (current_df['대본 예문'].notna())
        ].reset_index(drop=True)
        
        if sentence_df.empty:
            st.info("이 대본에서 추출된 문장이 없습니다.")
        else:
            mode = st.radio("모드 선택:", ["✍️ 빈칸 퀴즈", "🎴 문장 카드"], horizontal=True, key="sent_mode_radio")
            
            if mode == "✍️ 빈칸 퀴즈":
                if 'sent_idx' not in st.session_state: st.session_state.sent_idx = 0
                total_sent = len(sentence_df)
                
                if st.button("🎲 순서 새로 섞기", key="sent_reshuffle_btn"):
                    st.session_state.sentence_quiz_df = sentence_df.sample(frac=1).reset_index(drop=True)
                    st.session_state.sent_idx = 0
                    st.session_state.sent_answered = False
                    st.rerun()

                quiz_sent_df = st.session_state.get('sentence_quiz_df', sentence_df)
                
                if st.session_state.sent_idx < total_sent:
                    s_row = quiz_sent_df.iloc[st.session_state.sent_idx]
                    target_w = str(s_row['영어 단어']).strip()
                    full_sent = s_row['대본 예문']
                    trans_sent = s_row['대본 예문 번역']
                    blanked_sent = make_blank_sentence(full_sent, target_w)
                    
                    st.progress((st.session_state.sent_idx) / total_sent)
                    st.caption(f"문장 {st.session_state.sent_idx + 1} / {total_sent}")
                    
                    st.info(f"💡 **뜻:** {trans_sent}")
                    st.warning(f"🎬 **문장:** {blanked_sent}")
                    
                    if not st.session_state.get('sent_answered', False):
                        with st.form(key=f"sent_quiz_form_{st.session_state.sent_idx}"):
                            user_input = st.text_input("✏️ 빈칸 단어 입력:", "")
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
                            
                        st.markdown(f"> 👉 **완성된 전체 문장:** {highlight_target_word(full_sent, target_w)}")
                        
                        sent_audio = get_audio_bytes(full_sent)
                        if sent_audio:
                            st.caption("🔊 원어민 전체 대사 발음 듣기:")
                            st.audio(sent_audio, format="audio/mp3")
                        
                        if st.button("다음 문장 ➡️", key="sent_next_btn"):
                            st.session_state.sent_idx += 1
                            st.session_state.sent_answered = False
                            st.rerun()
                else:
                    st.balloons()
                    st.success("🏆 모든 문장 암기 완료!")
                    if st.button("다시 하기", key="sent_restart_btn"):
                        st.session_state.sent_idx = 0
                        st.session_state.sent_answered = False
                        st.rerun()

            else:
                for i, row in sentence_df.iterrows():
                    c_meaning = row.get('대본 문맥 뜻', row.get('사전 대표 뜻', ''))
                    d_meaning = row.get('사전 대표 뜻', c_meaning)
                    
                    with st.expander(f"{i+1}. 💡 {row['대본 예문 번역']}"):
                        st.markdown(f"**영어:** {highlight_target_word(row['대본 예문'], row['영어 단어'])}")
                        st.caption(f"단어: **{row['영어 단어']}** `{row['발음기호']}` | 문맥 뜻: **{c_meaning}** (사전 뜻: {d_meaning})")
                        
                        card_audio = get_audio_bytes(row['대본 예문'])
                        if card_audio:
                            st.audio(card_audio, format="audio/mp3")
