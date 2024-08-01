import streamlit as st
import nltk
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from datetime import datetime

# NLTK 데이터 다운로드
nltk.download('punkt')
nltk.download('wordnet')

# 인사말 및 작별 인사 정의
GREETING_INPUTS = ["hello", "hi", "greetings", "sup", "what's up", "hey", "hey there"]
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

GOODBYE_INPUTS = ["bye", "see you!", "unit", "exit"]
GOODBYE_RESPONSES = ["Goodbye!", "See you later!", "Take care!", "Farewell!"]

# 인사말을 수신하고 반환하는 함수
def greeting(sentence):
    for word in sentence.split(): # 문장의 각 단어를 살펴봅니다.
        if word.lower() in GREETING_INPUTS: # 단어가 GREETING_INPUT와 일치하는지 확인합니다.
            return random.choice(GREETING_RESPONSES) # Greeting_Response로 답장합니다.
    return None

# 끝인삿말을 수신하고 반환하는 함수
def goodbye(sentence):
    for word in sentence.split():
        if word.lower() in GOODBYE_INPUTS:
            return random.choice(GOODBYE_RESPONSES)
    return None

# 텍스트 파일에서 데이터 읽기
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().lower()

def LemTokens(tokens):
    lemmer = nltk.stem.WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def response(user_response, all_questions, ques_ans_pairs, normalized_tokens):
    robo_response = ''
    all_questions.append(user_response)  # 사용자 응답을 추가

    # TF-IDF 벡터화
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize)
    tfidf = TfidfVec.fit_transform(all_questions)  # tfidf 값 가져오기

    # 마지막 문장과 나머지 문장 간의 코사인 유사도 계산
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])

    # 사용자 입력 제거
    all_questions.pop()  # 사용자 입력을 all_questions에서 제거

    # 가장 유사한 문장의 인덱스 찾기
    idx = vals.argsort()[0][-1]  # 가장 큰 값을 가진 인덱스 찾기

    # 유사도 값 추출
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]  # 가장 큰 유사도 값

    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you"
    else:
        robo_response = f"{ques_ans_pairs[all_questions[idx]]} (Similarity Score: {req_tfidf:.2f})"

    return robo_response

# Streamlit 인터페이스
st.title("Hotel Chatbot")

# 파일 경로 설정
ques_filepath = '[Dataset] Module27(ques).txt'
ans_filepath = '[Dataset] Module27 (ans).txt'

try:
    raw_data_ques = read_file(ques_filepath)
    raw_data_ans = read_file(ans_filepath)
    
    sent_tokens_ques = nltk.sent_tokenize(raw_data_ques)
    sent_tokens_ans = nltk.sent_tokenize(raw_data_ans)
    
    if len(sent_tokens_ques) != len(sent_tokens_ans):
        st.error("The number of questions and answers do not match.")
    else:
        ques_ans_pairs = dict(zip(sent_tokens_ques, sent_tokens_ans))
        all_questions = list(ques_ans_pairs.keys())
        
        # 챗봇 상호작용
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            st.session_state.chat_history.append("Jane: My name is Jane. I will answer your queries about this hotel. If you want to exit, type Bye!")

        user_input = st.text_input("You: ", key="input")
        if user_input:
            st.session_state.chat_history.append(f"You: {user_input}")
            
            # 인사말 처리
            greet_response = greeting(user_input)
            if greet_response:
                st.session_state.chat_history.append(f"Jane: {greet_response}")
            else:
                # 끝인사 처리
                bye_response = goodbye(user_input)
                if bye_response:
                    st.session_state.chat_history.append(f"Jane: {bye_response}")
                else:
                    if user_input.lower() in ['thanks', 'thank you']:
                        st.session_state.chat_history.append("Jane: You are welcome..")
                    else:
                        normalized_tokens = LemNormalize(' '.join(all_questions))
                        answer = response(user_input, all_questions, ques_ans_pairs, normalized_tokens)
                        st.session_state.chat_history.append(f"Jane: {answer}")

        with st.expander("Chat History", expanded=True):
            for chat in st.session_state.chat_history:
                if "You:" in chat:
                    st.write(f'<div style="display:flex;align-items:center;"><div style="background-color:#007AFF;color:white;border-radius:12px;padding:8px 12px;margin-right:8px;">{chat}</div></div>', unsafe_allow_html=True)
                else:
                    st.write(f'<div style="display:flex;align-items:center;justify-content:flex-end;"><div style="background-color:lightgray;border-radius:12px;padding:8px 12px;margin-left:8px;">{chat}</div></div>', unsafe_allow_html=True)
except FileNotFoundError as e:
    st.error(f"Error: {e}")



