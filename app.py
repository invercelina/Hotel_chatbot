import streamlit as st
import nltk
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import string

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

        user_input = st.text_input("You: ", key="input")
        if user_input:
            st.session_state.chat_history.append(f"You: {user_input}")
            if user_input.lower() == 'bye':
                st.session_state.chat_history.append("Jane: Goodbye! Take care!")
            elif user_input.lower() in ['thanks', 'thank you']:
                st.session_state.chat_history.append("Jane: You are welcome..")
            else:
                normalized_tokens = LemNormalize(' '.join(all_questions))
                answer = response(user_input, all_questions, ques_ans_pairs, normalized_tokens)
                st.session_state.chat_history.append(f"Jane: {answer}")

        for chat in st.session_state.chat_history:
            st.write(chat)
except FileNotFoundError as e:
    st.error(f"Error: {e}")

