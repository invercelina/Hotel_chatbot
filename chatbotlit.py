import streamlit as st
import pandas as pd
import nltk
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

# 파일 경로
questions_file_path = '[Dataset] Module27(ques).txt'
answers_file_path = '[Dataset] Module27 (ans).txt'

# 파일 읽기
with open(questions_file_path, 'r', encoding='utf-8') as file:
    raw_data_ques = file.read().lower()
    
with open(answers_file_path, 'r', encoding='utf-8') as file:
    raw_data_ans = file.read().lower()

# 데이터 프레임으로 변환
questions = nltk.sent_tokenize(raw_data_ques)
answers = nltk.sent_tokenize(raw_data_ans)
qa_df = pd.DataFrame({'question': questions, 'answer': answers})

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_sentence_embeddings(sentences):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    return sentence_embeddings

# 전체 데이터셋 질문 임베딩 생성
embeddings = get_sentence_embeddings(qa_df['question'].tolist())

def find_most_similar_question(input_question):
    input_embedding = get_sentence_embeddings([input_question])
    similarities = cosine_similarity(input_embedding, embeddings)
    most_similar_idx = similarities.argmax()
    return qa_df.iloc[most_similar_idx], similarities[0][most_similar_idx]

def chatbot_response(user_input):
    most_similar_qa, similarity = find_most_similar_question(user_input)
    return most_similar_qa['answer'], similarity

# Streamlit 인터페이스
st.title("Hotel Information Chatbot")
st.write("Ask a question about the hotel services, facilities, and more!")

chat_history = []

user_input = st.text_input("Your question:")
if user_input:
    response, similarity = chatbot_response(user_input)
    chat_history.append(f"User: {user_input}\n")
    chat_history.append(f"Chatbot: {response} (Similarity: {similarity:.4f})\n")

if chat_history:
    st.write("\n".join(chat_history))
