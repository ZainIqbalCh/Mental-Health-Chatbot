import streamlit as st
import pip 
import os
import base64
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import InferenceClient
import os
import base64


model_name = "qazws345/DialogGPT_Psycho8k"
tokenizer = AutoTokenizer.from_pretrained(model_name)   
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)



def chat_with_model(user_input):

    chat_history = []      
    chat_history.append({"role": "user", "content": user_input})
    inputs = tokenizer(user_input + tokenizer.eos_token, return_tensors='pt').to(device)

    chat_history_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],  
            max_length=1500,
            pad_token_id=tokenizer.eos_token_id, 
            do_sample=True,  
            temperature=0.6,
            top_p=0.92,
            top_k=50
        )

    bot_response = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
    bot_response = bot_response[len(user_input):].strip()

    chat_history.append({"role": "bot", "content": bot_response})
    return bot_response


# API Key


st.set_page_config(
    page_title='Chat with Gemini',

    layout='centered',

)

st.markdown(f"""
    <style>
    .stApp {{
  
        background-position: center;
        background-repeat: no-repeat;
        background-size: cover;
        opacity: 0.9;
        height: 100vh;
    }}
    </style>
""", unsafe_allow_html=True)



def translate_role_streamlit(user_role):
    if user_role=='model':
        return 'assistant'
    else:
        return user_role
    
with st.sidebar:
    st.header('Welcome to Your AI Counsellor')
st.markdown("""
    <style>
    .main {
        max-height: 100vh;
        overflow-y: scroll;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
        background: #000000;

        padding-top: 20px;
    }

    /* Chat bubbles styling */
    .user-message {
        text-align: right;
        background-color: 	#008000;
        
        border-color: #c0c0c0;
        color: white;
        padding: 8px;
        border-radius: 10px;
        margin: 10px;
        width: fit-content;
        float: right;
        clear: both;
    }
    h1 {
        text-align: center;
        font-size: 2.5rem;
        color: white;
    }
    .ai-message {
        text-align: left;
        background-color: #282828;
        color: #FFFFFF;
        padding: 8px;
        border-radius: 10px;
        margin: 10px;
        width: fit-content;
        float: left;
        clear: both;
    }
    footer { 
        visibility: hidden;
    }
        .custom-footer {
        background-color: #f5f5f5;
        color: #000000;
        text-align: center;
        padding: 10px;
        position: fixed;
        bottom: 0;
        width: 100%;
        border-top: 2px solid #ccc;
    }

    </style>
""", unsafe_allow_html=True)

if 'chat_session' not in st.session_state:
  
    st.session_state.chat_session = {"history": []}

st.markdown("# ✨✨ EmpathAI ✨✨")



# Display all the previous messages in the chat history
for message in st.session_state.chat_session["history"]:
    if message['role'] == 'user':
        user_html = f"<div class='user-message'>{message['content']}</div>"
        st.markdown(user_html, unsafe_allow_html=True)
    else:
        ai_html = f"<div class='ai-message'>{message['content']}</div>"
        st.markdown(ai_html, unsafe_allow_html=True)

        
user_input=st.chat_input('Ask Therapy Bot')
if user_input:
    user_html = f"<div class='user-message'>{user_input}</div>"
    st.markdown(user_html, unsafe_allow_html=True)
    
    # Get response from model
    bot_response = chat_with_model(user_input)

    # Display bot response
    ai_html = f"<div class='ai-message'>{bot_response}</div>"
    st.markdown(ai_html, unsafe_allow_html=True)
    
    # Save conversation history
    st.session_state.chat_session["history"].append({'role': 'user', 'content': user_input})
    st.session_state.chat_session["history"].append({'role': 'bot', 'content': bot_response})
