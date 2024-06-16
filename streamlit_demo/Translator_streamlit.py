import model_setup
from model_setup import get_supporter, get_model, SupportTransformer, vi_tokenize, TransformerModel
import streamlit as st

SUPPORTER_FILPATH = './model/support_transformer.pkl'
MODEL_FILEPATH = './model/translator_EnVi.pth'

# Get supporter and model
supporter = get_supporter(SUPPORTER_FILPATH)
model = get_model(MODEL_FILEPATH)

# Streamlit show
st.set_page_config(
    page_title="My Translator",  # Đổi tên tiêu đề của trang
    page_icon=":nazar_amulet:",  # Biểu tượng favicon, có thể dùng emoji hoặc đường dẫn tới hình ảnh
    layout="centered",  # Cách bố trí, có thể là "centered" hoặc "wide"
    initial_sidebar_state="auto"  # Trạng thái ban đầu của sidebar, có thể là "auto", "expanded" hoặc "collapsed"
)

st.title("Transformer Translator")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({'role':'assistant', 'content':"Let's me translate English to Vietnamese for you, bro!"})
    
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        
if prompt := st.chat_input("Translate English to Vietnamese"):
    # input
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    
    # response
    try:
        response = supporter.completely_generate(model, prompt)
        response = "Please try again!" if response == "" else response
    except:
        response = "Please try again!"
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({'role':'assistant', 'content':response})