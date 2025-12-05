import random
import re
from threading import Thread

import torch
import numpy as np
import streamlit as st

st.set_page_config(page_title="MiniMind", initial_sidebar_state="collapsed")

# ================================
# CSS UI Style Overrides
# ================================
st.markdown("""
    <style>
        /* Button styling */
        .stButton button {
            border-radius: 50% !important;
            width: 32px !important;
            height: 32px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;
            margin: 5px 10px 5px 0 !important;
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }

        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }

        /* Reset basic button styling */
        .stButton > button {
            all: unset !important;
            box-sizing: border-box !important;
            border-radius: 50% !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin: 0 2px !important;
        }

    </style>
""", unsafe_allow_html=True)

system_prompt = []
device = "cuda" if torch.cuda.is_available() else "cpu"


# ===================================================
# Process assistant output (handle <think> tags)
# ===================================================
def process_assistant_content(content):
    if model_source == "API" and 'R1' not in api_model_name:
        return content
    if model_source != "API" and 'R1' not in MODEL_PATHS[selected_model][1]:
        return content

    # collapse think sections
    if '<think>' in content and '</think>' in content:
        content = re.sub(
            r'(<think>)(.*?)(</think>)',
            r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">Reasoning (Click to expand)</summary>\2</details>',
            content, flags=re.DOTALL
        )

    if '<think>' in content and '</think>' not in content:
        content = re.sub(
            r'<think>(.*?)$',
            r'<details open style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">Reasoning...</summary>\1</details>',
            content, flags=re.DOTALL
        )

    if '<think>' not in content and '</think>' in content:
        content = re.sub(
            r'(.*?)</think>',
            r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">Reasoning (Click to expand)</summary>\1</details>',
            content, flags=re.DOTALL
        )

    return content


# ===================================================
# Load model + tokenizer (cached)
# ===================================================
@st.cache_resource
def load_model_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.eval().to(device)
    return model, tokenizer


def clear_chat_messages():
    del st.session_state.messages
    del st.session_state.chat_messages


# ===================================================
# Initialize chat session
# ===================================================
def init_chat_messages():
    if "messages" in st.session_state:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                with st.chat_message("assistant", avatar=image_url):
                    st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                    if st.button("🗑", key=f"delete_{i}"):
                        st.session_state.messages.pop(i)
                        st.session_state.chat_messages.pop(i)
                        st.rerun()
            else:
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end;"><div style="padding: 8px 12px; background-color: #ddd; border-radius: 10px;">{message["content"]}</div></div>',
                    unsafe_allow_html=True)
    else:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    return st.session_state.messages


def regenerate_answer(index):
    st.session_state.messages.pop()
    st.session_state.chat_messages.pop()
    st.rerun()


# Sidebar settings
st.sidebar.title("Model Configuration")

st.session_state.history_chat_num = st.sidebar.slider("Number of Historical Dialogues", 0, 6, 0, step=2)
st.session_state.max_new_tokens = st.sidebar.slider("Max Sequence Length", 256, 8192, 8192)
st.session_state.temperature = st.sidebar.slider("Temperature", 0.6, 1.2, 0.85)

model_source = st.sidebar.radio("Choose Model Source", ["Local Model", "API"], index=0)

if model_source == "API":
    api_url = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000/v1")
    api_model_id = st.sidebar.text_input("Model ID", value="minimind")
    api_model_name = st.sidebar.text_input("Model Name", value="MiniMind2")
    api_key = st.sidebar.text_input("API Key", value="none", type="password")
    slogan = f"Hi, I'm {api_model_name}"
else:
    MODEL_PATHS = {
        "MiniMind2-R1 (0.1B)": ["../MiniMind2-R1", "MiniMind2-R1"],
        "MiniMind2-Small-R1 (0.02B)": ["../MiniMind2-Small-R1", "MiniMind2-Small-R1"],
        "MiniMind2 (0.1B)": ["../MiniMind2", "MiniMind2"],
        "MiniMind2-MoE (0.15B)": ["../MiniMind2-MoE", "MiniMind2-MoE"],
        "MiniMind2-Small (0.02B)": ["../MiniMind2-Small", "MiniMind2-Small"]
    }

    selected_model = st.sidebar.selectbox("Models", list(MODEL_PATHS.keys()), index=2)
    model_path = MODEL_PATHS[selected_model][0]
    slogan = f"Hi, I'm {MODEL_PATHS[selected_model][1]}"

image_url = "https://www.modelscope.cn/api/v1/studio/gongjy/MiniMind/repo?Revision=master&FilePath=images%2Flogo2.png&View=true"

# Header
st.markdown(
    f"""
    <div style="text-align:center;">
        <div style="font-style: italic; font-weight: 900;">
            <img src="{image_url}" style="width: 45px; height: 45px;">
            <span style="font-size: 26px;">{slogan}</span>
        </div>
        <span style="color: #bbb; font-style: italic;">
            Content generated by AI — Please verify carefully
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# ===================================================
# Seed control
# ===================================================
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===================================================
# Main Chat Logic
# ===================================================
def main():
    if model_source == "Local Model":
        model, tokenizer = load_model_tokenizer(model_path)
    else:
        model, tokenizer = None, None

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    messages = st.session_state.messages

    # Render chat history
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar=image_url):
                st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="padding: 8px 12px; background-color: gray; border-radius: 10px; color:white;">{message["content"]}</div></div>',
                unsafe_allow_html=True)

    # User input
    prompt = st.chat_input(placeholder="Send a message to MiniMind")

    if prompt:
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;"><div style="padding: 8px 12px; background-color: gray; border-radius: 10px; color:white;">{prompt}</div></div>',
            unsafe_allow_html=True)

        messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # Assistant response
        with st.chat_message("assistant", avatar=image_url):
            placeholder = st.empty()

            # ==========================================
            # API MODE
            # ==========================================
            if model_source == "API":
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key, base_url=api_url)

                    history_num = st.session_state.history_chat_num + 1
                    conversation_history = system_prompt + st.session_state.chat_messages[-history_num:]

                    answer = ""
                    response = client.chat.completions.create(
                        model=api_model_id,
                        messages=conversation_history,
                        stream=True,
                        temperature=st.session_state.temperature
                    )

                    for chunk in response:
                        content = chunk.choices[0].delta.content or ""
                        answer += content
                        placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)

                except Exception as e:
                    answer = f"API request error: {str(e)}"
                    placeholder.markdown(answer, unsafe_allow_html=True)

            # ==========================================
            # LOCAL MODEL MODE
            # ==========================================
            else:
                random_seed = random.randint(0, 2 ** 32 - 1)
                setup_seed(random_seed)

                st.session_state.chat_messages = (
                    system_prompt + st.session_state.chat_messages[-(st.session_state.history_chat_num + 1):]
                )

                new_prompt = tokenizer.apply_chat_template(
                    st.session_state.chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

                generation_kwargs = {
                    "input_ids": inputs.input_ids,
                    "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
                    "do_sample": True,
                    "temperature": st.session_state.temperature,
                    "top_p": 0.85,
                    "streamer": streamer,
                }

                Thread(target=model.generate, kwargs=generation_kwargs).start()

                answer = ""
                for new_text in streamer:
                    answer += new_text
                    placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)

            # Save assistant response
            messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    main()
