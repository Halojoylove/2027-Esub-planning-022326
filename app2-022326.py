import os
import yaml
import base64
import random
import concurrent.futures
from datetime import datetime
from io import BytesIO
from pypdf import PdfReader

import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import httpx

# ==========================================
# 1. Configuration & Constants
# ==========================================

st.set_page_config(page_title="FDA 510(k) Agentic Reviewer", layout="wide", page_icon="🔬")

ALL_MODELS = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "claude-3-5-sonnet-2024-10",
    "claude-3-5-haiku-20241022",
    "grok-4-fast-reasoning"
]

# 30 Musician Styles - Custom CSS injection mapping
MUSICIAN_STYLES = {
    "Mozart": {"bg": "linear-gradient(135deg, #fdfcf0, #f4e8c1)", "text": "#3e2723", "border": "#d4af37", "font": "Georgia"},
    "Beethoven": {"bg": "linear-gradient(135deg, #2c3e50, #000000)", "text": "#ecf0f1", "border": "#c0392b", "font": "Georgia"},
    "Bach": {"bg": "linear-gradient(to right, #eaddcf, #d5c3aa)", "text": "#2c3e50", "border": "#8b4513", "font": "serif"},
    "Chopin": {"bg": "linear-gradient(135deg, #fff0f5, #ffe4e1)", "text": "#4a4a4a", "border": "#ffb6c1", "font": "serif"},
    "Vivaldi": {"bg": "linear-gradient(45deg, #a8e6cf, #dcedc1, #ffd3b6, #ffaaa5)", "text": "#333", "border": "#ff8b94", "font": "serif"},
    "Miles Davis": {"bg": "linear-gradient(to bottom, #0f2027, #203a43, #2c5364)", "text": "#ece9e6", "border": "#4ca1af", "font": "monospace"},
    "John Coltrane": {"bg": "radial-gradient(circle, #301847, #000000)", "text": "#d4af37", "border": "#b8860b", "font": "monospace"},
    "Louis Armstrong": {"bg": "linear-gradient(to right, #ffecd2 0%, #fcb69f 100%)", "text": "#3e2723", "border": "#d35400", "font": "sans-serif"},
    "Billie Holiday": {"bg": "linear-gradient(135deg, #232526 0%, #414345 100%)", "text": "#e0e0e0", "border": "#8e44ad", "font": "serif"},
    "B.B. King": {"bg": "linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)", "text": "#f1c40f", "border": "#f39c12", "font": "sans-serif"},
    "The Beatles": {"bg": "linear-gradient(135deg, #ffffff, #e0e0e0)", "text": "#000000", "border": "#333333", "font": "Arial"},
    "Freddie Mercury": {"bg": "linear-gradient(135deg, #e52d27 0%, #b31217 100%)", "text": "#ffffff", "border": "#f1c40f", "font": "sans-serif"},
    "David Bowie": {"bg": "linear-gradient(45deg, #000000 30%, #ff0000 40%, #0000ff 60%, #000000 70%)", "text": "#ffffff", "border": "#f1c40f", "font": "sans-serif"},
    "Jimi Hendrix": {"bg": "linear-gradient(to right, #4b1248, #f0c27b)", "text": "#ffffff", "border": "#9b59b6", "font": "sans-serif"},
    "Elvis Presley": {"bg": "linear-gradient(135deg, #000000, #434343)", "text": "#d4af37", "border": "#ffd700", "font": "sans-serif"},
    "Michael Jackson": {"bg": "linear-gradient(135deg, #000000 40%, #ffffff 40%, #ffffff 60%, #ff0000 60%)", "text": "#111", "border": "#000", "font": "sans-serif"},
    "Prince": {"bg": "radial-gradient(circle, #8e44ad, #2c3e50)", "text": "#f1c40f", "border": "#9b59b6", "font": "sans-serif"},
    "Pink Floyd": {"bg": "linear-gradient(to right, #000 20%, #ff0000 30%, #ffa500 40%, #ffff00 50%, #008000 60%, #0000ff 70%, #4b0082 80%, #ee82ee 90%, #000 100%)", "text": "#fff", "border": "#fff", "font": "sans-serif"},
    "Led Zeppelin": {"bg": "linear-gradient(135deg, #4e4376, #2b5876)", "text": "#d4af37", "border": "#bdc3c7", "font": "sans-serif"},
    "Kurt Cobain": {"bg": "repeating-linear-gradient(45deg, #1b5e20, #1b5e20 20px, #000000 20px, #000000 40px)", "text": "#ffffff", "border": "#c62828", "font": "monospace"},
    "Daft Punk": {"bg": "linear-gradient(135deg, #000000, #111111)", "text": "#00ffcc", "border": "#ff00ff", "font": "Courier New"},
    "Kraftwerk": {"bg": "linear-gradient(to bottom, #ff0000, #800000)", "text": "#ffffff", "border": "#000000", "font": "Courier New"},
    "Björk": {"bg": "linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%)", "text": "#333", "border": "#ffffff", "font": "sans-serif"},
    "Taylor Swift": {"bg": "linear-gradient(135deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%)", "text": "#4a4a4a", "border": "#ffb6c1", "font": "sans-serif"},
    "Lady Gaga": {"bg": "linear-gradient(135deg, #fc00ff, #00dbde)", "text": "#ffffff", "border": "#ffff00", "font": "sans-serif"},
    "The Weeknd": {"bg": "linear-gradient(to bottom, #000000, #430000)", "text": "#ff0000", "border": "#ff0000", "font": "sans-serif"},
    "Billie Eilish": {"bg": "linear-gradient(135deg, #000000, #111111)", "text": "#00ff00", "border": "#00ff00", "font": "sans-serif"},
    "Bob Marley": {"bg": "linear-gradient(to bottom, #198754, #ffc107, #dc3545)", "text": "#ffffff", "border": "#000000", "font": "sans-serif"},
    "Hans Zimmer": {"bg": "radial-gradient(circle at 50% 50%, #1a2a6c, #112b3c, #000000)", "text": "#ecf0f1", "border": "#3498db", "font": "sans-serif"},
    "Yo-Yo Ma": {"bg": "linear-gradient(to right, #8e4a23, #d35400)", "text": "#ffffff", "border": "#f39c12", "font": "serif"},
}

LABELS = {
    "en": {
        "title": "🔬 FDA 510(k) Agentic AI Reviewer",
        "tab1": "📂 Upload & Preview",
        "tab2": "🤖 Agent Studio",
        "tab3": "🚀 Advanced AI Hub",
        "upload": "Upload Document (PDF, TXT, MD)",
        "paste": "Or Paste Document Text Here",
        "agent": "Select Review Agent",
        "model": "Select AI Model",
        "prompt": "Customize Agent Prompt",
        "run": "▶ Execute Agent",
        "jackpot": "🎰 Style Jackpot!",
        "parallel_title": "⚡ Parallel Auto-Orchestrator",
        "parallel_desc": "Select multiple agents to process the document simultaneously.",
        "search_title": "🧭 Regulatory Radar (Q&A)",
        "search_desc": "Ask specific regulatory questions against the document context.",
        "memo_title": "📑 Executive Memo Synthesizer",
        "memo_desc": "Synthesize all outputs into a final, downloadable FDA Review Memo."
    },
    "zh": {
        "title": "🔬 FDA 510(k) 智能代理解析系統",
        "tab1": "📂 上傳與預覽",
        "tab2": "🤖 代理工作室",
        "tab3": "🚀 進階 AI 中心",
        "upload": "上傳文件 (PDF, TXT, MD)",
        "paste": "或在此貼上文件文字",
        "agent": "選擇審查代理 (Agent)",
        "model": "選擇 AI 模型",
        "prompt": "自訂代理提示詞 (Prompt)",
        "run": "▶ 執行代理",
        "jackpot": "🎰 風格大樂透！",
        "parallel_title": "⚡ 並行自動協調器",
        "parallel_desc": "選擇多個代理同時處理文件，大幅節省時間。",
        "search_title": "🧭 法規雷達 (智能問答)",
        "search_desc": "針對文件內容提出特定的法規審查問題。",
        "memo_title": "📑 高層決策備忘錄生成",
        "memo_desc": "將所有代理產出綜合為最終 FDA 審查備忘錄並匯出。"
    }
}

# ==========================================
# 2. Session State Initialization
# ==========================================

if "lang" not in st.session_state: st.session_state.lang = "en"
if "theme" not in st.session_state: st.session_state.theme = "Dark"
if "musician" not in st.session_state: st.session_state.musician = "Daft Punk"
if "doc_text" not in st.session_state: st.session_state.doc_text = ""
if "history" not in st.session_state: st.session_state.history = []
if "api_keys" not in st.session_state: st.session_state.api_keys = {}

if "agents_cfg" not in st.session_state:
    try:
        with open("agents.yaml", "r", encoding="utf-8") as f:
            st.session_state.agents_cfg = yaml.safe_load(f)
    except:
        st.error("agents.yaml not found! Please ensure it is in the same directory.")
        st.stop()

def get_label(key):
    return LABELS[st.session_state.lang][key]

# ==========================================
# 3. Core Functions (LLM & Utils)
# ==========================================

def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        return file.read().decode("utf-8")

def display_pdf(file):
    base64_pdf = base64.b64encode(file.getvalue()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700" type="application/pdf" style="border-radius: 10px; border: 2px solid #555;"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def call_llm(model, system_prompt, user_prompt, max_tokens=8000):
    keys = st.session_state.api_keys
    
    try:
        if "gemini" in model:
            if not keys.get("gemini"): return "⚠️ Gemini API key missing."
            genai.configure(api_key=keys["gemini"])
            llm = genai.GenerativeModel(model)
            resp = llm.generate_content(f"System: {system_prompt}\n\nUser: {user_prompt}")
            return resp.text
            
        elif "gpt" in model:
            if not keys.get("openai"): return "⚠️ OpenAI API key missing."
            client = OpenAI(api_key=keys["openai"])
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content
            
        elif "claude" in model:
            if not keys.get("anthropic"): return "⚠️ Anthropic API key missing."
            client = Anthropic(api_key=keys["anthropic"])
            resp = client.messages.create(
                model=model, system=system_prompt, messages=[{"role": "user", "content": user_prompt}], max_tokens=max_tokens
            )
            return resp.content[0].text
            
        return "Model not implemented."
    except Exception as e:
        return f"Error executing model: {str(e)}"

def run_agent_task(aid, model, doc_text):
    """Function wrapper for concurrent execution"""
    agent_data = st.session_state.agents_cfg["agents"][aid]
    res = call_llm(model, agent_data["system_prompt"], f"Document Text:\n{doc_text[:30000]}") # truncation for safety
    return aid, agent_data["name"], res

# ==========================================
# 4. Sidebar & WOW UI Injection
# ==========================================

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.session_state.lang = st.radio("Language / 語言", ["en", "zh"], index=0 if st.session_state.lang=="en" else 1, horizontal=True)
    st.session_state.theme = st.radio("Light/Dark Mode", ["Light", "Dark"], index=0 if st.session_state.theme=="Light" else 1, horizontal=True)
    
    st.markdown("---")
    st.markdown("### 🎸 Musician UI Styles")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.musician = st.selectbox("Style", list(MUSICIAN_STYLES.keys()), index=list(MUSICIAN_STYLES.keys()).index(st.session_state.musician))
    with col2:
        if st.button("🎰"):
            st.session_state.musician = random.choice(list(MUSICIAN_STYLES.keys()))
            st.toast(f"Jackpot! You unlocked: {st.session_state.musician} 🎸", icon="✨")
            st.rerun()
            
    st.markdown("---")
    st.markdown("### 🔑 API Keys")
    st.session_state.api_keys["openai"] = st.text_input("OpenAI Key", type="password", value=st.session_state.api_keys.get("openai", ""))
    st.session_state.api_keys["gemini"] = st.text_input("Gemini Key", type="password", value=st.session_state.api_keys.get("gemini", ""))
    st.session_state.api_keys["anthropic"] = st.text_input("Anthropic Key", type="password", value=st.session_state.api_keys.get("anthropic", ""))

# CSS Injection for WOW UI
style = MUSICIAN_STYLES[st.session_state.musician]
tint = "rgba(0,0,0,0.8)" if st.session_state.theme == "Dark" else "rgba(255,255,255,0.8)"
text_color = "#fff" if st.session_state.theme == "Dark" else "#000"

st.markdown(f"""
    <style>
    /* Global App Background */
    .stApp {{
        background: {style['bg']} !important;
        font-family: {style['font']}, sans-serif !important;
        background-size: 400% 400% !important;
        animation: gradientBG 15s ease infinite !important;
    }}
    
    @keyframes gradientBG {{
        0% {{background-position: 0% 50%;}}
        50% {{background-position: 100% 50%;}}
        100% {{background-position: 0% 50%;}}
    }}
    
    /* Container Tints */
    .stApp > header {{ background-color: transparent !important; }}
    .block-container {{
        background-color: {tint};
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.5);
        color: {text_color} !important;
        border: 2px solid {style['border']};
        margin-top: 2rem;
    }}

    /* Inputs and Buttons */
    .stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] {{
        background-color: rgba(128,128,128,0.2) !important;
        color: {text_color} !important;
        border: 1px solid {style['border']} !important;
        border-radius: 8px !important;
    }}
    
    .stButton>button {{
        background-color: {style['border']} !important;
        color: #fff !important;
        font-weight: bold !important;
        border-radius: 50px !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }}
    
    .stButton>button:hover {{
        transform: scale(1.05);
        box-shadow: 0px 0px 15px {style['border']};
    }}

    h1, h2, h3, p, span {{ color: {text_color} !important; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 5. Main Application Layout
# ==========================================

st.title(get_label("title"))

tab_upload, tab_agent, tab_advanced = st.tabs([get_label("tab1"), get_label("tab2"), get_label("tab3")])

# ------------------------------------------
# TAB 1: Upload & Preview
# ------------------------------------------
with tab_upload:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Data Ingestion")
        uploaded_file = st.file_uploader(get_label("upload"), type=["pdf", "txt", "md"])
        pasted_text = st.text_area(get_label("paste"), height=250)
        
        if st.button("Load Document Context"):
            if uploaded_file:
                st.session_state.doc_text = extract_text(uploaded_file)
                st.success("File Processed & Loaded!")
            elif pasted_text:
                st.session_state.doc_text = pasted_text
                st.success("Text Loaded!")
            else:
                st.warning("Please upload or paste a document first.")
                
        if st.session_state.doc_text:
            st.info(f"Loaded Context: {len(st.session_state.doc_text)} characters.")
            with st.expander("View Raw Extracted Text"):
                st.text(st.session_state.doc_text[:5000] + "\n\n...[TRUNCATED]")

    with col2:
        st.subheader("👁️ Document Preview")
        if uploaded_file and uploaded_file.name.endswith(".pdf"):
            display_pdf(uploaded_file)
        else:
            st.markdown("""
            <div style="height: 400px; display: flex; align-items: center; justify-content: center; border: 2px dashed gray; border-radius: 10px;">
                <h3 style="color: gray;">PDF Preview Window</h3>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 2: Agent Studio
# ------------------------------------------
with tab_agent:
    st.subheader("🔬 Single Agent Execution")
    
    col_a, col_b = st.columns([2, 1])
    agent_options = list(st.session_state.agents_cfg["agents"].keys())
    
    with col_a:
        selected_agent_id = st.selectbox(get_label("agent"), ["-- Select --"] + agent_options)
    with col_b:
        selected_model = st.selectbox(get_label("model"), ALL_MODELS, index=0)

    if selected_agent_id != "-- Select --":
        agent_data = st.session_state.agents_cfg["agents"][selected_agent_id]
        edited_prompt = st.text_area(get_label("prompt"), value=agent_data["system_prompt"], height=200)
        
        if st.button(get_label("run"), use_container_width=True):
            if not st.session_state.doc_text:
                st.error("Please load a document in Tab 1 first.")
            else:
                with st.spinner(f"Agent {agent_data['name']} is analyzing..."):
                    result = call_llm(selected_model, edited_prompt, f"Doc Context:\n{st.session_state.doc_text}")
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "agent": agent_data["name"],
                        "model": selected_model,
                        "result": result
                    })
                    st.success("Analysis Complete!")
                    st.markdown("### Output")
                    st.markdown(result)

# ------------------------------------------
# TAB 3: Advanced AI Hub (The 3 AWESOME Features)
# ------------------------------------------
with tab_advanced:
    
    # Feature 1: Parallel Auto-Orchestrator
    st.markdown(f"### {get_label('parallel_title')}")
    st.caption(get_label('parallel_desc'))
    
    selected_batch = st.multiselect("Select Agents for Parallel Execution", agent_options)
    batch_model = st.selectbox("Model for Batch", ["gemini-2.5-flash", "gpt-4o-mini", "claude-3-5-haiku-20241022"], key="batch_model")
    
    if st.button("🚀 Run Parallel Analysis"):
        if not st.session_state.doc_text:
            st.error("Please load document text first.")
        elif not selected_batch:
            st.warning("Select at least one agent.")
        else:
            progress_text = "Running agents concurrently..."
            my_bar = st.progress(0, text=progress_text)
            
            results = []
            # TRUE PARALLELISM via ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_batch)) as executor:
                futures = {executor.submit(run_agent_task, aid, batch_model, st.session_state.doc_text): aid for aid in selected_batch}
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    aid, aname, res = future.result()
                    results.append({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "agent": aname, "model": batch_model, "result": res})
                    my_bar.progress((i + 1) / len(selected_batch), text=f"Finished: {aname}")
            
            st.session_state.history.extend(results)
            st.success("All parallel tasks complete! Check outputs below or in the Memo Synthesizer.")
            
            for r in results:
                with st.expander(f"Output: {r['agent']}"):
                    st.markdown(r['result'])

    st.markdown("---")
    
    # Feature 2: Regulatory Radar (Semantic Chat)
    st.markdown(f"### {get_label('search_title')}")
    st.caption(get_label('search_desc'))
    
    radar_q = st.text_input("Ask a specific question (e.g., 'What is the shelf life testing standard used?')", key="radar_q")
    if st.button("🔍 Scan Document"):
        if st.session_state.doc_text:
            with st.spinner("Scanning semantics..."):
                radar_prompt = "You are a FDA 510k semantic search engine. Find the exact answer to the user's question based ONLY on the provided document. Quote the relevant sections."
                radar_res = call_llm("gemini-2.5-flash", radar_prompt, f"Document:\n{st.session_state.doc_text}\n\nQuestion: {radar_q}")
                st.info(radar_res)
        else:
            st.error("Load document first.")

    st.markdown("---")
    
    # Feature 3: Executive Memo Synthesizer
    st.markdown(f"### {get_label('memo_title')}")
    st.caption(get_label('memo_desc'))
    
    if st.button("📑 Generate & Export Executive Memo"):
        if not st.session_state.history:
            st.warning("No agent history found. Run some agents first!")
        else:
            with st.spinner("Synthesizing final executive memo..."):
                # Concatenate all history
                raw_history = ""
                for h in st.session_state.history:
                    raw_history += f"## {h['agent']}\n{h['result']}\n\n"
                
                # Ask LLM to synthesize it
                synth_prompt = "You are an FDA branch chief. Read the following outputs from various AI review agents and synthesize them into a highly professional, 1-page Executive Review Memo."
                executive_summary = call_llm("gemini-3-flash-preview", synth_prompt, raw_history)
                
                # Build final markdown
                final_markdown = f"# FDA 510(k) AI Review Executive Memo\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                final_markdown += "## 🌟 Executive Summary\n" + executive_summary + "\n\n---\n\n"
                final_markdown += "# Detailed Agent Appendices\n\n" + raw_history
                
                st.success("Memo Generated Successfully!")
                st.download_button(
                    label="📥 Download Full Markdown Report",
                    data=final_markdown.encode("utf-8"),
                    file_name=f"FDA_510k_Memo_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
                
                with st.expander("Preview Executive Summary"):
                    st.markdown(executive_summary)
