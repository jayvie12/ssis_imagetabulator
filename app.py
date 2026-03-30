import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
import json
import re
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Secure Data Extractor", layout="wide")

# --- API KEY CONFIGURATION ---
# The app will look for the key in Streamlit Secrets first, then Sidebar input.
api_key = st.sidebar.text_input("🔑 Enter Gemini API Key", type="password")

# If you want to use Streamlit Secrets (Recommended for Cloud):
# 1. Go to Streamlit Cloud Dashboard -> Settings -> Secrets
# 2. Add: GEMINI_KEY = "your_new_key_here"
if not api_key and "GEMINI_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_KEY"]

if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("Please enter your Gemini API Key in the sidebar to begin.")
    st.stop()

# --- INITIALIZE SESSION STATE ---
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None
if 'error_log' not in st.session_state:
    st.session_state.error_log = []

# --- HELPER FUNCTIONS ---
def clean_json_output(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if match: return match.group(1).strip()
    return text.strip()

def extract_data(image_file, model_name):
    try:
        model_id = model_name.split('/')[-1]
        model = genai.GenerativeModel(model_id)
        img = Image.open(image_file)
        
        prompt = "Extract all data from this image into a JSON list. Return ONLY valid JSON."
        response = model.generate_content([prompt, img])
        
        if not response.text: return {"error": f"Empty response: {image_file.name}"}
        
        json_str = clean_json_output(response.text)
        data = json.loads(json_str)
        return data if isinstance(data, list) else [data]
    except Exception as e:
        return {"error": f"Failed {image_file.name}: {str(e)}"}

def consolidate(all_results):
    rows = []
    for res in all_results:
        if isinstance(res, list):
            entry = {}
            for item in res:
                if isinstance(item, dict):
                    if 'label' in item and 'value' in item:
                        k = str(item['label']).lower().strip().replace(" ", "_")
                        entry[k] = item['value']
                    else:
                        for k, v in item.items():
                            entry[str(k).lower().strip().replace(" ", "_")] = v
            if entry: rows.append(entry)
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Mapping for your specific BDO box images
    synonyms = {'dams_box_ref._no': 'box_ref', 'dams_box_ref_no': 'box_ref'}
    return df.rename(columns=synonyms).groupby(level=0, axis=1).first()

# --- MAIN UI ---
st.title("📸 Universal Image Data Tabulator")

with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    
    if st.button("🔍 Test API Key"):
        try:
            models = [m.name for m in genai.list_models()]
            st.success("Connection Successful!")
        except Exception as e:
            st.error(f"Connection Failed: {e}")

    if st.button("🗑 Reset App"):
        st.session_state.combined_df = None
        st.session_state.error_log = []
        st.rerun()

uploaded_files = st.file_uploader("Upload Images (Max 100)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    if st.button(f"🚀 Process {len(uploaded_files)} Images"):
        extracted = []
        bar = st.progress(0)
        status = st.empty()
        
        for i, f in enumerate(uploaded_files):
            status.info(f"Processing {f.name}...")
            res = extract_data(f, model_choice)
            
            if isinstance(res, dict) and "error" in res:
                st.session_state.error_log.append(res["error"])
            else:
                extracted.append(res)
            
            bar.progress((i + 1) / len(uploaded_files))
            time.sleep(2.0) # Stay under 15 RPM Free Tier limit
            
            # Batch Limiter Pause
            if (i + 1) % 30 == 0 and (i + 1) < len(uploaded_files):
                status.warning("Batch limit reached. Pausing 4 seconds...")
                time.sleep(4)

        st.session_state.combined_df = consolidate(extracted)
        status.success("Processing Complete!")

if st.session_state.combined_df is not None:
    st.divider()
    st.dataframe(st.session_state.combined_df, use_container_width=True)
    csv = st.session_state.combined_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", data=csv, file_name="data.csv")

if st.session_state.error_log:
    with st.expander("⚠️ View Errors"):
        for e in st.session_state.error_log: st.write(e)
