import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
import json
import re
from PIL import Image

# --- CONFIGURATION ---
# Your integrated API Key
GEMINI_API_KEY = "AIzaSyCEI0mq0NqOzPbuk5PLP_HAMdnqVEWlhYY"
genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="Gemini Data Extractor", layout="wide")

# --- Initialize Session State ---
if 'quota_used' not in st.session_state:
    st.session_state.quota_used = 0
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None
if 'error_log' not in st.session_state:
    st.session_state.error_log = []

# --- Helper Functions ---

def find_json_in_text(text):
    """Clean AI response to find valid JSON."""
    try:
        match = re.search(r'(\[.*\])', text, re.DOTALL)
        if match: return match.group(1)
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match: return match.group(1)
        return text
    except:
        return text

def extract_data(image_file, model_name):
    """Processes image with model name safety."""
    try:
        # We strip 'models/' prefix if it exists to prevent double-prefixing
        model_id = model_name.replace("models/", "")
        model = genai.GenerativeModel(model_id)
        
        img = Image.open(image_file)
        
        prompt = "Extract all fields from this image into a JSON array. Return ONLY the JSON."
        
        response = model.generate_content([prompt, img])
        
        if not response or not hasattr(response, 'text'):
            return {"error": f"No response for {image_file.name}. Check safety filters."}

        clean_json = find_json_in_text(response.text)
        data = json.loads(clean_json)
        return data if isinstance(data, list) else [data]

    except Exception as e:
        return {"error": f"Error on {image_file.name}: {str(e)}"}

def consolidate(all_results):
    flat_data = []
    for res in all_results:
        if isinstance(res, list): flat_data.extend(res)
            
    if not flat_data: return pd.DataFrame()
    df = pd.DataFrame(flat_data)
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
    
    # Unified Mapping
    synonyms = {
        'dams_box_ref_no': 'box_ref', 'unit_code': 'unit_code', 
        'unit/branch_name': 'branch', 'dis_date': 'disposal_date'
    }
    df = df.rename(columns=synonyms)
    return df.groupby(lambda x: x, axis=1).first()

# --- UI ---
st.title("📸 Data Extractor")

# Model Selection - Using standard stable names
available_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro-vision"]
model_choice = st.sidebar.selectbox("Select Model", available_models)

if st.sidebar.button("Show Available Models"):
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        st.sidebar.write(models)
    except Exception as e:
        st.sidebar.error(f"Could not list models: {e}")

if st.sidebar.button("🗑 Reset"):
    st.session_state.combined_df = None
    st.session_state.error_log = []
    st.session_state.quota_used = 0
    st.rerun()

uploaded_files = st.file_uploader("Upload Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    if st.button(f"Process {len(uploaded_files)} Images"):
        all_data = []
        progress_bar = st.progress(0)
        status = st.empty()
        
        for idx, img_file in enumerate(uploaded_files):
            status.info(f"Processing {idx+1}/{len(uploaded_files)}...")
            result = extract_data(img_file, model_choice)
            
            if isinstance(result, dict) and "error" in result:
                st.session_state.error_log.append(result["error"])
            else:
                all_data.append(result)
            
            # Quota/Rate Limit protection
            st.session_state.quota_used += 1
            progress_bar.progress((idx + 1) / len(uploaded_files))
            time.sleep(2.0) # 2s per image is safest for Free Tier

        st.session_state.combined_df = consolidate(all_data)
        status.success("Done!")

if st.session_state.combined_df is not None:
    if not st.session_state.combined_df.empty:
        st.subheader("📊 Results")
        st.dataframe(st.session_state.combined_df, use_container_width=True)
        csv = st.session_state.combined_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name="data.csv")

if st.session_state.error_log:
    with st.expander("⚠️ View Errors"):
        for err in st.session_state.error_log: st.write(err)
