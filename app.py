import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
import json
import re
from PIL import Image

# --- CONFIGURATION ---
GEMINI_API_KEY = "AIzaSyCEI0mq0NqOzPbuk5PLP_HAMdnqVEWlhYY"
genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="Gemini Data Extractor", layout="wide")

# --- Initialize Session State ---
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None
if 'error_log' not in st.session_state:
    st.session_state.error_log = []

# --- Helper Functions ---

def clean_json_output(text):
    """Robust JSON extraction from AI response."""
    # Strip markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    # Find the first '[' or '{' and last ']' or '}'
    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def extract_data(image_file, model_name):
    """Processes image with specific error handling for 404s."""
    try:
        # Use the ID directly (the SDK handles 'models/' prefix internally)
        model_id = model_name.split('/')[-1]
        model = genai.GenerativeModel(model_id)
        
        img = Image.open(image_file)
        
        prompt = """
        Read this image carefully. Extract all data labels and their corresponding values.
        Return the result ONLY as a JSON list of objects.
        Example: [{"label": "Unit Code", "value": "00733"}]
        """
        
        response = model.generate_content([prompt, img])
        
        if not response.text:
            return {"error": f"Image {image_file.name} returned empty text."}

        json_str = clean_json_output(response.text)
        data = json.loads(json_str)
        return data if isinstance(data, list) else [data]

    except Exception as e:
        return {"error": f"Failed {image_file.name}: {str(e)}"}

def consolidate_data(all_results):
    """Standardizes columns for the final table."""
    rows = []
    for res in all_results:
        if isinstance(res, list):
            # Many AI responses return [{'label': 'x', 'value': 'y'}]
            # We want to pivot that into a single row
            entry = {}
            for item in res:
                if isinstance(item, dict) and 'label' in item and 'value' in item:
                    k = str(item['label']).lower().strip().replace(" ", "_")
                    entry[k] = item['value']
                elif isinstance(item, dict):
                    # If AI already returned a flat dict
                    for k, v in item.items():
                        clean_k = str(k).lower().strip().replace(" ", "_")
                        entry[clean_k] = v
            if entry: rows.append(entry)
            
    if not rows: return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # logical mapping for your specific image (BDO Box)
    mapping = {
        'dams_box_ref._no': 'box_reference',
        'dams_box_ref_no': 'box_reference',
        'unit_branch_name': 'branch',
        'unit/branch_name': 'branch'
    }
    df = df.rename(columns=mapping)
    return df.groupby(level=0, axis=1).first() # Merge duplicate columns

# --- UI ---
st.title("📦 Universal Data Tabulator")

with st.sidebar:
    st.header("Settings")
    
    # DIAGNOSTIC TOOL
    if st.button("🔍 Scan My Available Models"):
        try:
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            st.success("Your key has access to these models:")
            st.write(models)
        except Exception as e:
            st.error(f"Cannot list models: {e}")

    # Choice of common models
    model_choice = st.selectbox(
        "Choose Model", 
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro-vision"],
        help="If 1.5-flash gives a 404, try gemini-pro-vision"
    )
    
    st.divider()
    if st.button("🗑 Reset App"):
        st.session_state.combined_df = None
        st.session_state.error_log = []
        st.rerun()

# --- Upload & Processing ---
uploaded_files = st.file_uploader("Upload up to 100 Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    num_files = len(uploaded_files)
    if num_files > 100:
        st.error("Too many files! Please limit to 100.")
    elif st.button(f"🚀 Process {num_files} Images"):
        extracted_data = []
        progress_bar = st.progress(0)
        status = st.empty()
        
        # Batching Logic (30 images per batch)
        BATCH_LIMIT = 30
        for i in range(0, num_files, BATCH_LIMIT):
            batch = uploaded_files[i : i + BATCH_LIMIT]
            
            for idx, f in enumerate(batch):
                status.info(f"Reading: {f.name}...")
                res = extract_data(f, model_choice)
                
                if isinstance(res, dict) and "error" in res:
                    st.session_state.error_log.append(res["error"])
                else:
                    extracted_data.append(res)
                
                # Small per-image pause to prevent rate limit
                time.sleep(2.0)
            
            # Progress Update
            progress_bar.progress(min((i + BATCH_LIMIT) / num_files, 1.0))
            
            # Batch Pause (4 seconds as requested)
            if i + BATCH_LIMIT < num_files:
                status.warning("Batch limit reached. Pausing 4s for Free Tier...")
                time.sleep(4)

        status.info("Generating unified table...")
        st.session_state.combined_df = consolidate_data(extracted_data)
        status.success("Done!")

# --- Results ---
if st.session_state.combined_df is not None:
    if not st.session_state.combined_df.empty:
        st.divider()
        st.subheader("📊 Consolidated Data Table")
        st.dataframe(st.session_state.combined_df, use_container_width=True)
        
        csv = st.session_state.combined_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Table (CSV)", data=csv, file_name="extracted_data.csv")
    else:
        st.warning("No data extracted. See errors in the sidebar logs.")

if st.session_state.error_log:
    with st.expander("⚠️ View Error Log"):
        for err in st.session_state.error_log:
            st.write(err)
