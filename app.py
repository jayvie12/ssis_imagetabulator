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

st.set_page_config(page_title="Gemini Data Extractor Pro", layout="wide")

# --- Initialize Session State ---
if 'quota_used' not in st.session_state:
    st.session_state.quota_used = 0
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None
if 'error_log' not in st.session_state:
    st.session_state.error_log = []

# --- Helper Functions ---

def find_json_in_text(text):
    """Extracts JSON array or object from text even if AI adds conversational filler."""
    try:
        # Look for the first '[' and last ']'
        match = re.search(r'(\[.*\])', text, re.DOTALL)
        if match:
            return match.group(1)
        # Look for the first '{' and last '}'
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            return match.group(1)
        return text
    except:
        return text

def extract_data(image_file, model_name):
    """Processes image with safety checks and error handling."""
    try:
        model = genai.GenerativeModel(model_name)
        img = Image.open(image_file)
        
        # Optimized Prompt for logic
        prompt = "Extract all data from this image into a structured JSON array. Return ONLY the JSON. Use clear keys like 'date', 'item', 'total'."
        
        response = model.generate_content([prompt, img])
        
        # Check if response was blocked by safety filters
        if not response.candidates or not response.candidates[0].content.parts:
            return {"error": f"Image {image_file.name} was blocked by Safety Filters or returned empty."}

        raw_text = response.text
        clean_json = find_json_in_text(raw_text)
        data = json.loads(clean_json)
        
        return data if isinstance(data, list) else [data]

    except Exception as e:
        return {"error": f"Failed {image_file.name}: {str(e)}"}

def consolidate(all_results):
    """Merges disparate data into one clean table."""
    flat_data = []
    for res in all_results:
        if isinstance(res, list):
            flat_data.extend(res)
            
    if not flat_data:
        return pd.DataFrame()

    df = pd.DataFrame(flat_data)
    # Cleanup column names
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
    
    # Logical Merging
    synonyms = {
        'total_amount': 'total', 'grand_total': 'total', 'price': 'total', 'amt': 'total',
        'date_of_purchase': 'date', 'transaction_date': 'date', 'vendor': 'merchant', 
        'store': 'merchant', 'description': 'details', 'item_name': 'details'
    }
    df = df.rename(columns=synonyms)
    # Group identical columns together
    df = df.groupby(lambda x: x, axis=1).first()
    return df

# --- UI ---
st.title("🚀 Robust Image-to-Table Extractor")
st.sidebar.header("System Status")
st.sidebar.info("API Key: Loaded")
model_choice = st.sidebar.selectbox("Brain Level", ["gemini-1.5-flash", "gemini-1.5-pro"])
st.sidebar.warning("Note: Flash is faster; Pro is smarter.")

if st.sidebar.button("🗑 Clear All Data"):
    st.session_state.combined_df = None
    st.session_state.error_log = []
    st.rerun()

uploaded_files = st.file_uploader("Upload up to 100 images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 100:
        st.error("Maximum 100 images allowed.")
    elif st.button(f"Begin Extraction ({len(uploaded_files)} images)"):
        all_data = []
        progress_bar = st.progress(0)
        status = st.empty()
        
        batch_size = 30
        total = len(uploaded_files)

        for i in range(0, total, batch_size):
            batch = uploaded_files[i : i + batch_size]
            
            for idx, img_file in enumerate(batch):
                status.info(f"Processing image {i + idx + 1} of {total}...")
                
                result = extract_data(img_file, model_choice)
                
                if isinstance(result, dict) and "error" in result:
                    st.session_state.error_log.append(result["error"])
                else:
                    all_data.append(result)
                
                st.session_state.quota_used += 1
                
                # CRITICAL: 1.5s delay between images to prevent 15 RPM Rate Limit crash
                time.sleep(1.5)
            
            # Update Progress
            progress_bar.progress(min((i + batch_size) / total, 1.0))
            
            # Batch Pause
            if i + batch_size < total:
                status.warning("Batch complete. Cooling down for 4 seconds...")
                time.sleep(4)

        status.success("Consolidating data...")
        st.session_state.combined_df = consolidate(all_data)
        status.empty()

# --- Results ---
if st.session_state.combined_df is not None:
    if not st.session_state.combined_df.empty:
        st.subheader("📋 Consolidated Table")
        st.dataframe(st.session_state.combined_df, use_container_width=True)
        
        csv = st.session_state.combined_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name="extracted_data.csv")
    else:
        st.error("No data could be extracted. See error log below.")

if st.session_state.error_log:
    with st.expander("⚠️ View Extraction Errors/Skipped Files"):
        for err in st.session_state.error_log:
            st.write(err)
