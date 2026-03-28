import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
import json
import re
from PIL import Image
import io

# --- CONFIGURATION ---
# Your integrated API Key
GEMINI_API_KEY = "AIzaSyCEI0mq0NqOzPbuk5PLP_HAMdnqVEWlhYY"

# Configure the SDK immediately
genai.configure(api_key=GEMINI_API_KEY)

# --- Page Config ---
st.set_page_config(page_title="Gemini Auto-Extractor", layout="wide")

# --- Initialize Session State ---
if 'quota_used' not in st.session_state:
    st.session_state.quota_used = 0
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None
if 'debug_logs' not in st.session_state:
    st.session_state.debug_logs = []

# --- Constants ---
BATCH_LIMIT = 30
PAUSE_TIME = 4 
DAILY_QUOTA_LIMIT = 1500 

# --- Sidebar ---
st.sidebar.title("🛠 Settings & Quota")
st.sidebar.success("✅ API Key Integrated")

model_choice = st.sidebar.selectbox("Select Model", ["gemini-1.5-flash", "gemini-1.5-pro"])

st.sidebar.subheader("Daily Quota Tracker")
st.sidebar.progress(min(st.session_state.quota_used / DAILY_QUOTA_LIMIT, 1.0))
st.sidebar.write(f"Requests used: {st.session_state.quota_used} / {DAILY_QUOTA_LIMIT}")

if st.sidebar.button("Reset App / Clear Data"):
    st.session_state.combined_df = None
    st.session_state.debug_logs = []
    st.session_state.quota_used = 0
    st.rerun()

# --- Extraction Logic ---
def clean_json_string(text):
    """Removes markdown backticks and extra text from AI response."""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()

def extract_from_gemini(image_file):
    """Sends image to Gemini and returns a list of dictionaries."""
    try:
        model = genai.GenerativeModel(model_choice)
        img = Image.open(image_file)
        
        prompt = """
        Extract all data from this image into a structured JSON array.
        Rules:
        1. Return ONLY a valid JSON array of objects (e.g., [{"field": "value"}]).
        2. Use logical keys (e.g., 'date', 'total', 'item_name', 'quantity').
        3. If there are multiple rows/items, create one object per item.
        4. No conversational text, no markdown.
        """
        
        response = model.generate_content([prompt, img])
        
        if not response.text:
            return None
        
        cleaned_response = clean_json_string(response.text)
        data = json.loads(cleaned_response)
        
        return data if isinstance(data, list) else [data]
    
    except Exception as e:
        st.session_state.debug_logs.append(f"Error in {image_file.name}: {str(e)}")
        return None

def consolidate_data(results_list):
    """Merges all extracted dictionaries into one unified logical table."""
    flat_data = [item for sublist in results_list if sublist for item in sublist]
    
    if not flat_data:
        return pd.DataFrame()

    df = pd.DataFrame(flat_data)
    
    # Standardize column names
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
    
    # Logical mapping to unify similar fields
    mapping = {
        'total_amount': 'total', 'grand_total': 'total', 'price': 'total', 'amount': 'total',
        'date_of_purchase': 'date', 'transaction_date': 'date', 'timestamp': 'date',
        'vendor': 'merchant', 'store': 'merchant', 'company': 'merchant', 'seller': 'merchant',
        'description': 'details', 'item': 'details', 'particulars': 'details', 'item_name': 'details'
    }
    
    df = df.rename(columns=mapping)
    
    # Merge duplicate columns (e.g., if we now have two 'total' columns)
    df = df.groupby(lambda x: x, axis=1).first()
    
    return df

# --- Main App Interface ---
st.title("🌐 Gemini Web Image Tabulator")
st.write("Upload up to 100 images to extract data into one unified CSV file.")

files = st.file_uploader("Upload Images (PNG, JPG, JPEG)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if files:
    num_files = len(files)
    if num_files > 100:
        st.error("Please limit your upload to 100 images.")
    elif st.button(f"🚀 Process {num_files} Images"):
        all_extracted_results = []
        progress_bar = st.progress(0)
        status = st.empty()
        
        for i in range(0, num_files, BATCH_LIMIT):
            batch = files[i : i + BATCH_LIMIT]
            status.info(f"Processing Batch {i//BATCH_LIMIT + 1}...")
            
            for img_file in batch:
                res = extract_from_gemini(img_file)
                if res:
                    all_extracted_results.append(res)
                st.session_state.quota_used += 1
            
            # Update Progress
            progress_bar.progress(min((i + BATCH_LIMIT) / num_files, 1.0))
            
            # Rate Limiter (4s pause for Free Tier)
            if i + BATCH_LIMIT < num_files:
                status.warning(f"Rate Limiter: Pausing {PAUSE_TIME}s to prevent API crash...")
                time.sleep(PAUSE_TIME)

        # Final Consolidation
        status.info("Consolidating all data into unified table...")
        st.session_state.combined_df = consolidate_data(all_extracted_results)
        status.success("Processing Complete!")

# --- Display Results ---
if st.session_state.combined_df is not None:
    if not st.session_state.combined_df.empty:
        st.divider()
        st.subheader("📊 Unified Data Table")
        st.dataframe(st.session_state.combined_df, use_container_width=True)
        
        # Download
        csv = st.session_state.combined_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name="gemini_extracted_data.csv",
            mime="text/csv"
        )
    else:
        st.warning("No data found in images. Check Debug Logs below.")

# --- Debug Section ---
if st.session_state.debug_logs:
    with st.expander("Show Processing Errors"):
        for log in st.session_state.debug_logs:
            st.write(log)
