import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
import json
from PIL import Image
import io

# --- Page Config ---
st.set_page_config(page_title="Gemini Web Extractor", layout="wide")

# --- Initialize Session State ---
if 'quota_used' not in st.session_state:
    st.session_state.quota_used = 0
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None

# --- Constants ---
BATCH_LIMIT = 30
PAUSE_TIME = 4 
DAILY_QUOTA_LIMIT = 1500 

# --- Sidebar ---
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Get one at aistudio.google.com")

if api_key:
    genai.configure(api_key=api_key)

st.sidebar.subheader("Live Quota Tracker")
quota_pct = min(st.session_state.quota_used / DAILY_QUOTA_LIMIT, 1.0)
st.sidebar.progress(quota_pct)
st.sidebar.write(f"Requests used: {st.session_state.quota_used} / {DAILY_QUOTA_LIMIT}")

# --- Helper Functions ---
def process_image(image_file):
    model = genai.GenerativeModel('gemini-1.5-flash')
    img = Image.open(image_file)
    
    prompt = """
    Extract all data from this image into a structured JSON format. 
    Consolidate similar fields (e.g., use 'total' instead of 'grand_total').
    Return ONLY the JSON. No markdown formatting.
    """
    
    try:
        response = model.generate_content([prompt, img])
        clean_text = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(clean_text)
        return data if isinstance(data, list) else [data]
    except Exception as e:
        return None

def consolidate_data(all_results):
    flat_list = [item for sublist in all_results if sublist for item in sublist]
    if not flat_list: return pd.DataFrame()
    
    df = pd.DataFrame(flat_list)
    # Basic logic to merge similar column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    
    # Mapping common variations to unified names
    synonyms = {
        'total_amount': 'total', 'grand_total': 'total', 'price': 'total',
        'date_of_purchase': 'date', 'transaction_date': 'date', 'timestamp': 'date',
        'vendor': 'merchant', 'store_name': 'merchant', 'company': 'merchant'
    }
    df = df.rename(columns=synonyms)
    # Merge columns with same name after renaming
    df = df.groupby(level=0, axis=1).first() 
    return df

# --- Main Interface ---
st.title("🌐 Gemini Web Image Tabulator")
st.info("Upload up to 100 images. The app will process them in batches of 30 with a 4s delay to stay within the Free Tier limits.")

uploaded_files = st.file_uploader("Upload Images (Max 100)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 100:
        st.error("Too many files! Please limit to 100.")
    elif st.button("🚀 Start Processing"):
        if not api_key:
            st.error("Please enter your API Key in the sidebar first!")
        else:
            all_data = []
            progress_bar = st.progress(0)
            status = st.empty()
            
            total_files = len(uploaded_files)
            
            for i in range(0, total_files, BATCH_LIMIT):
                batch = uploaded_files[i : i + BATCH_LIMIT]
                status.info(f"Processing batch starting at image {i+1}...")
                
                for img_file in batch:
                    res = process_image(img_file)
                    if res:
                        all_data.append(res)
                    st.session_state.quota_used += 1
                
                # Update progress
                current_progress = min((i + BATCH_LIMIT) / total_files, 1.0)
                progress_bar.progress(current_progress)
                
                # Rate Limiter Pause
                if i + BATCH_LIMIT < total_files:
                    status.warning(f"Batch complete. Waiting {PAUSE_TIME}s to respect Free Tier limits...")
                    time.sleep(PAUSE_TIME)
            
            st.session_state.combined_df = consolidate_data(all_data)
            status.success("Done! See data below.")

if st.session_state.combined_df is not None:
    st.divider()
    st.dataframe(st.session_state.combined_df)
    
    csv = st.session_state.combined_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Unified CSV", data=csv, file_name="extracted_data.csv", mime="text/csv")