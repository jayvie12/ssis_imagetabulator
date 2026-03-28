import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
import json
import re
from PIL import Image
import io

# --- Page Config ---
st.set_page_config(page_title="Gemini Data Extractor Pro", layout="wide")

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
api_key = st.sidebar.text_input("Gemini API Key", type="password")
model_choice = st.sidebar.selectbox("Select Model", ["gemini-1.5-flash", "gemini-1.5-pro"])

if api_key:
    genai.configure(api_key=api_key)

st.sidebar.subheader("Quota Tracker")
st.sidebar.progress(min(st.session_state.quota_used / DAILY_QUOTA_LIMIT, 1.0))
st.sidebar.write(f"Used: {st.session_state.quota_used} / {DAILY_QUOTA_LIMIT}")

if st.sidebar.button("Clear Data/Logs"):
    st.session_state.combined_df = None
    st.session_state.debug_logs = []
    st.rerun()

# --- Extraction Logic ---
def clean_json_string(text):
    """Removes markdown backticks and extra text from AI response."""
    # Remove markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()

def extract_from_gemini(image_file):
    """Sends image to Gemini and returns a list of dictionaries."""
    try:
        model = genai.GenerativeModel(model_choice)
        img = Image.open(image_file)
        
        prompt = """
        Act as an OCR and data extraction expert. 
        Extract all relevant data (names, dates, amounts, ID numbers, table rows, etc.) from this image.
        
        Rules:
        1. Return ONLY a valid JSON array of objects.
        2. Use consistent keys (e.g., 'date', 'amount', 'item_name').
        3. If there are multiple items or rows, create an object for each.
        4. If no data is found, return [].
        5. DO NOT include any conversational text or explanations.
        """
        
        response = model.generate_content([prompt, img])
        
        if not response.text:
            return None
        
        cleaned_response = clean_json_string(response.text)
        data = json.loads(cleaned_response)
        
        # Ensure it's always a list
        return data if isinstance(data, list) else [data]
    
    except Exception as e:
        st.session_state.debug_logs.append(f"Error in {image_file.name}: {str(e)}")
        return None

def consolidate_data(results_list):
    """Merges all extracted dictionaries into one clean DataFrame."""
    # Flatten the list of lists
    flat_data = [item for sublist in results_list if sublist for item in sublist]
    
    if not flat_data:
        return pd.DataFrame()

    df = pd.DataFrame(flat_data)
    
    # 1. Standardize column names (lowercase, no spaces)
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
    
    # 2. Logic Consolidation Map
    # This maps common variants found by AI into unified columns
    mapping = {
        'total_amount': 'total', 'grand_total': 'total', 'price': 'total', 'amt': 'total',
        'date_of_purchase': 'date', 'transaction_date': 'date', 'dt': 'date',
        'vendor': 'merchant', 'store': 'merchant', 'company_name': 'merchant', 'seller': 'merchant',
        'description': 'details', 'item': 'details', 'particulars': 'details'
    }
    
    # Rename columns based on mapping
    df = df.rename(columns=mapping)
    
    # 3. Merge duplicate columns (if 'total' and 'price' both existed, they are now both 'total')
    # We take the first non-null value for each merged column group
    df = df.groupby(lambda x: x, axis=1).first()
    
    return df

# --- Main App Interface ---
st.title("📸 Universal Image Data Tabulator")
st.markdown("""
This app processes up to 100 images, extracts text into data using Gemini AI, 
and merges them into a **single logical table**.
""")

files = st.file_uploader("Upload Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if files:
    if len(files) > 100:
        st.error("Please limit your upload to 100 images.")
    elif st.button(f"Extract Data from {len(files)} Images"):
        if not api_key:
            st.warning("Please enter your Gemini API Key in the sidebar.")
        else:
            all_extracted_results = []
            progress_bar = st.progress(0)
            status = st.empty()
            
            for i in range(0, len(files), BATCH_LIMIT):
                batch = files[i : i + BATCH_LIMIT]
                status.info(f"Processing batch {i//BATCH_LIMIT + 1}...")
                
                for img_file in batch:
                    res = extract_from_gemini(img_file)
                    if res:
                        all_extracted_results.append(res)
                    st.session_state.quota_used += 1
                
                # Update UI Progress
                progress_bar.progress(min((i + BATCH_LIMIT) / len(files), 1.0))
                
                # Rate Limiter for Free Tier
                if i + BATCH_LIMIT < len(files):
                    status.warning("Rate Limiter: Waiting 4 seconds to prevent API crash...")
                    time.sleep(PAUSE_TIME)

            # Final Consolidation
            status.info("Consolidating columns and cleaning data...")
            st.session_state.combined_df = consolidate_data(all_extracted_results)
            status.success("Processing Complete!")

# --- Display Results ---
if st.session_state.combined_df is not None:
    if not st.session_state.combined_df.empty:
        st.subheader("📊 Unified Results Table")
        st.dataframe(st.session_state.combined_df, use_container_width=True)
        
        # Download
        csv = st.session_state.combined_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name="extracted_data.csv", mime="text/csv")
    else:
        st.warning("No data could be extracted. Check 'Debug Logs' in the sidebar.")

# --- Debug Section ---
if st.session_state.debug_logs:
    with st.expander("See Debug Logs (Errors)"):
        for log in st.session_state.debug_logs:
            st.write(log)
