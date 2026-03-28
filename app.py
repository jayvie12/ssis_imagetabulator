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
        # Fallback to look for the first '{' and last '}'
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            return match.group(1)
        return text
    except:
        return text

def extract_data(image_file, model_name):
    """Processes image with safety checks and error handling."""
    try:
        # Initialize model with the specific name
        model = genai.GenerativeModel(model_name)
        img = Image.open(image_file)
        
        prompt = """
        Extract all data from this image into a structured JSON array. 
        Rules:
        - Return ONLY the JSON.
        - Use logical keys like 'date', 'item', 'quantity', 'total'.
        - If multiple rows exist, create a list of objects.
        """
        
        # Call API
        response = model.generate_content([prompt, img])
        
        # Check if response has content
        if not response or not hasattr(response, 'text'):
            return {"error": f"Image {image_file.name} was blocked or returned no content."}

        raw_text = response.text
        clean_json = find_json_in_text(raw_text)
        data = json.loads(clean_json)
        
        return data if isinstance(data, list) else [data]

    except Exception as e:
        # Specific fix for 404: If 'flash' fails, the error message is caught here
        return {"error": f"Error on {image_file.name}: {str(e)}"}

def consolidate(all_results):
    """Merges all dictionaries into one unified logical table."""
    flat_data = []
    for res in all_results:
        if isinstance(res, list):
            flat_data.extend(res)
            
    if not flat_data:
        return pd.DataFrame()

    df = pd.DataFrame(flat_data)
    
    # Standardize headers
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
    
    # Logical Column Merging (unify similar field names)
    synonyms = {
        'total_amount': 'total', 'grand_total': 'total', 'price': 'total', 'amt': 'total',
        'date_of_purchase': 'date', 'transaction_date': 'date', 'dt': 'date',
        'vendor': 'merchant', 'store': 'merchant', 'company': 'merchant',
        'description': 'details', 'item_name': 'details', 'particulars': 'details'
    }
    df = df.rename(columns=synonyms)
    
    # Merge duplicate columns created by renaming
    df = df.groupby(lambda x: x, axis=1).first()
    return df

# --- UI ---
st.title("🚀 Robust Image-to-Table Extractor")

st.sidebar.header("System Settings")
# Updated model names to be more compatible with the current API
model_choice = st.sidebar.selectbox(
    "Select Model Version", 
    ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-pro-vision"]
)

st.sidebar.info(f"API Key: Active")
st.sidebar.write(f"Requests used: {st.session_state.quota_used}")

if st.sidebar.button("🗑 Reset App"):
    st.session_state.combined_df = None
    st.session_state.error_log = []
    st.session_state.quota_used = 0
    st.rerun()

uploaded_files = st.file_uploader("Upload Images (Max 100)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 100:
        st.error("Please limit to 100 images.")
    elif st.button(f"Process {len(uploaded_files)} Images"):
        all_data = []
        progress_bar = st.progress(0)
        status = st.empty()
        
        total = len(uploaded_files)
        # We process in small batches to stay within free tier limits
        BATCH_SIZE = 30 

        for i in range(0, total, BATCH_SIZE):
            batch = uploaded_files[i : i + BATCH_SIZE]
            
            for idx, img_file in enumerate(batch):
                current_num = i + idx + 1
                status.info(f"Extracting Image {current_num} of {total}...")
                
                result = extract_data(img_file, model_choice)
                
                if isinstance(result, dict) and "error" in result:
                    st.session_state.error_log.append(result["error"])
                else:
                    all_data.append(result)
                
                st.session_state.quota_used += 1
                
                # IMPORTANT: Per-image delay to prevent 429/Rate Limit crashes
                time.sleep(1.8) 
            
            # Update Progress Bar
            progress_bar.progress(min((i + BATCH_SIZE) / total, 1.0))
            
            # Batch Delay (4 seconds between every 30 images)
            if i + BATCH_SIZE < total:
                status.warning("Batch limit reached. Cooling down for 4 seconds...")
                time.sleep(4)

        status.success("Consolidating into unified table...")
        st.session_state.combined_df = consolidate(all_data)
        status.empty()

# --- Results ---
if st.session_state.combined_df is not None:
    if not st.session_state.combined_df.empty:
        st.subheader("📋 Consolidated Table")
        st.dataframe(st.session_state.combined_df, use_container_width=True)
        
        csv = st.session_state.combined_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Data (CSV)", data=csv, file_name="extracted_data.csv", mime="text/csv")
    else:
        st.error("No data could be extracted. Check error logs below.")

if st.session_state.error_log:
    with st.expander("⚠️ View Errors / Skipped Images"):
        for err in st.session_state.error_log:
            st.write(err)
