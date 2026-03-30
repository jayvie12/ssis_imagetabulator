import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
import json
import re
from PIL import Image
import io

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Data Smart-Tabulator", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    .status-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: CONFIG & QUOTA ---
with st.sidebar:
    st.title("🛠️ Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password", help="Get a free key at aistudio.google.com")
    
    st.divider()
    st.subheader("📊 Quota Monitor")
    if 'quota_count' not in st.session_state:
        st.session_state.quota_count = 0
    
    progress = min(st.session_state.quota_count / 1500, 1.0)
    st.progress(progress)
    st.caption(f"Used today: {st.session_state.quota_count} / 1500 requests (Est. Free Limit)")
    
    if st.button("🗑️ Clear All Data"):
        st.session_state.final_df = None
        st.session_state.quota_count = 0
        st.rerun()

# --- DATA NORMALIZATION LOGIC ---
def normalize_dataframe(df):
    """Smartly merges columns that represent the same data."""
    if df.empty: return df
    
    # 1. Clean headers: lowercase and underscores
    df.columns = [str(c).lower().strip().replace(" ", "_").replace("/", "_") for c in df.columns]
    
    # 2. Define Synonyms Map
    synonym_map = {
        'total': ['amount', 'price', 'grand_total', 'total_amount', 'amt', 'value', 'net_amount'],
        'date': ['transaction_date', 'dated', 'purchase_date', 'disposal_date', 'dt'],
        'merchant': ['vendor', 'store', 'company', 'seller', 'branch_name', 'unit_name'],
        'reference': ['invoice_no', 'ref_no', 'receipt_id', 'box_ref', 'id_number']
    }
    
    # 3. Rename columns based on map
    rename_dict = {}
    for standard, variations in synonym_map.items():
        for col in df.columns:
            if col in variations or standard in col:
                rename_dict[col] = standard
    
    df = df.rename(columns=rename_dict)
    
    # 4. Merge duplicate columns (keep first non-null)
    df = df.groupby(level=0, axis=1).first()
    
    return df

# --- AI PROCESSING ---
def extract_data(image_file):
    if not api_key:
        st.error("Please provide an API Key!")
        return None
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        img = Image.open(image_file)
        prompt = """
        Extract all data from this image. 
        Format as a JSON object of key-value pairs. 
        If multiple items exist, return a list of objects.
        Return ONLY valid JSON. No conversational text.
        """
        
        response = model.generate_content([prompt, img])
        
        # Extract JSON using Regex to be safe
        raw_text = response.text
        json_match = re.search(r'(\[.*\]|\{.*\})', raw_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            return data if isinstance(data, list) else [data]
        return None
    except Exception as e:
        st.error(f"Error processing {image_file.name}: {e}")
        return None

# --- MAIN INTERFACE ---
st.title("📸 AI Smart-Tabulator")
st.info("Upload up to 100 images. The AI will extract data, merge similar fields, and create a unified table.")

uploaded_files = st.file_uploader("Choose Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if 'final_df' not in st.session_state:
    st.session_state.final_df = None

if uploaded_files:
    num_files = len(uploaded_files)
    
    if num_files > 100:
        st.warning("⚠️ Limit reached: Only the first 100 images will be processed.")
        uploaded_files = uploaded_files[:100]

    if st.button(f"🚀 Process {len(uploaded_files)} Images"):
        all_results = []
        progress_bar = st.progress(0)
        status = st.empty()
        
        for i, img_file in enumerate(uploaded_files):
            # Batch Limiter Check
            if i > 0 and i % 30 == 0:
                status.warning(f"⏸️ Batch limit (30) reached. Cooling down 5s to stay in Free Tier...")
                time.sleep(5)
            
            status.info(f"🔍 Extracting data from Image {i+1} of {len(uploaded_files)}...")
            
            result = extract_data(img_file)
            if result:
                all_results.extend(result)
                st.session_state.quota_count += 1
            
            # Rate Limiter: 1.5s pause between images to stay under 15 RPM
            time.sleep(1.5)
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        if all_results:
            raw_df = pd.DataFrame(all_results)
            st.session_state.final_df = normalize_dataframe(raw_df)
            status.success("✅ Tabulation Complete!")
        else:
            status.error("❌ No data could be extracted.")

# --- RESULTS DISPLAY & EXPORT ---
if st.session_state.final_df is not None:
    st.divider()
    st.subheader("📋 Unified Results Table")
    st.dataframe(st.session_state.final_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = st.session_state.final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="tabulated_data.csv",
            mime="text/csv",
        )
    
    with col2:
        # Simple Copy logic for Streamlit
        text_val = st.session_state.final_df.to_csv(sep='\t', index=False)
        st.text_area("Copy Table Data (Tab Separated for Excel):", value=text_val, height=100)
        st.caption("Select all (Ctrl+A) and Copy (Ctrl+C) to paste into Excel.")
