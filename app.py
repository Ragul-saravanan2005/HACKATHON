import pandas as pd
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
import streamlit as st
import re, os

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Hackathon Unified App", page_icon="🚀", layout="wide")

# ----------------------------
# Load datasets
# ----------------------------
@st.cache_data
def load_nco_data():
    return pd.read_csv(r"C:\Users\ASUS\Downloads\penta\MOCK_DATA_with_NCO.csv")

@st.cache_data
def load_survey_data():
    ctl_file_path = r"C:\Users\ASUS\Downloads\penta\survey_data.ctl"
    
    with open(ctl_file_path, "r", encoding="utf-8") as f:
        ctl_content = f.read()
    
    # Extract INFILE path
    infile_match = re.search(r"INFILE\s+'([^']+)'", ctl_content, re.IGNORECASE)
    if not infile_match:
        st.error("❌ No INFILE path found inside the .ctl file")
        return pd.DataFrame()
    
    raw_path = infile_match.group(1).strip()
    
    # If the path is relative, join with ctl file directory
    if not os.path.isabs(raw_path):
        data_file_path = os.path.join(os.path.dirname(ctl_file_path), raw_path)
    else:
        data_file_path = raw_path
    
    # If the file doesn't exist, try looking in ctl folder
    if not os.path.exists(data_file_path):
        possible_path = os.path.join(os.path.dirname(ctl_file_path), os.path.basename(data_file_path))
        if os.path.exists(possible_path):
            data_file_path = possible_path
        else:
            st.error(f"❌ Data file not found: {data_file_path}")
            return pd.DataFrame()
    
    # Extract delimiter
    delimiter_match = re.search(r"FIELDS TERMINATED BY\s+'([^']+)'", ctl_content, re.IGNORECASE)
    delimiter = delimiter_match.group(1) if delimiter_match else ","
    
    # Try reading file
    try:
        df = pd.read_csv(data_file_path, delimiter=delimiter, encoding="utf-8", quotechar='"')
        st.success(f"✅ Survey data loaded from: {data_file_path}")
        return df
    except UnicodeDecodeError:
        df = pd.read_csv(data_file_path, delimiter=delimiter, encoding="latin-1", quotechar='"')
        st.success(f"✅ Survey data loaded from: {data_file_path} (latin-1 encoding)")
        return df
    except Exception as e:
        st.error(f"❌ Error loading survey data: {e}")
        return pd.DataFrame()

# ----------------------------
# Load NCO Model + Embeddings
# ----------------------------
@st.cache_resource
def load_model_and_embeddings(df):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    df = df.copy()
    df['embeddings'] = df['occupation_title'].apply(
        lambda x: model.encode(str(x), convert_to_tensor=True)
    )
    return model, df

# ----------------------------
# NCO Search Function
# ----------------------------
def search_occupation(query, model, df, top_k=3):
    query_emb = model.encode(query, convert_to_tensor=True)
    temp_df = df.copy()
    temp_df['semantic_score'] = temp_df['embeddings'].apply(
        lambda emb: float(util.cos_sim(query_emb, emb))
    )
    temp_df['fuzzy_score'] = temp_df['occupation_title'].apply(
        lambda title: fuzz.token_sort_ratio(query.lower(), title.lower()) / 100
    )
    temp_df['final_score'] = 0.7 * temp_df['semantic_score'] + 0.3 * temp_df['fuzzy_score']
    results = temp_df.sort_values(by='final_score', ascending=False).head(top_k)
    return results[['occupation_title', 'nco_code', 'final_score']]

# ----------------------------
# Streamlit Tabs
# ----------------------------
st.title("SurveyX")

tab1, tab2 = st.tabs(["📊 DATA DISSEMINATION - Survey Data", "🔍 DATA COLLECTION AND PROCESSING - NCO Search"])

# ----------------------------
# TAB 1: Problem Statement 2
# ----------------------------
with tab1:
    st.header("📊 Survey Data Explorer")

    df_survey = load_survey_data()
    if not df_survey.empty:
        st.write(f"Total records: {len(df_survey)}")
        st.dataframe(df_survey.head(10), use_container_width=True)

        # Search by column values
        st.subheader("🔍 Search in Survey Data")
        column = st.selectbox("Select column", df_survey.columns)
        value = st.text_input("Enter search value")
        if st.button("Search Survey Data"):
            if value.strip() != "":
                results = df_survey[df_survey[column].astype(str).str.contains(value, case=False, na=False)]
                st.write(f"Found {len(results)} records")
                st.dataframe(results, use_container_width=True)
            else:
                st.warning("Please enter a search value")

        # CSV download
        st.subheader("⬇ Download Full Data")
        csv = df_survey.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "survey_data.csv", "text/csv")
    else:
        st.warning("⚠ Survey data not loaded.")

# ----------------------------
# TAB 2: Problem Statement 5
# ----------------------------
with tab2:
    st.header("🔍 Multilingual NCO Occupation Search")

    df_nco = load_nco_data()
    model, df_nco = load_model_and_embeddings(df_nco)

    query = st.text_input("Enter job title", placeholder="Type here... e.g. மென்பொருள் பொறியாளர் / Software Engineer")
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)

    if st.button("Search NCO"):
        if query.strip() == "":
            st.warning("⚠ Please enter a job title to search.")
        else:
            results = search_occupation(query, model, df_nco, top_k)
            st.subheader("Results:")
            st.dataframe(results, use_container_width=True)
