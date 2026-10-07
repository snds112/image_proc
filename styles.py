CSS_STYLES = """
<style>
    .stApp {
        background-color: #0e1117;
    }
    .image-container {
        border: 2px dashed #4a4a4a;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        min-height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #1a1a1a;
    }
    .effect-badge {
        display: inline-block;
        padding: 4px 12px;
        margin: 2px;
        border-radius: 15px;
        background-color: #ff4b4b;
        color: white;
        font-size: 0.85em;
    }
    .metric-card {
        background-color: #1a1a2e;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 10px;
        border-left: 3px solid #ff4b4b;
        text-align: center;
    }
    .metric-label {
        font-size: 0.75em;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6em;
        font-weight: bold;
        color: #fff;
    }
    
    /* ===== WIDER SIDEBAR (only when expanded) ===== */
[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 380px !important;
    max-width: 450px !important;
}
    
    /* Smaller expander labels */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        font-size: 0.9em !important;
        padding: 0.3rem 0.5rem !important;
    }
    
    /* Smaller button text */
    [data-testid="stSidebar"] .stButton button {
        font-size: 0.8em !important;
        padding: 0.25rem 0.5rem !important;
    }
    
    /* ===== FIX: Left-align text in sidebar buttons ===== */
    /* Streamlit buttons use flexbox - override justify-content */
    [data-testid="stSidebar"] .stButton > button {
        justify-content: flex-start !important;
        text-align: left !important;
    }
    [data-testid="stSidebar"] .stButton > button > div {
        justify-content: flex-start !important;
        text-align: left !important;
    }
    [data-testid="stSidebar"] .stButton > button p {
        text-align: left !important;
    }
    
    /* Smaller slider labels */
    [data-testid="stSidebar"] .stSlider label {
        font-size: 0.8em !important;
    }
    
    /* Smaller selectbox labels */
    [data-testid="stSidebar"] .stSelectbox label {
        font-size: 0.8em !important;
    }
    
    /* Smaller markdown text in sidebar */
    [data-testid="stSidebar"] p {
        font-size: 0.85em !important;
        line-height: 1.3 !important;
    }
    
    /* Smaller code blocks in sidebar */
    [data-testid="stSidebar"] code {
        font-size: 0.75em !important;
    }
    
    /* Compact column spacing in pipeline list */
    [data-testid="stSidebar"] [data-testid="column"] {
        padding: 0 2px !important;
    }
</style>
"""