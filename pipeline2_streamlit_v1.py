import streamlit as st
import pandas as pd
import os
import re
import requests
from bs4 import BeautifulSoup
import json

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
LOCAL_DATA_FILE = "data/merged_data.csv"

# --- 1. HELPER FUNCTIONS ---

@st.cache_data
def load_data(file_path):
    """
    Loads the main dataset.
    MODIFIED: Assumes data is ALREADY in the correct format
    with columns: ['Product', 'Company', 'doc_text'].
    """
    try:
        merged_data = pd.read_csv(file_path)
        merged_data['doc_text'] = merged_data['doc_text'].fillna('')

        # --- FIX: Removed legacy code that referenced 'Product_normalized' and 'Name_x'.
        # The source CSV is now expected to have the correct columns directly,
        # as per your data prep script.

        # --- Validation: Ensure the new required columns exist ---
        required_cols = ['Product', 'Company', 'doc_text']
        if not all(col in merged_data.columns for col in required_cols):
             st.error(f"Data is missing one of the required columns: {required_cols}. Found: {merged_data.columns.tolist()}")
             return None, None

        # Filter DataFrame to only contain the three required columns
        merged_data = merged_data[required_cols]

        # 'Product' is the original product name
        unique_products = sorted(merged_data['Product'].unique().tolist())
        product_list = ["-- Select a Product --"] + unique_products

        return merged_data, product_list

    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None, None

# --- Extract URL Helper ---
def extract_website_url(doc_text: str) -> str:
    """Extracts the website URL using pure Python regex."""
    match = re.search(r"URLs_y:\s*website\s*-\s*([^;,\s]+)", doc_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

# --- Hyper-Aggressive, Unfiltered Scraping Implementation ---
def get_external_data_snippets(website_url: str) -> str:
    """
    Fetches the maximum amount of raw, visible text content from the given URL.
    Ensures the URL has a scheme (https://) if none is provided.
    """
    if not website_url:
        return f"*** EXTERNAL WEB DATA (FAILURE) ***\n(No website URL available for scraping.)"

    # Fix: Prepend scheme if missing
    if not re.match(r'http(s)?://', website_url, re.IGNORECASE):
        checked_url = 'https://' + website_url
    else:
        checked_url = website_url

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(checked_url, timeout=20, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove code/style elements
        for tag in soup(['script', 'style']):
            tag.decompose()

        # Extract ALL text from the body
        if soup.body:
            full_raw_text = soup.body.get_text(separator='\n', strip=True)
        else:
            full_raw_text = soup.get_text(separator='\n', strip=True)

        # Clean and format the final text block
        cleaned_text = re.sub(r'[\r\n]+', '\n', full_raw_text)
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)

        if len(cleaned_text.split()) < 10:
             return f"*** EXTERNAL WEB DATA (FAILURE) ***\nOnly minimal text found ({len(cleaned_text.split())} words). Site may be entirely dynamic."

        return (
            f"*** EXTERNAL WEB DATA (MAXIMUM RAW CONTENT FROM {checked_url}) ***\n"
            f"--- START OF RAW DATA ---\n"
            f"{cleaned_text}"
        )

    except requests.exceptions.RequestException as e:
        return f"*** EXTERNAL WEB DATA (FAILURE) ***\nRequest failed for {checked_url}. Reason: {e}"
    except Exception as e:
        return f"*** EXTERNAL WEB DATA (FAILURE) ***\nAn unexpected error occurred during parsing: {e}"


# --- Callback for API Key Persistence ---
def set_api_key():
    """Callback function to update the API key in session state."""
    if 'openai_api_key_input' in st.session_state:
        st.session_state.api_key_set = st.session_state.openai_api_key_input

def toggle_web_data():
    """Callback for the web data toggle to clear previous results if mode changes."""
    collected_data = st.session_state.get('collected_data')

    if collected_data is None:
        return

    current_mode_is_web = collected_data.get('mode') == 'Web Data ON'
    new_mode_is_web = st.session_state.web_data_toggle

    # If the stored data mode does NOT match the new toggle state, invalidate the data.
    if current_mode_is_web != new_mode_is_web:
         st.session_state.collected_data = None
         st.session_state.analysis_ready = False

def define_analyst_chain(api_key, prompt_template):
    """Defines the heavy LLM chain for detailed synergy analysis (LLM 3)."""

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)

    # --- Use the editable prompt template ---
    analyst_prompt = ChatPromptTemplate.from_template(prompt_template)
    return analyst_prompt | llm | StrOutputParser()

# --- MODIFIED: calculate_llm_cost now measures the full, combined input ---
def calculate_llm_cost(prompt_template_string, product_doc_1, product_doc_2):
    """
    Estimates the max token usage and cost for the gpt-4o API call.
    Substitutes the product data into the template before measuring length.
    """
    COST_INPUT_PER_M = 5.00
    COST_OUTPUT_PER_M = 15.00
    output_tokens = 1800 # Fixed estimate for the output length

    # 1. Substitute the large document strings into the template placeholders
    # Note: We must handle the possibility that the template still contains {data_source_label}
    # since it's not a standard variable passed to .invoke. We use a placeholder here.

    # Create a partial prompt with documents substituted
    # This is a slightly fragile string format operation, but necessary for accurate cost estimation.

    # We must use the .format() method to perform the substitution correctly
    try:
        # Step 1: Replace the product placeholders with the actual data
        temp_prompt = prompt_template_string.replace(
            '{product_1_doc}', product_doc_1
        ).replace(
            '{product_2_doc}', product_doc_2
        )
        # Step 2: Replace the dynamic {data_source_label} (LangChain ignores it, but we need it gone for accurate count)
        input_text = temp_prompt.replace(
            '{data_source_label}', 'Internal Data + External Web Data' # Use max length for safety
        )

    except Exception as e:
        # Fallback in case substitution fails (shouldn't happen with current design)
        st.warning(f"Cost calculation substitution failed: {e}. Using estimated default word count.")
        input_text = product_doc_1 + product_doc_2

    # 2. Measure the total length of the final prompt + data
    input_words = len(input_text.split())

    # 3. Calculate tokens and cost
    input_tokens = int(input_words * 1.33)

    input_cost = (input_tokens / 1_000_000) * COST_INPUT_PER_M
    output_cost = (output_tokens / 1_000_000) * COST_OUTPUT_PER_M

    total_cost = input_cost + output_cost

    return input_tokens, output_tokens, total_cost
# --- END MODIFIED calculate_llm_cost ---

# --- DEFAULT EDITABLE INSTRUCTIONS (Does NOT include placeholders) ---
DEFAULT_EDITABLE_INSTRUCTIONS = """
You are a senior M&A (Mergers and Acquisitions) analyst with decades of experience identifying intricate, high-value partnership opportunities. Your task is to conduct a deep-dive analysis of the synergy potential between 'Product 1' and 'Product 2'. Go beyond surface-level similarities and think like a creative business strategist to uncover the non-obvious ways these companies could create exponential value together.

First, read and internalize the data for both companies. Then, structure your analysis as an **Executive Partnership Briefing** with the following sections:

**1. Executive Summary:**
Start with a concise, top-level paragraph summarizing the core synergy thesis. What is the single most compelling reason this partnership could work?

**2. Key Synergy Opportunities (The 'Why'):**
In a detailed, bullet-pointed list, identify the most significant opportunities for synergy. For each point, explain the underlying \"why\". Consider questions like:
- **Value Chain Integration:** Can one company's product fill a critical gap in the other's customer value chain?
- **Data & Insights Synergy:** Could their combined data assets unlock new insights, products, or efficiencies?
- **Complementary Go-to-Market:** Beyond simple market access, do their sales motions or brand positions complement each other in a unique way?
- **Unlocking New Business Models:** Could a partnership enable a completely new product, service, or pricing model that neither could achieve alone?

**3. Potential Risks & Mitigations:**
In a detailed, bullet-pointed list, identify potential risks or challenges in this partnership (e.g.: conflicting company cultures, channel conflict, technological integration challenges). For each risk, suggest a potential mitigation strategy.

**4. Seven-Axis Synergy Scores:**
Provide a score from 1 (low synergy) to 10 (high synergy) for each of the following seven axes. For each score, provide a brief justification based on the provided data.
- **Clinical & Operational Alignment:** <Score and Justification> (Considers clinical pathways, TAT, and disease specificity match)
- **Patient Experience & Journey Continuity:** <Score and Justification> (Considers care continuity and patient drop-off risk)
- **Financial & Reimbursement Compatibility:** <Score and Justification> (Considers insurance coverage and out-of-pocket cost alignment)
- **Technological Interoperability:** <Score and Justification> (Considers APIs, EHR/EMR systems, and data standards)
- **Regulatory & Compliance Alignment:** <Score and Justification> (Considers overlap in regulatory pathways and certifications)
- **Geographic & Logistics Fit:** <Score and Justification> (Considers specimen logistics and physical location compatibility)
- **Strategic & Innovation Potential:** <Score and Justification> (Considers co-development opportunity and market expansion complementarity)

**5. Final Recommendation:**
Conclude with a single, direct recommendation: \\\"Highly Recommended\\\", \\\"Recommended with Reservations\\\", or \\\"Not Recommended\\\".
"""

# --- FIXED PLACEHOLDERS (Always appended to the end of the instructions) ---
# FIX: Using single braces {} for LangChain placeholders.
FIXED_PLACEHOLDERS = """
--- PRODUCT 1 ({data_source_label}) ---
{product_1_doc}

--- PRODUCT 2 ({data_source_label}) ---
{product_2_doc}
"""
# --- 2. STREAMLIT UI LAYOUT AND LOGIC ---

st.set_page_config(layout="wide")
st.title("🤝 Pipeline 2: Deep-Dive Synergy Analysis")

# --- Initial Data Load (Caches the data and gets product list) ---
if not os.path.exists(LOCAL_DATA_FILE):
    merged_data, product_options = None, []
    st.error(f"Data file not found at '{LOCAL_DATA_FILE}'. Please ensure the file is in the correct location.")
else:
    merged_data, product_options = load_data(LOCAL_DATA_FILE)

# --- Initialize Session State ---
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = ""
if 'openai_api_key_input' not in st.session_state:
    st.session_state.openai_api_key_input = ""
if 'collected_data' not in st.session_state:
    st.session_state.collected_data = None
if 'analysis_ready' not in st.session_state:
    st.session_state.analysis_ready = False
# Initialize the toggle state
if 'web_data_toggle' not in st.session_state:
    st.session_state.web_data_toggle = False
if 'editable_prompt' not in st.session_state: # Prompt storage
    st.session_state.editable_prompt = DEFAULT_EDITABLE_INSTRUCTIONS



# Sidebar for Configuration Inputs
with st.sidebar:
    st.header("Configuration")

    # --- API Key Input ---
    st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key_set,
        key='openai_api_key_input',
        on_change=set_api_key,
        help="The key is applied when you press Enter or click away."
    )

    # Active key is read from the source of truth
    api_key = st.session_state.api_key_set

    # --- RESET BUTTONS ---

    # 1. Reset API Key Button
    if st.button("Reset API Key", type="secondary"):
        st.session_state.api_key_set = ""

        # Delete the widget's internal key state to clear the text input
        if 'openai_api_key_input' in st.session_state:
            del st.session_state['openai_api_key_input']

        st.rerun()

    # 2. New Analysis / Reset Inputs Button
    if st.button("New Analysis / Reset Inputs", type="secondary"):
        # This button clears everything EXCEPT the API key (st.session_state.api_key_set)
        st.session_state.collected_data = None
        st.session_state.analysis_ready = False

        # Fix: Explicitly delete the selectbox keys to reset the dropdowns to index 0
        if 'product_1_select' in st.session_state:
            del st.session_state['product_1_select']
        if 'product_2_select' in st.session_state:
            del st.session_state['product_2_select']

        # Reset the editable prompt to its default version
        st.session_state.editable_prompt = DEFAULT_EDITABLE_INSTRUCTIONS

        st.rerun()

    st.header("Product Inputs")

    # Toggle for scraping
    st.checkbox(
        "Include External Web Scraping",
        value=st.session_state.web_data_toggle,
        help="If checked, the pipeline will scrape data from product websites to augment the internal data for deeper analysis (increases LLM cost).",
        key='web_data_toggle',
        on_change=toggle_web_data # Calls function to check if data needs to be cleared
    )

    # Selectboxes with keys
    product_1_input = st.selectbox("Product 1 Name", product_options, key='product_1_select', index=0)
    product_2_input = st.selectbox("Product 2 Name", product_options, key='product_2_select', index=0)

    data_source_label = "Internal Data + Web" if st.session_state.web_data_toggle else "Internal Data Only"
    # NOTE: The "Prepare Data" button will grab the latest prompt text upon press.
    run_collection_button = st.button(f"Prepare Data ({data_source_label})", type="primary")

# Main Content Area

# --- Editable Prompt Section ---
with st.expander("🛠️ View/Edit LLM Prompt Template", expanded=False):
    # The editable area now only contains the instructions, NOT the placeholder section.
    st.session_state.editable_prompt = st.text_area(
        "Edit Analyst Chain Prompt Template (Instructions Only)",
        value=st.session_state.editable_prompt,
        height=600,
        key='prompt_text_area',
        help="Customize the LLM instructions, scoring axes, and output format. The data placeholders will be added automatically."
    )

    # --- NEW: Apply and Recalculate Prompt Cost Button ---
    if st.button("Apply & Recalculate Prompt Cost", type="secondary"):
        if st.session_state.collected_data is None:
            st.warning("Please press the 'Prepare Data' button first to collect data and establish the base prompt context.")
        elif not api_key:
             st.error("Please enter your OpenAI API Key first.")
        else:
            # 1. Get current data for cost estimation
            doc1 = st.session_state.collected_data['product_1_doc']
            doc2 = st.session_state.collected_data['product_2_doc']

            # 2. Get the current toggle state and labels
            use_web_data = st.session_state.web_data_toggle
            data_source_label = "Internal Data + External Web Data" if use_web_data else "Internal Data ONLY"
            data_source_prompt_text = "internal data AND external web data" if use_web_data else "internal data"

            # 3. Re-assemble the final prompt with the current edited instructions
            current_instructions = st.session_state.editable_prompt

            final_prompt_template = current_instructions.replace(
                "read and internalize the data for both companies",
                f"read and internalize the **{data_source_prompt_text}** for both companies"
            ).replace(
                "provided data.",
                f"provided **{data_source_prompt_text}**."
            )
            final_prompt_template += FIXED_PLACEHOLDERS.replace("{data_source_label}", data_source_label)

            # 4. Recalculate cost using the FULL final_prompt_template string and documents
            input_tokens, output_tokens, total_cost = calculate_llm_cost(
                final_prompt_template, doc1, doc2
            )

            # 5. Update session state with new cost and prompt
            st.session_state.collected_data['cost'] = {'input': input_tokens, 'output': output_tokens, 'total': total_cost}
            st.session_state.collected_data['final_prompt_template'] = final_prompt_template

            st.success("Prompt applied and cost recalculated!")
            st.rerun() # Rerun to refresh the cost display
# --- END Editable Prompt Section ---

if 'collected_data' not in st.session_state:
    st.session_state.collected_data = None

# --- Data Collection Logic (Triggered by button) ---
if run_collection_button:
    # Clear previous analysis state
    st.session_state.analysis_ready = False
    st.session_state.collected_data = None # Ensure it's cleared before collection starts

    use_web_data = st.session_state.web_data_toggle # Get current toggle state for this run

    if product_1_input == "-- Select a Product --" or product_2_input == "-- Select a Product --":
        st.error("Please select a valid product for both Product 1 and Product 2.")
    elif not api_key:
        st.error("Please enter your OpenAI API Key.")
    elif merged_data is None:
        st.error("Cannot run analysis because the merged data failed to load.")
    else:

        # --- CONDITIONAL DATA PREPARATION ---
        with st.spinner(f'Preparing data... (Web scraping {"ON" if use_web_data else "OFF"})...'):
            try:
                # Get data for Product 1 and Product 2
                # FIX: Use 'Product' column as the locator
                product_1_row = merged_data.loc[merged_data['Product'] == product_1_input].iloc[0]
                product_2_row = merged_data.loc[merged_data['Product'] == product_2_input].iloc[0]

                product_1_internal = product_1_row['doc_text']
                product_2_internal = product_2_row['doc_text']

                product_1_url = ""
                product_2_url = ""

                # Labels used for prompt construction and display
                data_source_label = "Internal Data ONLY"
                data_source_prompt_text = "internal data"

                if use_web_data:
                    # Web Scraping is ON
                    product_1_url = extract_website_url(product_1_internal)
                    product_2_url = extract_website_url(product_2_internal)

                    product_1_external = get_external_data_snippets(product_1_url)
                    product_2_external = get_external_data_snippets(product_2_url)

                    data_source_label = "Internal Data + External Web Data"
                    data_source_prompt_text = "internal data AND external web data"

                    # Prepare Combined Document (Internal + External Data)
                    # FIX: Use 'Product' column
                    PRODUCT_1_DOC = (
                        f"--- PRODUCT 1: {product_1_row['Product']} by {product_1_row['Company']} ---\n"
                        f"*** Internal Data ***\n{product_1_internal}\n\n"
                        f"*** External Web Data ***\n{product_1_external}\n"
                    )
                    # FIX: Use 'Product' column
                    PRODUCT_2_DOC = (
                        f"--- PRODUCT 2: {product_2_row['Product']} by {product_2_row['Company']} ---\n"
                        f"*** Internal Data ***\n{product_2_internal}\n\n"
                        f"*** External Web Data ***\n{product_2_external}\n"
                    )

                else:
                    # Web Scraping is OFF (Internal Data ONLY)
                    # FIX: Use 'Product' column
                    PRODUCT_1_DOC = f"--- PRODUCT 1: {product_1_row['Product']} by {product_1_row['Company']} ---\n{product_1_internal}\n"
                    # FIX: Use 'Product' column
                    PRODUCT_2_DOC = f"--- PRODUCT 2: {product_2_row['Product']} by {product_2_row['Company']} ---\n{product_2_internal}\n"

                # --- FINAL PROMPT CONSTRUCTION (INITIAL RUN) ---

                current_instructions = st.session_state.editable_prompt

                # 1. Dynamically replace the generic data description placeholders in the instructions
                final_prompt_template = current_instructions.replace(
                    "read and internalize the data for both companies",
                    f"read and internalize the **{data_source_prompt_text}** for both companies"
                ).replace(
                    "provided data.",
                    f"provided **{data_source_prompt_text}**."
                )

                # 2. Append the fixed placeholders for variable injection
                final_prompt_template += FIXED_PLACEHOLDERS.replace("{data_source_label}", data_source_label)


                # Estimate cost using the FULL final_prompt_template string
                input_tokens, output_tokens, total_cost = calculate_llm_cost(
                    final_prompt_template, PRODUCT_1_DOC, PRODUCT_2_DOC
                )

                # Store in session state for the next step
                st.session_state.collected_data = {
                    'product_1_doc': PRODUCT_1_DOC,
                    'product_2_doc': PRODUCT_2_DOC,
                    'api_key': api_key,
                    'cost': {'input': input_tokens, 'output': output_tokens, 'total': total_cost},
                    'product_1_url': product_1_url,
                    'product_2_url': product_2_url,
                    'mode': 'Web Data ON' if use_web_data else 'Internal Data Only',
                    'final_prompt_template': final_prompt_template # Store the final prompt template used
                }
                st.session_state.analysis_ready = True
                st.success(f"Data prepared ({st.session_state.collected_data['mode']}). Ready for analysis.")

            except IndexError:
                st.error(f"❌ Error: Product data could not be found internally.")
                st.session_state.analysis_ready = False
            except Exception as e:
                st.error(f"An unexpected error occurred during data processing: {e}")
                st.session_state.analysis_ready = False

# --- Manual Review and LLM Analysis Section ---

if st.session_state.analysis_ready and st.session_state.collected_data:
    data = st.session_state.collected_data
    cost_data = data['cost']
    mode = data['mode']

    # Get the current run mode to correctly parse the document structure
    was_web_data_used = mode == 'Web Data ON'

    st.header(f"🕵️ Manual Data Review and Cost Confirmation ({mode})")
    st.markdown("---")

    if was_web_data_used:
        st.warning("⚠️ **Mode**: Web scraping was ON. Note that external data increases input tokens and LLM cost.")

    col1, col2 = st.columns(2)

    # --- Parsing logic adjusted for mode ---

    def get_data_for_display(doc_content, is_web_mode):
        if is_web_mode:
            # Data is in combined format: Internal Data... \n\n*** External Web Data...
            internal = doc_content.split("*** Internal Data ***\n")[1].split("\n\n*** External Web Data ***")[0].strip()
            # External is everything after the marker (ignoring the raw data start marker if present)
            external = doc_content.split("*** External Web Data ***\n")[1].strip()
            return internal, external
        else:
            # Data is in internal-only format: --- PRODUCT 1... \n{content}\n
            # We use rpartition to get the content after the product header
            internal = doc_content.rpartition("---")[2].strip()
            return internal, None

    internal_data_1, external_data_1 = get_data_for_display(data['product_1_doc'], was_web_data_used)
    internal_data_2, external_data_2 = get_data_for_display(data['product_2_doc'], was_web_data_used)

    # --- Product 1 Data Review ---
    with col1:
        st.subheader(f"Product 1 ({product_1_input}) Data")
        st.text_area("Product 1 Internal Data", internal_data_1, height=150)

        # Display External Data only if it was used
        if was_web_data_used:
            st.caption(f"Source URL: {data.get('product_1_url') or 'N/A'}")
            st.text_area("Product 1 External Data", external_data_1, height=150)

    # --- Product 2 Data Review ---
    with col2:
        st.subheader(f"Product 2 ({product_2_input}) Data")
        st.text_area("Product 2 Internal Data", internal_data_2, height=150)

        # Display External Data only if it was used
        if was_web_data_used:
            st.caption(f"Source URL: {data.get('product_2_url') or 'N/A'}")
            st.text_area("Product 2 External Data", external_data_2, height=150)


    st.markdown("---")
    # --- COST AND CONFIRMATION SECTION ---
    st.subheader("💰 LLM Cost Estimate (GPT-4o)")
    st.warning(f"Estimated Total Cost for ONE analysis run: **${cost_data['total']:.4f} USD**")

    cost_col1, cost_col2 = st.columns(2)
    cost_col1.metric("Input Tokens (Est.)", f"{cost_data['input']:,}")
    cost_col2.metric("Output Tokens (Est.)", f"{cost_data['output']:,}")

    # Final Confirmation Button
    if st.button("Confirm & Run Final LLM Analysis", type="secondary"):

        # --- LLM Analysis Execution ---
        with st.spinner("🧠 Running Deep-Dive Analyst Chain (LLM 3)..."):
            try:
                # Pass the dynamically generated prompt template
                analyst_chain = define_analyst_chain(data['api_key'], data['final_prompt_template'])

                analysis_report = analyst_chain.invoke({
                    "product_1_doc": data['product_1_doc'],
                    "product_2_doc": data['product_2_doc']
                })

                st.balloons()
                st.success("Analysis Complete!")

                st.subheader("📈 Executive Partnership Briefing")
                st.markdown("---")

                # --- WRAPPING OUTPUT FIX ---
                st.markdown(
                    """
                    <style>
                    /* Target the custom-styled div */
                    .output-box-readable {
                        white-space: pre-wrap !important; /* Forces content to wrap */
                        word-wrap: break-word !important; /* Ensures long words break */
                        overflow-x: hidden !important;    /* Hides horizontal scrollbar */

                        /* Styling for readability */
                        font-family: monospace;
                        background-color: #262730;
                        color: #FAFAFA;
                        border-radius: 0.3rem;
                        padding: 1rem;
                        border: 1px solid #333;
                        max-width: 100%;
                    }
                    /* Ensure headers/bold text remain visible against the dark background */
                    .output-box-readable strong {
                        color: #FCF4A3;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Use st.markdown to inject the custom HTML structure
                output_content = analysis_report.replace('\n', '<br>')
                st.markdown(
                    f'<div class="output-box-readable">{output_content}</div>',
                    unsafe_allow_html=True
                )
                # --- END WRAPPING OUTPUT FIX ---

                st.session_state.analysis_ready = False

            except Exception as e:
                st.error(f"❌ An error occurred during the LLM analysis: {e}. Check your API key and network connection.")
