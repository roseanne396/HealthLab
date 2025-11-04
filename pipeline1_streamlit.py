import streamlit as st
import pandas as pd
import json
import os
import time
import re
from collections import Counter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda

# --- CONFIGURATION ---
LOCAL_DATA_FILE = "data/merged_data.csv"
CHROMA_DB_PATH = "chroma_db"

# --- 1. CACHED & HELPER FUNCTIONS ---

@st.cache_data
def load_data(file_path):
    """
    Loads the main dataset. Assumes the CSV now has 'Product', 'Company', 'doc_text'.
    """
    try:
        merged_data = pd.read_csv(file_path)
        merged_data['doc_text'] = merged_data['doc_text'].fillna('')

        # Check if the new schema is correct
        required_cols = ['Product', 'Company', 'doc_text']
        if not all(col in merged_data.columns for col in required_cols):
             st.error(f"Data is missing required columns: {required_cols}. Please ensure 'merged_data.csv' has been cleaned and renamed.")
             return None, None

        # Clean doc_text to remove non-ASCII characters upon load
        merged_data['doc_text'] = merged_data['doc_text'].astype(str).apply(
            lambda x: x.encode('ascii', 'ignore').decode('ascii')
        )

        unique_product_names = sorted(merged_data['Product'].unique().tolist())
        product_options = ["-- Select a Target Product --"] + unique_product_names

        return merged_data, product_options

    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None, None

@st.cache_resource
def load_vectorstore(api_key):
    """
    Loads the Chroma vector store from disk.
    """
    st.info("Loading Chroma vector store from disk... (This runs only once per session)")
    try:
        if not os.path.exists(CHROMA_DB_PATH):
            st.error(f"Vector store directory not found at '{CHROMA_DB_PATH}'. Please ensure you have rebuilt it with the new schema.")
            return None

        if not api_key:
             st.error("Cannot load vector store: OpenAI API Key is missing.")
             return None

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        st.success(f"Vector store loaded successfully from '{CHROMA_DB_PATH}'.")
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

def calculate_llm_cost(input_tokens_per_call, output_tokens_per_call, num_calls):
    """Estimates the max token usage and cost for gpt-4o API calls."""
    COST_INPUT_PER_M = 5.00
    COST_OUTPUT_PER_M = 15.00

    total_input_tokens = input_tokens_per_call * num_calls
    total_output_tokens = output_tokens_per_call * num_calls

    input_cost = (total_input_tokens / 1_000_000) * COST_INPUT_PER_M
    output_cost = (total_output_tokens / 1_000_000) * COST_OUTPUT_PER_M

    total_cost = input_cost + output_cost
    return total_input_tokens, total_output_tokens, total_cost


def define_llm_chains(api_key, num_angles):
    """Defines all three LLM chains."""
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)

    # --- LLM 1: The Dynamic Strategist ---
    strategist_prompt_template = f"""
    You are a top-tier product strategist. Your goal is to identify broad categories of highly synergistic **PRODUCT TYPES** (service, technology, or data products) for the 'Target Product' below.
    Analyze the 'Target Product' and its features, then devise **{num_angles}** most potent partnership pillars. Each pillar MUST describe a **PRODUCT TYPE**.
    For each pillar, write a paragraph focusing on the ideal synergistic **PRODUCT's features, function, and data**, and how they would enhance the Target Product.
    --- TARGET PRODUCT ---
    {{target_doc}}
    --- OUTPUT FORMAT ---
    Provide your response as a single, valid JSON object with exactly **{num_angles}** key-value pairs.
    """
    strategist_prompt = ChatPromptTemplate.from_template(strategist_prompt_template)
    strategist_chain = strategist_prompt | llm | JsonOutputParser()

    # --- LLM 2: The Profiler ---
    profiler_prompt_template = """
    You are a brilliant business strategist. Based on the following 'Partnership Strategy', generate a rich, abstract profile of an ideal partner product.
    Do not invent a name for a product. Describe its features, target users, and core value proposition in detail.
    --- PARTNERSHIP STRATEGY ---
    {strategy_description}
    --- TARGET PRODUCT ---
    {target_doc}
    """
    profiler_prompt = ChatPromptTemplate.from_template(profiler_prompt_template)
    profiler_chain = profiler_prompt | llm | StrOutputParser()

    # --- LLM 3: The Initial Scorer ---
    scorer_prompt_template = """
    You are a rapid-assessment business analyst. Assess the synergy potential between the 'Target Product' and the 'Potential Partner' based ONLY on the provided texts.
    Provide a single score from 1 (low synergy) to 10 (high synergy) and a one-sentence justification.
    --- TARGET PRODUCT ---
    {target_doc}
    --- POTENTIAL PARTNER CHUNKS ---
    {candidate_chunks_text}
    --- OUTPUT FORMAT ---
    Provide your response as a JSON object with two keys: "score" and "reasoning".
    """
    scorer_prompt = ChatPromptTemplate.from_template(scorer_prompt_template)
    scorer_chain = scorer_prompt | llm | JsonOutputParser()

    return strategist_chain, profiler_chain, scorer_chain

# --- 2. PIPELINE FUNCTIONS ---

def estimate_pipeline_cost(api_key, merged_data, target_product_input, num_angles):
    """
    Calculates the estimated cost for the entire pipeline run, now fully dynamic.
    """
    target_product_row = merged_data.loc[merged_data['Product'] == target_product_input].iloc[0]
    target_doc_length = len(target_product_row['doc_text'].split())
    target_doc_tokens = int(target_doc_length * 1.33) # Convert word count to estimated tokens

    # 1. Estimate Stage 1 (LLM 1 - Strategist)
    llm1_input_tokens = 500 + target_doc_tokens
    llm1_total_input, llm1_total_output, llm1_cost = calculate_llm_cost(llm1_input_tokens, 300, 1)

    # 2. Estimate Stage 2 (LLM 2 - Profiler)
    llm2_input_tokens = 800 + target_doc_tokens
    llm2_total_input, llm2_total_output, llm2_cost = calculate_llm_cost(llm2_input_tokens, 300, num_angles)

    # 3. Estimate Stage 4 (LLM 3 - Scorer)
    N_c_estimate = max(5, int(num_angles * 2.5))
    # FIX: Make the LLM3 estimate dynamic based on target doc length + a fixed chunk estimate
    llm3_input_tokens = 3500 + target_doc_tokens # (Base estimate for prompt + chunks) + target doc
    llm3_total_input, llm3_total_output, llm3_cost = calculate_llm_cost(llm3_input_tokens, 200, N_c_estimate)

    # Total Estimate
    total_tokens = llm1_total_input + llm1_total_output + llm2_total_input + llm2_total_output + llm3_total_input + llm3_total_output
    total_cost = llm1_cost + llm2_cost + llm3_cost

    return total_tokens, total_cost, N_c_estimate

@st.cache_data(show_spinner=False)
def run_pipeline_execution(_api_key, merged_data, target_product_input, num_angles, _vectorstore):
    """
    Executes the core RAG pipeline with the new, clean schema.
    """
    target_product_row = merged_data.loc[merged_data['Product'] == target_product_input].iloc[0]
    target_product_description = target_product_row['doc_text']

    strategist_chain, profiler_chain, scorer_chain = define_llm_chains(_api_key, num_angles)

    st.subheader(f"1️⃣ Dynamic Strategy Generation (LLM 1) - {num_angles} Pillars")
    with st.spinner("🧠 Analyzing product and defining strategic pillars..."):
        clean_target_doc = target_product_description.encode("ascii", "ignore").decode("ascii")
        synergy_strategies = strategist_chain.invoke({"target_doc": clean_target_doc})
        st.success("Strategic Pillars Generated.")
        st.json(synergy_strategies)

    st.markdown("---")

    # --- STAGE 2 & 3: Profile, Retrieve, and Aggregate Candidates ---
    st.subheader("2️⃣ Candidate Profiling and RAG Retrieval (LLM 2 + Chroma)")
    all_retrieved_chunks = []
    candidate_product_summary = {}

    def custom_retriever(query_text: str):
        return _vectorstore.similarity_search(query_text, k=8)

    progress_bar = st.progress(0, text="Starting profiling and retrieval...")

    for i, (strategy_type, strategy_desc) in enumerate(synergy_strategies.items()):
        progress_bar.progress((i + 1) / (num_angles + 1), text=f"Profiling & Retrieving for: '{strategy_type}'...")

        hypothetical_doc = profiler_chain.invoke({
            "strategy_description": strategy_desc,
            "target_doc": clean_target_doc
        })

        retrieved_docs = custom_retriever(hypothetical_doc)

        # --- FIX: Self-filtering now uses the clean 'Product' key ---
        filtered_docs = [doc for doc in retrieved_docs if doc.metadata.get('Product') != target_product_input]

        all_retrieved_chunks.extend(filtered_docs)

        for doc in filtered_docs:
            company_name = doc.metadata.get('Company')
            # --- FIX: Product name retrieval now uses the clean 'Product' key ---
            product_name = doc.metadata.get('Product')

            if isinstance(company_name, str) and isinstance(product_name, str):
                product_key = (company_name, product_name)
            else:
                continue

            candidate_product_summary.setdefault(product_key, { 'total_chunks': 0, 'angles': Counter() })
            candidate_product_summary[product_key]['total_chunks'] += 1
            candidate_product_summary[product_key]['angles'][strategy_type] += 1

    progress_bar.progress(1.0, text="Completed profiling and retrieval.")
    time.sleep(1); progress_bar.empty()

    sorted_candidates = sorted(candidate_product_summary.items(), key=lambda item: item[1]['total_chunks'], reverse=True)
    unique_candidates_keys = [key for key, value in sorted_candidates]

    st.success(f"Found a total of **{len(sorted_candidates)}** unique potential partners from **{len(all_retrieved_chunks)}** retrieved chunks.")

    intermediate_candidate_data = []
    for company_name, product_name in unique_candidates_keys:
        stats = candidate_product_summary[(company_name, product_name)]
        angle_breakdown = ", ".join([f"{angle.replace('_', ' ').title()} ({count})" for angle, count in stats['angles'].items()])
        intermediate_candidate_data.append({
            "Company": company_name, "Product": product_name,
            "Total Chunks": stats['total_chunks'], "Source Angles": angle_breakdown
        })
    df_intermediate = pd.DataFrame(intermediate_candidate_data).reset_index(names=['Rank (by chunk count)'])

    st.markdown("### Intermediate Candidate List (Ranked by Chunk Count)")
    if intermediate_candidate_data:
        st.dataframe(df_intermediate, use_container_width=True, hide_index=True)
    else:
         st.warning("No candidates were found in the RAG retrieval step.")

    st.markdown("---")

    # --- STAGE 4: Batch Initial Synergy Analysis and Ranking ---
    st.subheader("3️⃣ Automated Initial Synergy Ranking (LLM 3)")
    all_initial_results = []
    product_chunks_map = {}
    for doc in all_retrieved_chunks:
        company_name = doc.metadata.get('Company')
        # --- FIX: Chunk grouping now uses the clean 'Product' key ---
        product_name = doc.metadata.get('Product')
        if isinstance(company_name, str) and isinstance(product_name, str):
             product_key = (company_name, product_name)
             product_chunks_map.setdefault(product_key, []).append(doc.page_content)

    num_candidates = len(unique_candidates_keys)
    ranking_progress = st.progress(0, text=f"Starting initial analysis of {num_candidates} candidates...")

    for i, (company_name, product_name) in enumerate(unique_candidates_keys):
        ranking_progress.progress((i + 1) / num_candidates, text=f"Analyzing candidate {i+1}/{num_candidates}: {product_name}...")

        candidate_chunks = product_chunks_map.get((company_name, product_name), [])
        candidate_chunks_text = "\n\n---\n\n".join(candidate_chunks)
        clean_candidate_chunks_text = candidate_chunks_text.encode("ascii", "ignore").decode("ascii")

        initial_analysis = scorer_chain.invoke({
            "target_doc": clean_target_doc,
            "candidate_chunks_text": clean_candidate_chunks_text
        })
        score = initial_analysis.get('score', 0)
        reasoning = initial_analysis.get('reasoning', 'No reasoning provided.')

        all_initial_results.append({
            "Rank": 0, "Score": score, "Product": product_name,
            "Company": company_name, "Justification": reasoning
        })
        time.sleep(0.1)

    ranking_progress.progress(1.0, text="Analysis complete."); time.sleep(1); ranking_progress.empty()
    st.success("All candidates analyzed.")

    ranked_shortlist = sorted(all_initial_results, key=lambda x: x['Score'], reverse=True)
    for i, result in enumerate(ranked_shortlist):
        result['Rank'] = i + 1
    df_final = pd.DataFrame(ranked_shortlist)

    return df_final, df_intermediate, synergy_strategies

# --- 3. STREAMLIT UI LAYOUT AND LOGIC ---

st.set_page_config(layout="wide", page_title="RAG Partnership Shortlist")
st.title("💡 Pipeline 1: RAG-Based Partnership Shortlisting")
st.markdown("This app runs a multi-stage LangChain RAG pipeline to dynamically identify and rank potential partners for a selected **Target Product**.")

# --- Initialize Session State ---
if 'pipeline_stage' not in st.session_state: st.session_state.pipeline_stage = 'setup'
if 'estimated_cost_data' not in st.session_state: st.session_state.estimated_cost_data = None
if 'final_results' not in st.session_state: st.session_state.final_results = None

# --- SIDEBAR: Configuration and Input ---
with st.sidebar:
    st.header("Configuration")
    if 'api_key' not in st.session_state: st.session_state.api_key = ""
    api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.api_key)
    st.session_state.api_key = api_key
    num_angles = st.slider("Number of Synergy Angles (N)", 1, 5, 2, 1)
    if st.button("Reset Session"):
        st.session_state.clear(); st.cache_resource.clear(); st.cache_data.clear(); st.rerun()

    st.header("Target Product Selection")
    if not os.path.exists(LOCAL_DATA_FILE):
        st.error(f"Data file not found at '{LOCAL_DATA_FILE}'."); st.stop()
    merged_data, product_options = load_data(LOCAL_DATA_FILE)
    if merged_data is None: st.stop()
    target_product_input = st.selectbox("Select Target Product", product_options, index=0)

    if st.button("Estimate Cost & Setup Pipeline", type="primary"):
        if not api_key or target_product_input == product_options[0]:
            st.error("Please select a product and provide an API Key.")
            st.session_state.pipeline_stage = 'setup'
        else:
            try:
                vectorstore = load_vectorstore(api_key)
                if vectorstore is None:
                    st.session_state.pipeline_stage = 'setup'
                else:
                    total_tokens, total_cost, N_c_estimate = estimate_pipeline_cost(api_key, merged_data, target_product_input, num_angles)
                    st.session_state.estimated_cost_data = {
                        'target': target_product_input, 'angles': num_angles,
                        'tokens': total_tokens, 'cost': total_cost, 'N_c': N_c_estimate
                    }
                    st.session_state.pipeline_stage = 'estimated'
                    st.success("Cost estimate ready for review.")
            except Exception as e:
                st.error(f"Setup Error: {e}"); st.session_state.pipeline_stage = 'setup'
        st.rerun()

# --- MAIN CONTENT EXECUTION ---

if st.session_state.pipeline_stage == 'estimated':
    cost_data = st.session_state.estimated_cost_data
    st.header("💰 Cost Confirmation Required")
    st.warning(f"Target: **{cost_data['target']}** | Angles: **{cost_data['angles']}**")
    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.metric("Estimated Total Cost (GPT-4o)", f"${cost_data['cost']:.4f} USD")
    col2.metric("Estimated Total Tokens", f"{cost_data['tokens']:,}")
    st.info(f"Estimate based on **{cost_data['N_c']}** potential candidates. Actual cost may be lower.")
    if st.button("CONFIRM & RUN FULL PIPELINE", type="secondary"):
        st.session_state.pipeline_stage = 'running'; st.rerun()

if st.session_state.pipeline_stage == 'running':
    st.header("🚀 Running RAG Pipeline...")
    st.markdown("---")
    try:
        vectorstore = load_vectorstore(api_key)
        df_final, df_intermediate, synergy_strategies = run_pipeline_execution(
            st.session_state.api_key, merged_data, target_product_input, num_angles, vectorstore
        )
        st.session_state.final_results = {
            'final_df': df_final, 'intermediate_df': df_intermediate,
            'synergy_strategies': synergy_strategies
        }
        st.session_state.pipeline_stage = 'complete'
        st.rerun()
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
        st.session_state.pipeline_stage = 'setup'
        st.session_state.estimated_cost_data = None
        st.exception(e)

if st.session_state.pipeline_stage == 'complete':
    st.header("✅ Pipeline Complete: Results")
    st.markdown("---")
    results = st.session_state.final_results
    cost_data = st.session_state.estimated_cost_data
    df_intermediate = results.get('intermediate_df')
    df_final = results.get('final_df')
    synergy_strategies = results.get('synergy_strategies')

    if cost_data:
        st.subheader(f"Setup Summary (Target: {cost_data['target']})")
        col1, col2, col3 = st.columns(3)
        col1.metric("Synergy Angles (N)", cost_data['angles'])
        col2.metric("Est. Total Tokens", f"{cost_data['tokens']:,}")
        col3.metric("Est. Cost (GPT-4o)", f"${cost_data['cost']:.4f} USD")
        st.markdown("---")

    if synergy_strategies:
        st.subheader("1️⃣ Dynamic Strategy Generation (LLM 1)")
        st.json(synergy_strategies)
        st.markdown("---")

    if df_intermediate is not None and not df_intermediate.empty:
        st.subheader("2️⃣ Candidate List (Ranked by Chunk Count)")
        st.info("Candidates retrieved by RAG, ranked by volume of matching chunks.")
        st.dataframe(df_intermediate, use_container_width=True, hide_index=True)
        st.markdown("---")
    elif df_intermediate is not None:
         st.warning("No candidates were found in the RAG retrieval step.")

    if df_final is not None and not df_final.empty:
        st.subheader("3️⃣ Final Ranked Synergy Shortlist (Ranked by LLM Score)")
        st.info("The final ranking after LLM-powered synergy assessment (LLM 3).")
        st.dataframe(df_final.style.bar(subset=['Score'], color='#5fb2b7'), use_container_width=True, hide_index=True)
    else:
        st.warning("No final results were generated.")
    st.balloons()