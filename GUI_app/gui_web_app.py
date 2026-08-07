import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import glob
import os

# 🌟 NEW: We must import the custom target transformers directly from your 
# EDA script so joblib knows how to un-pickle the advanced mathematical objects!
from bioprocessing_eda import (
    extract_sequence_features, 
    custom_spearman,
    calculate_weighted_yeojohnson_lambda, 
    CustomYeoJohnsonTransformer, 
    SelfContainedTargetTransformRegressor
)

st.set_page_config(page_title="Antibody ML Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("🧪 Antibody Sequence & Developability Dashboard")
st.markdown("Visualize sequence building blocks, analyze machine learning predictions, and test *in-silico* mutations in real-time.")

# --- Sidebar Configuration ---
#st.sidebar.header("1. Loaded Models")

# Automatically find the models based on target names
poly_files = glob.glob("trained_models/*ELISA_Polyreactivity_Excell*.joblib")
hmw_files = glob.glob("trained_models/*HMW_Excell*.joblib")

if not poly_files or not hmw_files:
    st.sidebar.error("Could not find both Polyreactivity and HMW models in 'trained_models/'. Please ensure they exist.")
    st.stop()

@st.cache_resource
def load_model(filepath):
    return joblib.load(filepath)

# Load Polyreactivity Model
model_dict_poly = load_model(poly_files[0])
model_poly = model_dict_poly['model']
features_poly = model_dict_poly['features']
target_poly = model_dict_poly['target']

# Load HMW Model
model_dict_hmw = load_model(hmw_files[0])
model_hmw = model_dict_hmw['model']
features_hmw = model_dict_hmw['features']
target_hmw = model_dict_hmw['target']

# Helper to determine ESM size per model
def get_esm_name(features):
    if any("_ESM_Massive_3B_" in f for f in features): return "facebook/esm2_t36_3B_UR50D"
    if any("_ESM_Big_650M_" in f for f in features): return "facebook/esm2_t33_650M_UR50D"
    if any("_ESM_Large_150M_" in f for f in features): return "facebook/esm2_t30_150M_UR50D"
    if any("_ESM_Medium_35M_" in f for f in features): return "facebook/esm2_t12_35M_UR50D"
    return "facebook/esm2_t6_8M_UR50D" # Fallback to small

esm_poly = get_esm_name(features_poly)
esm_hmw = get_esm_name(features_hmw)

#st.sidebar.success(f"Models loaded successfully!\n- **{target_poly}**\n- **{target_hmw}**")

st.sidebar.header("1. Load Environment")

# 🌟 NEW: Dynamically find all CSV files in the current directory and the 'data' folder
csv_files = glob.glob("*.csv") + glob.glob("data/*.csv")

if not csv_files:
    st.sidebar.warning("No .csv dataset files found! Please add your data files to the folder.")
    st.stop()
    
# 🌟 NEW: Use format_func to hide the folder path in the dropdown UI
data_file = st.sidebar.selectbox(
    "Select Dataset:", 
    csv_files,
    format_func=lambda x: os.path.basename(x)
)


# 🌟 NEW: Separated Extraction Function
@st.cache_data
def load_base_environment_for_model(filepath, esm_model_name):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    id_cols = [c for c in df.columns if 'ID' in c.upper()]
    if id_cols:
        df['Display_ID'] = df[id_cols[0]].astype(str)
    else:
        df['Display_ID'] = "Row_" + df.index.astype(str)
        
    dataset_name = os.path.splitext(os.path.basename(filepath))[0]
    
    df_feat, seq_cols, _, _ = extract_sequence_features(
        df, is_inference=False, dataset_name=dataset_name, esm_model_name=esm_model_name
    )
    return df, df_feat, seq_cols

if os.path.exists(data_file):
    with st.spinner("Extracting Polyreactivity features..."):
        df, df_feat_poly, seq_cols = load_base_environment_for_model(data_file, esm_poly)
    with st.spinner("Extracting HMW features..."):
        _, df_feat_hmw, _ = load_base_environment_for_model(data_file, esm_hmw)
    st.sidebar.success(f"Base data loaded! ({len(df)} antibodies found)")
else:
    st.sidebar.error("Dataset not found. Please check filepath.")
    st.stop()

# --- Main Dashboard Tabs ---
tab1, tab2 = st.tabs(["🧬 Sequence Alignment Viewer", "🔬 Mutagenesis & Prediction"])

with tab1:
    st.subheader("Antibody Subregions (Building Blocks)")
    st.markdown("View all the variable sequence regions loaded from your dataset.")
    
    view_mode = st.radio("View Mode:", ["Colored Alignment (Physicochemical)", "Raw Data Table"], horizontal=True)
    
    if view_mode == "Colored Alignment (Physicochemical)":
        def colorize_sequence(seq):
            if pd.isna(seq) or str(seq).strip().upper() == 'NAN': 
                return ""
            
            # Standard Physicochemical Color Scheme (Darker, more vibrant pastels for better contrast)
            color_map = {
                'A': '#A5D6A7', 'I': '#A5D6A7', 'L': '#A5D6A7', 'M': '#A5D6A7', 'F': '#A5D6A7', 'W': '#A5D6A7', 'V': '#A5D6A7', # Hydrophobic (Green)
                'K': '#90CAF9', 'R': '#90CAF9', 'H': '#90CAF9', # Basic (Blue)
                'D': '#EF9A9A', 'E': '#EF9A9A', # Acidic (Red)
                'N': '#FFCC80', 'Q': '#FFCC80', 'S': '#FFCC80', 'T': '#FFCC80', # Polar (Orange)
                'C': '#FFF59D', # Cysteine (Yellow)
                'G': '#E0E0E0', 'P': '#E0E0E0' # Gly/Pro (Grey)
            }
            
            # Simplified HTML to avoid visual clutter (removed borders and individual margins)
            html = '<span style="font-family: \'Courier New\', Courier, monospace; letter-spacing: 1.5px; font-weight: 600;">'
            for aa in str(seq).upper().replace(' ', '').replace(',', ''):
                bg = color_map.get(aa, 'transparent')
                html += f'<span style="background-color:{bg}; border-radius: 2px;">{aa}</span>'
            html += '</span>'
            return html
        
        with st.spinner("Rendering colored alignment viewer..."):
            html_df = df[seq_cols].copy()
            for col in seq_cols:
                html_df[col] = html_df[col].apply(colorize_sequence)
            
            # Insert the actual ID column so you know which antibody you are looking at
            html_df.insert(0, "Antibody ID", df['Display_ID'])
            
            # Generate the raw HTML table
            table_html = html_df.to_html(escape=False, index=False, classes="seq-table")
            
            # Flush-left HTML string to prevent Streamlit from escaping it as a Markdown code block!
            full_html = f"""<div style="height: 600px; overflow-y: auto; overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 5px; margin-bottom: 15px;">
<style>
.seq-table {{ width: 100%; border-collapse: collapse; font-size: 14px; text-align: left; font-family: sans-serif; }}
.seq-table th {{ background-color: #f0f2f6; color: #31333f; padding: 12px 10px; font-weight: bold; position: sticky; top: 0; z-index: 1; border-bottom: 2px solid #e6e9ef; }}
.seq-table td {{ padding: 6px 10px; border-bottom: 1px solid #f0f2f6; vertical-align: middle; white-space: nowrap; }}
.seq-table tr:hover {{ background-color: #f8f9fb; }}
</style>
{table_html}
</div>"""
            st.write(full_html, unsafe_allow_html=True)
            
            # Provide the visual legend directly below the table
            st.markdown("""
            **Color Legend:** 🟢 `Hydrophobic (A, I, L, M, F, W, V)` | 
            🔵 `Basic (K, R, H)` | 
            🔴 `Acidic (D, E)` | 
            🟠 `Polar (N, Q, S, T)` | 
            🟡 `Cysteine (C)` | 
            ⚪ `Gly/Pro (G, P)`
            """)
    else:
        # Fallback to the raw dataframe if they want to copy-paste the raw strings easily
        st.dataframe(df[['Display_ID'] + seq_cols], use_container_width=True, height=600)

with tab2:
    # --- 1. Prepare Data & Predictions First ---
    X_base_poly = df_feat_poly[features_poly].copy()
    X_base_hmw = df_feat_hmw[features_hmw].copy()
    
    y_pred_poly = np.asarray(model_poly.predict(X_base_poly)).flatten()
    y_pred_hmw = np.asarray(model_hmw.predict(X_base_hmw)).flatten()
    
    # State management for selected parent
    if 'parent_idx' not in st.session_state:
        st.session_state.parent_idx = df.index[0]
        
    parent_idx = st.session_state.parent_idx
    parent_pred_poly = y_pred_poly[df.index.get_loc(parent_idx)]
    parent_pred_hmw = y_pred_hmw[df.index.get_loc(parent_idx)]
    
    plot_df = pd.DataFrame({
        'Predicted_Poly': y_pred_poly,
        'Predicted_HMW': y_pred_hmw,
        'Index': df.index,
        'Antibody ID': df['Display_ID']
    })

    # --- 2. Top Header & Parent Selection Controls ---
    st.subheader("🧬 In-Silico Sequence Editor")
    
    # 🌟 FIX: Use a callback to update the true parent state BEFORE the script runs. 
    # This completely eliminates the double-rerun scroll jump!
    def update_parent_from_dropdown():
        st.session_state.parent_idx = st.session_state.parent_dropdown

    st.selectbox(
        "Select Parent Antibody Scaffold:",
        df.index,
        index=list(df.index).index(st.session_state.parent_idx),
        format_func=lambda x: df.loc[x, 'Display_ID'],
        key='parent_dropdown',
        on_change=update_parent_from_dropdown
    )
    # --- 3. Sequence Editor Form (Full Width Grid) ---
    with st.form("mutation_form"):
        st.markdown("**🧬 Subregion Sequences (Edit below):**")
        
        # Display sequence text boxes side-by-side in 3 columns
        num_grid_cols = 3
        grid_cols = st.columns(num_grid_cols)
        
        edited_seqs = {}
        for idx, col in enumerate(seq_cols):
            with grid_cols[idx % num_grid_cols]:
                # 🌟 FIX: Swapped text_area for text_input to create compact, single-line boxes
                edited_seqs[col] = st.text_input(
                    f"{col}:", 
                    value=df.loc[parent_idx, col], 
                    key=f"edit_{col}_{st.session_state.parent_idx}"
                )
        
        submitted = st.form_submit_button("🧪 Extract Features & Predict Mutation", type="primary", use_container_width=True)

    st.markdown("---")

    # --- 4. Plot and AI Assist Layout ---
    plot_col, ai_col = st.columns([2.5, 1], gap="large")

    # Open the right column FIRST to define the dropdown value before drawing the plot
    with ai_col:
        st.markdown("### ⚙️ Plot Settings")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        symbol_col = st.selectbox(
            "Marker Shape (Group by format/type):", 
            ["None"] + [c for c in cat_cols if c not in seq_cols],
            key="marker_shape_dropdown"
        )

    # Build base Plotly figure (We build the data first, but don't draw it yet!)
    if symbol_col != "None":
        plot_df['Type'] = df[symbol_col]
        fig = px.scatter(
            plot_df, x='Predicted_Poly', y='Predicted_HMW', symbol='Type',
            hover_data={'Antibody ID': True, 'Index': False, 'Predicted_Poly': ':.2f', 'Predicted_HMW': ':.2f', 'Type': True},
            opacity=0.6, color_discrete_sequence=['royalblue']
        )
    else:
        fig = px.scatter(
            plot_df, x='Predicted_Poly', y='Predicted_HMW',
            hover_data={'Antibody ID': True, 'Index': False, 'Predicted_Poly': ':.2f', 'Predicted_HMW': ':.2f'},
            opacity=0.6, color_discrete_sequence=['royalblue']
        )
        
    fig.update_traces(marker=dict(size=10))
    
    # Highlight selected parent
    fig.add_trace(go.Scatter(
        x=[parent_pred_poly], y=[parent_pred_hmw],
        mode='markers',
        marker=dict(size=16, color='rgba(0,0,0,0)', line=dict(color='black', width=3)),
        name='Selected Parent',
        hoverinfo='skip'
    ))

    # --- Construct the Left Column Layout ---
    with plot_col:
        st.subheader("📊 2D Developability Landscape")
        st.markdown("💡 **Tip:** Click on any point in the plot to instantly select it as your parent scaffold!")

        # 🌟 FIX: 1. Reserve an empty space for the plot at the top
        plot_placeholder = st.container()
        
        st.markdown("---")
        
        # 🌟 FIX: 2. Draw the AI Button underneath the reserved plot space
        st.markdown("### 🤖 AI Assist")
        st.markdown("Automatically test naturally occurring variants in hypervariable regions to optimize this sequence.")
        suggest_button = st.button("✨ Auto-Suggest Optimization", type="secondary", use_container_width=True)
        
        # 🌟 FIX: 3. Reserve another space for the output text to appear below the button
        results_placeholder = st.container()

    # --- Handle Manual Mutation Submission ---
    if submitted:
        # Route the text output into the results container at the bottom
        with results_placeholder:
            with st.spinner("Extracting features for mutation..."):
                new_row_df = pd.DataFrame([edited_seqs])
                
                new_feat_df_poly, _, _, _ = extract_sequence_features(new_row_df, is_inference=True, esm_model_name=esm_poly)
                new_feat_df_hmw, _, _, _ = extract_sequence_features(new_row_df, is_inference=True, esm_model_name=esm_hmw)
                
                missing_poly = [f for f in features_poly if f not in new_feat_df_poly.columns]
                missing_hmw = [f for f in features_hmw if f not in new_feat_df_hmw.columns]
                
                if missing_poly or missing_hmw:
                    st.error("Error: The edited sequence dropped required features (likely due to invalid characters).")
                else:
                    X_new_poly = new_feat_df_poly[features_poly].copy()
                    X_new_hmw = new_feat_df_hmw[features_hmw].copy()
                    
                    if hasattr(model_poly, 'weight_col') and model_poly.weight_col:
                        X_new_poly[model_poly.weight_col] = 1.0 
                    if hasattr(model_hmw, 'weight_col') and model_hmw.weight_col:
                        X_new_hmw[model_hmw.weight_col] = 1.0 
                        
                    y_new_pred_poly = np.asarray(model_poly.predict(X_new_poly)).flatten()[0]
                    y_new_pred_hmw = np.asarray(model_hmw.predict(X_new_hmw)).flatten()[0]
                    
                    fig.add_trace(go.Scatter(
                        x=[y_new_pred_poly], y=[y_new_pred_hmw],
                        mode='markers',
                        marker=dict(size=20, color='gold', symbol='star', line=dict(color='black', width=2)),
                        name='Mutated Sequence',
                        hoverinfo='skip'
                    ))
                    
                    fig.add_shape(
                        type="line", line=dict(dash="dot", color="gold", width=2),
                        x0=parent_pred_poly, y0=parent_pred_hmw, x1=y_new_pred_poly, y1=y_new_pred_hmw
                    )
                    
                    st.success(f"### Predicted {target_poly}: **{y_new_pred_poly:.2f}** | Predicted {target_hmw}: **{y_new_pred_hmw:.2f}%**")
                    st.info(f"*(Parent Scaffold — {target_poly}: {parent_pred_poly:.2f} | {target_hmw}: {parent_pred_hmw:.2f}%)*")

    # --- Handle Auto-Suggest Optimization ---
    elif suggest_button:
        # Route the AI output into the results container at the bottom
        with results_placeholder:
            with st.spinner("Analyzing dataset variability and bulk-testing mutations..."):
                
                variability = df[seq_cols].nunique().sort_values(ascending=False)
                top_variable_cols = variability.head(3).index.tolist()
                
                candidates = []
                for col in top_variable_cols:
                    parent_val = df.loc[parent_idx, col]
                    top_variants = df[col][df[col] != parent_val].value_counts().head(3).index.tolist()
                    
                    for variant in top_variants:
                        candidate_seq = edited_seqs.copy()
                        candidate_seq[col] = variant
                        candidate_seq['Mutated_Region'] = col
                        candidate_seq['Variant_Used'] = variant
                        candidates.append(candidate_seq)
                        
                if not candidates:
                    st.warning("No alternative variants found in the dataset to test.")
                else:
                    cand_df = pd.DataFrame(candidates)
                    cand_df_clean = cand_df.drop(columns=['Mutated_Region', 'Variant_Used'])
                    
                    new_feat_df_poly, _, _, _ = extract_sequence_features(cand_df_clean, is_inference=True, esm_model_name=esm_poly)
                    new_feat_df_hmw, _, _, _ = extract_sequence_features(cand_df_clean, is_inference=True, esm_model_name=esm_hmw)
                    
                    X_new_poly = new_feat_df_poly[features_poly].copy()
                    X_new_hmw = new_feat_df_hmw[features_hmw].copy()
                    
                    if hasattr(model_poly, 'weight_col') and model_poly.weight_col:
                        X_new_poly[model_poly.weight_col] = 1.0 
                    if hasattr(model_hmw, 'weight_col') and model_hmw.weight_col:
                        X_new_hmw[model_hmw.weight_col] = 1.0 
                        
                    cand_df['Pred_Poly'] = np.asarray(model_poly.predict(X_new_poly)).flatten()
                    cand_df['Pred_HMW'] = np.asarray(model_hmw.predict(X_new_hmw)).flatten()
                    
                    improved = cand_df[(cand_df['Pred_Poly'] < parent_pred_poly) & (cand_df['Pred_HMW'] < parent_pred_hmw)].copy()
                    
                    if improved.empty:
                        st.warning(f"Routine finished. Tested {len(candidates)} candidates, but none improved BOTH metrics simultaneously.")
                    else:
                        improved['Score'] = improved['Pred_Poly'] + improved['Pred_HMW']
                        improved = improved.sort_values('Score')
                        best_candidate = improved.iloc[0]
                        
                        best_poly = best_candidate['Pred_Poly']
                        best_hmw = best_candidate['Pred_HMW']
                        mutated_region = best_candidate['Mutated_Region']
                        
                        fig.add_trace(go.Scatter(
                            x=[best_poly], y=[best_hmw],
                            mode='markers',
                            marker=dict(size=22, color='#00CC96', symbol='diamond', line=dict(color='black', width=2)),
                            name='Optimized Suggestion',
                            hoverinfo='skip'
                        ))
                        
                        fig.add_shape(
                            type="line", line=dict(dash="dot", color="#00CC96", width=2),
                            x0=parent_pred_poly, y0=parent_pred_hmw, x1=best_poly, y1=best_hmw
                        )
                        
                        st.success(f"### ✨ Optimization Found!")
                        st.markdown(f"Tested **{len(candidates)}** variants across hypervariable regions.")
                        st.markdown(f"**Reduction:** {target_poly} by **{parent_pred_poly - best_poly:.2f}** | {target_hmw} by **{parent_pred_hmw - best_hmw:.2f}%**.")
                        st.info(f"Predicted {target_poly}: **{best_poly:.2f}** | Predicted {target_hmw}: **{best_hmw:.2f}%**")
                        st.code(f"Original {mutated_region}:\n{edited_seqs[mutated_region]}\n\nOptimized {mutated_region}:\n{best_candidate['Variant_Used']}")
                        
                        with st.expander(f"View all {len(improved)} options"):
                            display_df = improved[['Mutated_Region', 'Variant_Used', 'Pred_Poly', 'Pred_HMW']].copy()
                            display_df = display_df.rename(columns={
                                'Mutated_Region': 'Region',
                                'Variant_Used': 'New Sequence',
                                'Pred_Poly': f'Pred {target_poly}',
                                'Pred_HMW': f'Pred {target_hmw}'
                            })
                            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # 🌟 FIX: 4. Finally, push the fully updated plot into the placeholder at the top!
    with plot_placeholder:
        fig.update_layout(
            title="Multivariate Developability Landscape",
            xaxis_title=f"Predicted {target_poly}",
            yaxis_title=f"Predicted {target_hmw}",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            margin=dict(l=0, r=0, t=40, b=0),
            height=550
        )
        
        try:
            event = st.plotly_chart(
                fig, 
                on_select="rerun", 
                selection_mode="points", 
                key="plot_selection", 
                use_container_width=True
            )
            
            if event and event.selection and event.selection.points:
                point = event.selection.points[0]
                clicked_x = point.get("x")
                clicked_y = point.get("y")
                
                if clicked_x is not None and clicked_y is not None:
                    matched_rows = plot_df[
                        np.isclose(plot_df['Predicted_Poly'], clicked_x, atol=1e-5) & 
                        np.isclose(plot_df['Predicted_HMW'], clicked_y, atol=1e-5)
                    ]
                    
                    if not matched_rows.empty:
                        clicked_idx = matched_rows.iloc[0]['Index']
                        if clicked_idx in df.index and clicked_idx != st.session_state.parent_idx:
                            st.session_state.parent_idx = clicked_idx
                            
                            if 'parent_dropdown' in st.session_state:
                                del st.session_state['parent_dropdown']
                                
                            st.rerun() 
        except TypeError:
            st.plotly_chart(fig, use_container_width=True)
