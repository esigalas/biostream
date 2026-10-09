import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pipeline import run_scfv_developability_pipeline, score_scfv_variants
import py3Dmol
from stmol import showmol
import plotly.graph_objects as go

st.set_page_config(page_title="scFv Developability", layout="wide")
st.title("🧬 scFv Variant Developability Pipeline")

# Initialize session state for storing Phase 2 results
if 'scored_df' not in st.session_state:
    st.session_state.scored_df = None

# Set up the two phases as distinct tabs
tab1, tab2 = st.tabs(["Phase 1: Generation", "Phase 2: Evaluation"])

# ==========================================
# TAB 1: PHASE 1 (Generation)
# ==========================================
with tab1:
    st.header("1. Structural Generation & Inverse Folding")
    st.markdown("Predict 3D topology, relax coordinates, and generate framework mutations using an ensemble of models.")
    
    col1, col2 = st.columns(2)
    with col1:
        run_name = st.text_input("Sequence Name", value="scFvLA-L2K")
        base_dir = st.text_input("Workspace Directory", value="./zero-shot_results")
        
        # NEW: Ensemble UI Controls
        st.markdown("**Ensemble Settings**")
        gen_models = st.multiselect(
            "Generative Models to Run:",
            options=["AntiFold (Antibody-specific)", "ProteinMPNN (Diverse)", "ProteinMPNN (Soluble)", "ThermoMPNN (ddG-Driven)", "ESM-C (Evolutionary LM)", "AbLang2 (Antibody LM)"],
            default=["AntiFold (Antibody-specific)"]
        )
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            num_seqs = st.number_input("Sequences per model", min_value=10, max_value=500, value=100, step=50)
        with col_s2:
            temp = st.slider("Sampling Temperature", 0.1, 1.0, 0.2, step=0.1)

    with col2:
        vh_seq = st.text_area("VH Sequence")
        vl_seq = st.text_area("VL Sequence")

    # 1. The Execution Block
    if st.button("Run Phase 1 (Ensemble)", type="primary"):
        if not vh_seq or not vl_seq:
            st.error("Please provide both VH and VL sequences.")
        elif not gen_models:
            st.error("Please select at least one generative model.")
        else:
            with st.spinner("Running Phase 1 Ensemble (ESMFold -> PyRosetta -> Generators)..."):
                # Pass the new ensemble arguments to the pipeline
                run_scfv_developability_pipeline(
                    vh_seq.strip(), vl_seq.strip(), base_dir, run_name, 
                    gen_models=gen_models, num_seqs=num_seqs, temp=temp
                )
                st.success("Phase 1 Complete! Sequences pooled and deduplicated.")
                
    # 2. The Persistent Download Block
    fasta_path = os.path.join(base_dir, run_name, f"{run_name}_camsol_upload.fasta")
    pdb_path = os.path.join(base_dir, run_name, f"{run_name}_relaxed.pdb")
    
    if os.path.exists(fasta_path):
        st.markdown("---")
        st.subheader("📥 Next Steps: Solubility Prediction")
        
        with open(fasta_path, "r") as f:
            fasta_data = f.read()
        
        st.download_button(
            label="⬇️ Download Ensemble FASTA for CamSol",
            data=fasta_data,
            file_name=f"{run_name}_camsol_upload.fasta",
            mime="text/plain"
        )
        
        st.info("Upload the downloaded FASTA file to the CamSol web server to calculate intrinsic solubility.\n\n🔗 [Open Cambridge CamSol Web Server](https://www-cohsoftware.ch.cam.ac.uk/index.php/camsolintrinsic)")

    # 3. Interactive 3D Structure Viewer
    if os.path.exists(pdb_path):
        st.markdown("---")
        st.subheader("🧬 3D Structure Viewer (PyRosetta Relaxed)")
        
        # UI Controls for the Viewer
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            show_sticks = st.checkbox("Show Amino Acid Sidechains (Sticks)", value=False)
        with col_v2:
            show_surface = st.checkbox("Show Protein Surface", value=False)
            
        with open(pdb_path, "r") as f:
            pdb_string = f.read()
            
        # Initialize the 3Dmol.js viewer
        view = py3Dmol.view(width=800, height=500)
        view.addModel(pdb_string, "pdb")
        
        # Base cartoon style
        view.setStyle({'chain': 'A'}, {'cartoon': {'color': '#1f77b4'}})
        view.setStyle({'chain': 'B'}, {'cartoon': {'color': '#ff7f0e'}})
        
        # Overlay sidechain sticks if toggled
        if show_sticks:
            view.addStyle({'chain': 'A'}, {'stick': {'colorscheme': 'blueCarbon', 'radius': 0.15}})
            view.addStyle({'chain': 'B'}, {'stick': {'colorscheme': 'orangeCarbon', 'radius': 0.15}})
            
        # Overlay semi-transparent surface if toggled
        if show_surface:
            view.addSurface(py3Dmol.VDW, {'opacity': 0.4, 'color': 'white'})
            
        # Inject JavaScript for RCSB-style hover tooltips
        hover_js = """
        function(atom,viewer,event,container) {
            if(!atom.label) {
                atom.label = viewer.addLabel(
                    atom.resn + atom.resi + " (" + atom.chain + ")",
                    {position: atom, backgroundColor: '#222222', fontColor: 'white', backgroundOpacity: 0.85}
                );
            }
        }
        """
        unhover_js = """
        function(atom,viewer) { 
            if(atom.label) {
                viewer.removeLabel(atom.label);
                delete atom.label;
            }
        }
        """
        view.setHoverable({}, True, hover_js, unhover_js)
        
        # Center the molecule and render it in Streamlit
        view.zoomTo()
        showmol(view, height=500, width=800)

# ==========================================
# TAB 2: PHASE 2 (Evaluation)
# ==========================================
with tab2:
    st.header("2. Biophysical Scoring & Selection")
    st.markdown("Merge CamSol results, calculate thermodynamic stability via JanusDDG, and filter candidates.")
    
    col3, col4 = st.columns(2)
    with col3:
        p2_run_name = st.text_input("Sequence Name (Processed in Phase 1)", value="scFvLA-L2K", key="p2_name")
        p2_base_dir = st.text_input("Workspace Directory", value="./zero-shot_results", key="p2_dir")
    with col4:
        camsol_upload = st.file_uploader("Upload Intrinsic Solubility Results (CamSol .txt)", type=["txt", "tsv"])

    if st.button("Run Phase 2 (Scoring)", type="primary"):
        run_dir = os.path.join(p2_base_dir, p2_run_name)
        fasta_path = os.path.join(run_dir, f"{p2_run_name}_camsol_upload.fasta")
        
        if camsol_upload is None:
            st.error("Please upload the CamSol results file.")
        elif not os.path.exists(fasta_path):
            st.error("Could not find the Phase 1 FASTA file. Ensure your Run Name and Base Directory are correct.")
        else:
            os.makedirs(run_dir, exist_ok=True)
            
            # Save the uploaded CamSol file directly into the run directory
            camsol_path = os.path.join(run_dir, f"{p2_run_name}_camsol.txt")
            with open(camsol_path, "wb") as f:
                f.write(camsol_upload.getbuffer())
                
            # Dynamically fetch the Wild-Type sequence from the Phase 1 FASTA
            from Bio import SeqIO
            with open(fasta_path, "r") as f:
                wt_seq = str(next(SeqIO.parse(f, "fasta")).seq)
            
            with st.spinner("Calculating local JanusDDG stability..."):
                try:
                    df = score_scfv_variants(run_dir, p2_run_name, wt_seq)
                    st.session_state.scored_df = df
                    st.success("Scoring Complete!")
                except Exception as e:
                    st.error(f"Error during scoring: {e}")

    # ---------------------------------------------------------
    # Interactive Thresholding
    # ---------------------------------------------------------
    if st.session_state.scored_df is not None:
        df = st.session_state.scored_df.copy()
        
        st.markdown("---")
        
        # Pre-calculate min/max for all sliders
        camsol_min, camsol_max = float(df['camsol_score'].min()), float(df['camsol_score'].max())
        ddg_min, ddg_max = float(df['janus_ddg'].min()), float(df['janus_ddg'].max())
        charge_min, charge_max = float(df['net_charge'].min()), float(df['net_charge'].max())
        bind_min, bind_max = float(df['bindpred_kd'].min()), float(df['bindpred_kd'].max())
        
        col_left, col_right = st.columns(2)
        
        # ==========================================
        # 1. READ ALL UI SLIDERS FIRST
        # ==========================================
        with col_left:
            st.subheader("1. Stability & Solubility")
            ddg_threshold = st.slider(
                "Min. Thermodynamic Stability (JanusDDG ΔΔG)", 
                min_value=ddg_min - 0.5, max_value=ddg_max + 0.5, value=-1.0, step=0.1
            )
            camsol_threshold = st.slider(
                "Min. Intrinsic Solubility (CamSol Score)", 
                min_value=camsol_min, max_value=camsol_max, value=float(df['camsol_score'].quantile(0.10)), step=0.05
            )

        with col_right:
            st.subheader("2. VH-VL Affinity & Bioprocessing")
            bindpred_threshold = st.slider(
                "Max. Dissociation Constant (BINDPRED Log10 Kd)", 
                min_value=bind_min, max_value=bind_max, value=bind_max, step=0.1,
                help="Lower values indicate tighter VH-VL interface binding."
            )
            charge_range = st.slider(
                "Acceptable Net Charge Range (pH 7.4)", 
                min_value=charge_min - 1.0, max_value=charge_max + 1.0, 
                value=(charge_min, charge_max), step=0.5
            )

        # ==========================================
        # 2. CALCULATE GLOBAL SURVIVAL
        # ==========================================
        df['survives_ddg'] = df['janus_ddg'] >= ddg_threshold
        df['survives_camsol'] = df['camsol_score'] >= camsol_threshold
        df['survives_bind'] = df['bindpred_kd'] <= bindpred_threshold
        df['survives_charge'] = df['net_charge'].between(charge_range[0], charge_range[1])
        
        # The ultimate winners
        df['survival_status'] = df['survives_ddg'] & df['survives_camsol'] & df['survives_bind'] & df['survives_charge']
        
        # Assign 3-tier color categories for Plot 1
        df['plot1_color'] = 'Fails Local'
        df.loc[df['survives_ddg'] & df['survives_camsol'], 'plot1_color'] = 'Passes Local (Fails Other)'
        df.loc[df['survival_status'], 'plot1_color'] = 'Passes All 4 Metrics'
        
        # Assign 3-tier color categories for Plot 2
        df['plot2_color'] = 'Fails Local'
        df.loc[df['survives_bind'] & df['survives_charge'], 'plot2_color'] = 'Passes Local (Fails Other)'
        df.loc[df['survival_status'], 'plot2_color'] = 'Passes All 4 Metrics'
        
        color_map = {
            'Passes All 4 Metrics': '#2ca02c',       # Green
            'Passes Local (Fails Other)': '#7F00FF',#'#1f77b4', # Blue
            'Fails Local': '#d62728'                 # Red
        }

        # ==========================================
        # 3. RENDER PLOTS BACK INTO COLUMNS
        # ==========================================
        with col_left:
            fig1 = px.scatter(
                df, x="janus_ddg", y="camsol_score", 
                color="plot1_color", color_discrete_map=color_map,
                hover_name="id", hover_data={"mutations": True},
                labels={"janus_ddg": "Thermodynamic Stability (ΔΔG)", "camsol_score": "Intrinsic Solubility", "plot1_color": ""},
                opacity=0.8
            )
            fig1.add_vline(x=ddg_threshold, line_dash="dash", line_color="gray")
            fig1.add_hline(y=camsol_threshold, line_dash="dot", line_color="gray")
            fig1.update_layout(
                height=500, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col_right:
            fig2 = px.scatter(
                df, x="net_charge", y="bindpred_kd", 
                color="plot2_color", color_discrete_map=color_map,
                hover_name="id", hover_data={"mutations": True},
                labels={"net_charge": "Net Charge (pH 7.4)", "bindpred_kd": "VH-VL Affinity (Log10 Kd)", "plot2_color": ""},
                opacity=0.8
            )
            fig2.add_hline(y=bindpred_threshold, line_dash="dash", line_color="gray")
            fig2.add_vline(x=charge_range[0], line_dash="dot", line_color="gray")
            fig2.add_vline(x=charge_range[1], line_dash="dot", line_color="gray")
            fig2.update_layout(
                height=500, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ==========================================
        # Final Export Table & Multi-Metric Radar Chart
        # ==========================================
        st.markdown("---")
        
        passed_df = df[df['survival_status'] == True].sort_values('global_composite_score', ascending=False).reset_index(drop=True)
        st.markdown(f"**Top Candidates Passing All Thresholds ({len(passed_df)} found)**")
        
        if len(passed_df) > 0:
            col_radar, col_table = st.columns([1, 2])
            
            with col_radar:
                st.subheader("7-Axis Developability Footprint")
                
                selected_ids = st.multiselect(
                    "Select Variants to Compare:", 
                    options=passed_df['id'], 
                    default=passed_df['id'].head(2).tolist()
                )
                
                if selected_ids:
                    radar_fig = go.Figure()
                    # 7 categories matching our 7 normalizations
                    categories = [
                        'Solubility (CamSol)', 'Stability (Janus)', 'Affinity (BINDPRED)', 
                        'General (ESM-2)', 'Ab-Specific (AntiBERTy)', 
                        'Paired-Ab (AbLang2)', 'Evolutionary (ProtBERT)'
                    ]
                    
                    for variant_id in selected_ids:
                        v_data = df[df['id'] == variant_id].iloc[0]
                        
                        values = [
                            v_data.get('camsol_norm', 0.5),
                            v_data.get('janus_norm', 0.5), 
                            v_data.get('bindpred_norm', 0.5),
                            v_data.get('esm_norm', 0.5),
                            v_data.get('ab_norm', 0.5),
                            v_data.get('ablang_norm', 0.5),
                            v_data.get('protbert_norm', 0.5)
                        ]
                        
                        values.append(values[0]) # Close the polygon
                        cat_closed = categories + [categories[0]]
                        
                        radar_fig.add_trace(go.Scatterpolar(
                            r=values, theta=cat_closed,
                            fill='toself', name=variant_id,
                            hoverinfo="text",
                            text=[f"{val:.2f}" for val in values]
                        ))
                        
                    radar_fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=True, legend=dict(orientation="h", y=-0.2),
                        margin=dict(l=40, r=40, t=20, b=20), height=400
                    )
                    st.plotly_chart(radar_fig, use_container_width=True)
                else:
                    st.info("Select a variant to view its developability footprint.")

            with col_table:
                rename_map = {
                    'id': 'Sequence ID', 'mutations': 'Mutations',
                    'global_composite_score': 'Global Score', 
                    'esm_fitness': 'ESM-2 (General)', 
                    'antiberty_fitness': 'AntiBERTy (Ab-Specific)',
                    'ablang_fitness': 'AbLang2 (Paired)',
                    'protbert_fitness': 'ProtBERT (Evolution)',
                    'janus_ddg': 'Stability (Janus)',
                    'camsol_score': 'Solubility (CamSol)', 
                    'bindpred_kd': 'Affinity (BINDPRED)',
                    'net_charge': 'Net Charge',
                    'VH': 'VH Sequence', 'VL': 'VL Sequence'
                }
                
                # REORDERED: All 4 PLMs grouped immediately after the Global Score
                download_cols = [
                    'id', 'mutations', 'global_composite_score', 
                    'esm_fitness', 'antiberty_fitness', 'ablang_fitness', 'protbert_fitness',
                    'janus_ddg', 'camsol_score', 'bindpred_kd', 'net_charge', 
                    'VH', 'VL'
                ]
                
                download_df = passed_df[download_cols].rename(columns=rename_map)
                
                # Filter out the heavy sequence strings for the UI table
                display_cols = [c for c in download_df.columns if c not in ['VH Sequence', 'VL Sequence']]
                display_df = download_df[display_cols]
                
                st.dataframe(
                    display_df.style.background_gradient(cmap='viridis', subset=['Global Score']),
                    use_container_width=True, hide_index=True
                )
                
                st.download_button(
                    label="Download Filtered Candidates",
                    data=download_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{p2_run_name}_top_candidates.csv",
                    mime='text/csv'
                )
