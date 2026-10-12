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
    st.markdown("Calculate thermodynamic stability, evaluate optional solubility metrics, and rank all candidates.")
    
    col3, col4 = st.columns(2)
    with col3:
        p2_run_name = st.text_input("Sequence Name (Processed in Phase 1)", value="scFvLA-L2K", key="p2_name")
        p2_base_dir = st.text_input("Workspace Directory", value="./zero-shot_results", key="p2_dir")
    with col4:
        # CamSol is now explicitly optional
        camsol_upload = st.file_uploader("Upload Intrinsic Solubility Results (CamSol .txt) [Optional]", type=["txt", "tsv"])

    if st.button("Run Phase 2 (Scoring)", type="primary"):
        run_dir = os.path.join(p2_base_dir, p2_run_name)
        fasta_path = os.path.join(run_dir, f"{p2_run_name}_camsol_upload.fasta") # Or your generic phase 1 fasta
        
        if not os.path.exists(fasta_path):
            st.error("Could not find the Phase 1 FASTA file. Ensure your Run Name and Base Directory are correct.")
        else:
            os.makedirs(run_dir, exist_ok=True)
            
            # Save the uploaded CamSol file only if provided
            if camsol_upload is not None:
                camsol_path = os.path.join(run_dir, f"{p2_run_name}_camsol.txt")
                with open(camsol_path, "wb") as f:
                    f.write(camsol_upload.getbuffer())
                st.success("CamSol data detected and integrated.")
            else:
                st.info("No CamSol data provided. Proceeding without solubility scores.")
                
            # Dynamically fetch the Wild-Type sequence from the Phase 1 FASTA
            from Bio import SeqIO
            with open(fasta_path, "r") as f:
                wt_seq = str(next(SeqIO.parse(f, "fasta")).seq)
            
            with st.spinner("Calculating biophysical and structural metrics..."):
                try:
                    df = score_scfv_variants(run_dir, p2_run_name, wt_seq)
                    st.session_state.scored_df = df
                    st.success("Scoring Complete!")
                except Exception as e:
                    st.error(f"Error during scoring: {e}")

    # ---------------------------------------------------------
    # Final Export Table & Multi-Metric Radar Chart
    # ---------------------------------------------------------
    if st.session_state.scored_df is not None:
        df = st.session_state.scored_df.copy()
        
        st.markdown("---")
        st.subheader(f"All Ranked Candidates ({len(df)} found)")
        
        has_camsol = 'camsol_score' in df.columns and df['camsol_score'].notna().any()
        
        # ==========================================
        # 1. Main Data Table & Download Export
        # ==========================================
        rename_map = {
            'id': 'Sequence ID', 
            'mutations': 'Mutations',
            'global_composite_score': 'Global Score', 
            'janus_ddg': 'Stability (Janus)',
            'pyrosetta_ddg': 'Stability (PyRosetta)',
            'camsol_score': 'Solubility (CamSol)', 
            'pI': 'Isoelectric Point (pI)',
            'gravy': 'Hydrophobicity (GRAVY)',
            'net_charge': 'Net Charge',
            'charge_asymmetry': 'VH-VL Charge Asym.',
            'bindpred_kd': 'Affinity (BINDPRED)',
            'esm_fitness': 'ESM-2 (General)', 
            'antiberty_fitness': 'AntiBERTy (Ab-Specific)',
            'ablang_fitness': 'AbLang2 (Paired)',
            'protbert_fitness': 'ProtBERT (Evolution)',
            'VH': 'VH Sequence', 
            'VL': 'VL Sequence'
        }
        
        # --- ENFORCE LOGICAL COLUMN GROUPING FOR UI & DOWNLOAD ---
        download_cols = [
            'id', 'mutations', 'global_composite_score', 
            'janus_ddg', 'pyrosetta_ddg'
        ]
        if has_camsol:
            download_cols.append('camsol_score')
            
        download_cols.extend([
            'pI', 'gravy', 'net_charge', 'charge_asymmetry',
            'bindpred_kd', 'esm_fitness', 'antiberty_fitness', 'ablang_fitness', 'protbert_fitness',
            'VH', 'VL'
        ])
        
        download_df = df[download_cols].rename(columns=rename_map)
        
        # Filter out the heavy sequence strings for the UI table to keep it clean
        display_cols = [c for c in download_df.columns if c not in ['VH Sequence', 'VL Sequence']]
        display_df = download_df[display_cols]
        
        st.dataframe(
            display_df.style.background_gradient(cmap='viridis', subset=['Global Score']),
            use_container_width=True, hide_index=True
        )
        
        st.download_button(
            label="Download All Ranked Candidates",
            data=download_df.to_csv(index=False).encode('utf-8'),
            file_name=f"{p2_run_name}_all_ranked_candidates.csv",
            mime='text/csv'
        )

        # ==========================================
        # 2. Multi-Metric Radar Chart (Below Table)
        # ==========================================
        st.markdown("---")
        st.subheader("Developability Footprint")
        import plotly.graph_objects as go
        
        selected_ids = st.multiselect(
            "Select Variants to Compare:", 
            options=df['id'], 
            default=df['id'].head(2).tolist() if len(df) >= 2 else df['id'].tolist()
        )
        
        if selected_ids:
            radar_fig = go.Figure()
            
            # Dynamically build radar categories based on CamSol presence
            categories = ['Stability (Janus)', 'Stability (PyRosetta)', 'Affinity (BINDPRED)', 
                          'General (ESM-2)', 'Ab-Specific (AntiBERTy)', 
                          'Paired-Ab (AbLang2)', 'Evolutionary (ProtBERT)']
            if has_camsol:
                categories.insert(0, 'Solubility (CamSol)')
            
            for variant_id in selected_ids:
                v_data = df[df['id'] == variant_id].iloc[0]
                
                values = [
                    v_data.get('janus_norm', 0.5),
                    v_data.get('pyrosetta_norm', 0.5),
                    v_data.get('bindpred_norm', 0.5),
                    v_data.get('esm_norm', 0.5),
                    v_data.get('ab_norm', 0.5),
                    v_data.get('ablang_norm', 0.5),
                    v_data.get('protbert_norm', 0.5)
                ]
                if has_camsol:
                    values.insert(0, v_data.get('camsol_norm', 0.5))
                
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
                margin=dict(l=40, r=40, t=20, b=20), height=500
            )
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("Select a variant to view its developability footprint.")

