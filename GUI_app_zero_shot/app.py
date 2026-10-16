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
                
                # Clean the inputs first so the lengths match the actual amino acids
                vh_clean = vh_seq.strip()
                vl_clean = vl_seq.strip()
                
                # Pass the dynamically calculated lengths into the pipeline
                run_scfv_developability_pipeline(
                    vh_clean, vl_clean, base_dir, run_name, 
                    vh_len=len(vh_clean),
                    vl_len=len(vl_clean),
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
            'global_composite_score': 'Global Composite Score', 
            
            # Structural Thermodynamics
            'janus_ddg': 'Stability ΔΔG (Janus)',
            'pyrosetta_ddg': 'Stability ΔΔG (PyRosetta)',
            
            # Sequence Biophysics
            'camsol_score': 'Intrinsic Solubility (CamSol)', 
            'pI': 'Isoelectric Point (pI)',
            'gravy': 'Hydrophobicity Index (GRAVY)',
            'net_charge': 'Net Charge (pH 7.4)',
            'charge_asymmetry': 'VH-VL Charge Delta',
            
            # Interface Affinity
            'bindpred_kd': 'Affinity Log10 Kd (BINDPRED)',
            
            # Deep Learning Language Models
            'esm_fitness': 'ΔLog-Likelihood (ESM-2 General)', 
            'antiberty_fitness': 'ΔLog-Likelihood (AntiBERTy)',
            'ablang_fitness': 'ΔLog-Likelihood (AbLang2 Paired)',
            'protbert_fitness': 'ΔLog-Likelihood (ProtBERT)',
            
            # Raw Data
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
            display_df.style.background_gradient(cmap='viridis', subset=['Global Composite Score']),
            use_container_width=True, hide_index=True
        )
        
        st.download_button(
            label="Download All Ranked Candidates",
            data=download_df.to_csv(index=False).encode('utf-8'),
            file_name=f"{p2_run_name}_all_ranked_candidates.csv",
            mime='text/csv'
        )
        
        # ==========================================
        # 2. Multi-Gate Aggregation Triage & 4D Landscape
        # ==========================================
        st.markdown("---")
        st.subheader("🔬 Anti-Aggregation Multi-Gate Triage & Landscape")
        
        # Isolate the Wild-Type baseline row for relative comparisons
        wt_rows = df[df['mutations'] == 'WT']
        wt_row = wt_rows.iloc[0] if not wt_rows.empty else df.iloc[0]
        
        with st.expander("⚙️ Configure Anti-Aggregation Biophysical Gates (Filter Candidates)", expanded=True):
            st.markdown("Filter the ensemble using orthogonal constraints relative to your **Wild-Type (WT)** baseline:")
            f_col1, f_col2, f_col3 = st.columns(3)
            
            with f_col1:
                st.markdown("**1. Conformational & Interface**")
                max_rosetta = st.slider(
                    "Max PyRosetta ΔΔG (REU) [Lower = Better]", 
                    min_value=-15.0, max_value=25.0, value=10.0, step=0.5,
                    help="In PyRosetta, negative (< 0) is stabilizing. Variants above 25.0 REU have severe steric clashes and receive a 0.0 normalized score."
                )
                # FIX 1: Changed from max_janus (<=) to min_janus (>=) because raw JanusDDG is Positive = Stabilizing
                min_janus = st.slider(
                    "Min Janus ΔΔG (kcal/mol) [Higher = Better]", 
                    min_value=-5.0, max_value=5.0, value=-0.5, step=0.1,
                    help="In raw JanusDDG, positive (> 0) is stabilizing. Setting to -0.5 keeps neutral and stabilizing variants."
                )
                strict_interface = st.checkbox(
                    f"Require Equal/Better VH-VL Affinity (BINDPRED ≤ WT: {wt_row.get('bindpred_kd', 0):.2f})", value=False
                )
                
            with f_col2:
                st.markdown("**2. Colloidal & Electrostatic**")
                improve_solubility = st.checkbox(
                    f"Improve Hydrophobicity (GRAVY ≤ WT: {wt_row.get('gravy', 0):.2f})", value=True
                )
                if has_camsol:
                    improve_camsol = st.checkbox(
                        f"Improve Intrinsic Solubility (CamSol ≥ WT: {wt_row.get('camsol_score', 0):.2f})", value=True
                    )
                else:
                    improve_camsol = False
                reduce_dipole = st.checkbox(
                    f"Reduce/Maintain VH-VL Charge Dipole (≤ WT: {wt_row.get('charge_asymmetry', 0):.2f})", value=False
                )
                
            with f_col3:
                st.markdown("**3. Evolutionary Guardrails**")
                min_ablang = st.slider(
                    "Min AbLang2 Paired ΔLL", 
                    min_value=-10.0, max_value=5.0, value=-1.0, step=0.5,
                    help="Ensures mutations do not violate paired antibody germline rules."
                )
                min_esm = st.slider(
                    "Min ESM-2 ΔLL", 
                    min_value=-150.0, max_value=20.0, value=-10.0, step=1.0
                )

        # Apply the boolean masks
        filtered_df = df.copy()
        if 'pyrosetta_ddg' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['pyrosetta_ddg'] <= max_rosetta]
        if 'janus_ddg' in filtered_df.columns:
            # FIX 1: Keep variants >= min_janus (neutral and positive/stabilizing)
            filtered_df = filtered_df[filtered_df['janus_ddg'] >= min_janus]
        if strict_interface and 'bindpred_kd' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['bindpred_kd'] <= wt_row['bindpred_kd']]
        if improve_solubility and 'gravy' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['gravy'] <= wt_row['gravy']]
        if improve_camsol and has_camsol:
            filtered_df = filtered_df[filtered_df['camsol_score'] >= wt_row['camsol_score']]
        if reduce_dipole and 'charge_asymmetry' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['charge_asymmetry'] <= wt_row['charge_asymmetry']]
        if 'ablang_fitness' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['ablang_fitness'] >= min_ablang]
        if 'esm_fitness' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['esm_fitness'] >= min_esm]

        # Always keep WT in the plot for visual reference
        if 'WT' not in filtered_df['mutations'].values and not wt_rows.empty:
            filtered_df = pd.concat([wt_rows, filtered_df], ignore_index=True)

        st.markdown(f"**🎯 Surviving Anti-Aggregation Candidates:** `{len(filtered_df)}` out of `{len(df)}` total variants (including WT)")

        # --- Interactive 4D Plot Controls (X, Y, Color, Size) ---
        metric_options = [c for c in download_cols if c not in ['id', 'mutations', 'VH', 'VL'] and c in df.columns]
        
        ax_col1, ax_col2, ax_col3, ax_col4 = st.columns(4)
        with ax_col1:
            x_axis = st.selectbox("X-Axis (e.g., Stability)", options=metric_options, index=metric_options.index('pyrosetta_ddg') if 'pyrosetta_ddg' in metric_options else 0)
        with ax_col2:
            y_axis = st.selectbox("Y-Axis (e.g., Antibody PLM)", options=metric_options, index=metric_options.index('ablang_fitness') if 'ablang_fitness' in metric_options else 1)
        with ax_col3:
            default_color = 'camsol_score' if has_camsol else 'gravy'
            color_axis = st.selectbox("Color (e.g., Solubility)", options=metric_options, index=metric_options.index(default_color) if default_color in metric_options else 0)
        with ax_col4:
            size_axis = st.selectbox("Bubble Size", options=["Uniform"] + metric_options, index=0)

        if not filtered_df.empty:
            plot_data = filtered_df.copy()
            
            # Scale bubble size safely to positive numbers if a metric is selected
            if size_axis != "Uniform":
                s_min, s_max = plot_data[size_axis].min(), plot_data[size_axis].max()
                if s_max > s_min:
                    plot_data['_bubble_size'] = 8 + 20 * (plot_data[size_axis] - s_min) / (s_max - s_min)
                else:
                    plot_data['_bubble_size'] = 12
            else:
                plot_data['_bubble_size'] = 10

            fig_scatter = px.scatter(
                plot_data,
                x=x_axis, 
                y=y_axis,
                color=color_axis,
                size='_bubble_size',
                hover_name='id',
                hover_data=['mutations', 'global_composite_score', 'janus_ddg', 'gravy', 'charge_asymmetry'],
                color_continuous_scale='Viridis',
                labels={
                    x_axis: rename_map.get(x_axis, x_axis),
                    y_axis: rename_map.get(y_axis, y_axis),
                    color_axis: rename_map.get(color_axis, color_axis)
                },
                title=f"Filtered Developability Landscape ({len(plot_data)} Candidates)"
            )
            
            # Highlight the exact Wild-Type coordinates with red crosshairs
            wt_x = wt_row.get(x_axis, 0)
            wt_y = wt_row.get(y_axis, 0)
            fig_scatter.add_hline(y=wt_y, line_dash="dash", line_color="#FF4B4B", opacity=0.7, annotation_text="WT Baseline")
            fig_scatter.add_vline(x=wt_x, line_dash="dash", line_color="#FF4B4B", opacity=0.7, annotation_text="WT Baseline")
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Show the shortlisted table of surviving candidates right below the chart
            with st.expander(f"📋 View Shortlisted Anti-Aggregation Candidates ({len(filtered_df)} variants)", expanded=False):
                shortlist_df = filtered_df[download_cols].rename(columns=rename_map)
                st.dataframe(shortlist_df[display_cols], use_container_width=True, hide_index=True)
                st.download_button(
                    label="⬇️ Download Shortlisted Candidates CSV",
                    data=shortlist_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{p2_run_name}_shortlisted_candidates.csv",
                    mime='text/csv'
                )
        else:
            st.warning("No variants passed all filters! Try relaxing one of the sliders above.")

        # ==========================================
        # 3. Multi-Metric Radar Chart (Below Table)
        # ==========================================
        st.markdown("---")
        st.subheader("Developability Footprint")
        
        # Default to comparing Wild-Type against the #1 top surviving mutant
        wt_id = wt_row['id']
        non_wt_survivors = [vid for vid in filtered_df['id'].tolist() if vid != wt_id]
        default_radar_ids = [wt_id] + non_wt_survivors[:1] if non_wt_survivors else [wt_id]
        
        selected_ids = st.multiselect(
            "Select Variants to Compare (Normalized 0 to 1, where 1 = Best in Cohort):", 
            options=df['id'].tolist(), 
            default=default_radar_ids
        )
        
        if selected_ids:
            radar_fig = go.Figure()
            
            # Dynamically build radar categories based on CamSol presence
            categories = [
                'Stability (Janus ΔΔG)', 'Stability (PyRosetta ΔΔG)', 
                'Affinity (Log10 Kd)', 'ESM-2 (ΔLL)', 
                'AntiBERTy (ΔLL)', 'AbLang2 (ΔLL)', 
                'ProtBERT (ΔLL)', 'Hydrophobicity (GRAVY)', 
                'Charge Asymmetry', 'Colloidal (pI)'
            ]
            if has_camsol:
                categories.insert(0, 'Solubility (CamSol)')
    
            for variant_id in selected_ids:
                v_data = df[df['id'] == variant_id].iloc[0]
                
                # FIX 2: Added gravy_norm, charge_asym_norm, and pi_norm to match categories 1-to-1
                values = [
                    v_data.get('janus_norm', 0.5),
                    v_data.get('pyrosetta_norm', 0.5),
                    v_data.get('bindpred_norm', 0.5),
                    v_data.get('esm_norm', 0.5),
                    v_data.get('ab_norm', 0.5),
                    v_data.get('ablang_norm', 0.5),
                    v_data.get('protbert_norm', 0.5),
                    v_data.get('gravy_norm', 0.5),
                    v_data.get('charge_asym_norm', 0.5),
                    v_data.get('pi_norm', 0.5)
                ]
                if has_camsol:
                    values.insert(0, v_data.get('camsol_norm', 0.5))
                
                values.append(values[0]) # Close the polygon
                cat_closed = categories + [categories[0]]
                
                radar_fig.add_trace(go.Scatterpolar(
                    r=values, theta=cat_closed,
                    fill='toself', name=variant_id,
                    hoverinfo="text+name",
                    text=[f"{cat}: {val:.2f}" for cat, val in zip(cat_closed, values)]
                ))
                
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True, legend=dict(orientation="h", y=-0.2),
                margin=dict(l=40, r=40, t=20, b=20), height=500
            )
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("Select a variant to view its developability footprint.")
