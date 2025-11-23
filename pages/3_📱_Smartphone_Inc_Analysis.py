import streamlit as st
import pandas as pd
import sys
import os

# Add the parent directory to the path to import utils
# This assumes utils.py is in the root directory and this file is in a 'pages' subdirectory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils import (
        generate_combinations_data, 
        get_optimal_pareto_front, 
        plot_box, 
        get_item_description
    )
except ImportError:
    st.error("Could not find 'utils.py'. Make sure it is in the root directory.")
    st.stop()

# --- Initialize Session State for Step Control and Data Storage ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'combina_box_record' not in st.session_state:
    st.session_state.combina_box_record = None
if 'optimal_df' not in st.session_state:
    st.session_state.optimal_df = None
if 'selected_visual_id' not in st.session_state:
    st.session_state.selected_visual_id = None


# --- Step Handlers ---
def set_step(new_step):
    """Function to change the current step in the session state."""
    st.session_state.step = new_step
    
def load_data_and_set_step(new_step):
    """Load data (cached) and transition to the next step."""
    with st.spinner("Generating and calculating all combinations..."):
        # The data is retrieved from cache instantly after the first run
        result_df, combina_box_record = generate_combinations_data()
        st.session_state.result_df = result_df
        st.session_state.combina_box_record = combina_box_record
        st.session_state.optimal_df = get_optimal_pareto_front(result_df)
    st.session_state.step = new_step

# --- Main Page Function ---
def smartphone_inc_analysis_page():
    st.set_page_config(
        page_title="Smartphone Inc. Box Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📱 Smartphone Inc. Box Combination Analyzer")
    st.markdown(
        """
        This tool analyzes all possible combinations of two game boxes (W and B) in 
        the *Smartphone Inc.* board game. Follow the steps below to generate the full
        analysis and identify the optimal product configurations.
        """
    )
    
    st.divider()

    # --- Step 1: Introduction & Start ---
    st.header("Step 1: Overview & Preparation")
    st.info("Click the button to begin the analysis. This step will compute all combinations, rotations, and placements, storing the results in memory (cached).")
    
    st.markdown("**Analysis Scope:**")
    st.markdown(
        """
        - **4** base box definitions (W0, W1, B0, B1).
        - **4** rotations for Box B.
        - **2** placement orders (W on top vs. B on top).
        - **All valid** overlap positions (1 to 4 squares replaced).
        """
    )

    if st.session_state.step == 1:
        if st.button("1. Start Analysis: Generate All Combinations", use_container_width=True, type="primary"):
            load_data_and_set_step(2)
            st.rerun() # Rerun to display Step 2
    
    st.divider()

    # --- Step 2: Generation & Metrics ---
    if st.session_state.step >= 2:
        if st.session_state.result_df is None:
             st.error("Data unavailable. Returning to Step 1.")
             if st.button("Go Back to Step 1", key="back_step2_err"): set_step(1); st.rerun()
             return
             
        result_df = st.session_state.result_df
        optimal_df = st.session_state.optimal_df
        
        total_combinations = len(result_df)
        total_optimal = len(optimal_df)

        st.header("Step 2: Analysis Results & Summary Metrics")

        # --- Metrics Section ---
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Total Unique Combination Shapes", f"{total_combinations // 2}", help="The number of unique ways the two boxes can overlap, regardless of which box is on top.")
        col2.metric("Total Resulting Products", f"{total_combinations}", help="Each unique combination shape results in two products (W on top / B on top).")
        col3.metric("Unique Pareto Optimal Results", f"{total_optimal}", help="The number of non-dominated stat lines.")
        
        st.markdown("---")
        
        if st.session_state.step == 2:
            col_next, col_back = st.columns(2)
            with col_next:
                if st.button("2. Analyze Pareto Optimal Solutions", use_container_width=True, type="primary"):
                    set_step(3)
                    st.rerun()
            with col_back:
                if st.button("Go Back to Step 1: Restart Analysis", use_container_width=True):
                    set_step(1); st.session_state.result_df = None; st.rerun()
        
        st.divider()


    # --- Step 3: Pareto Optimal Analysis ---
    if st.session_state.step >= 3:
        if st.session_state.optimal_df is None:
             st.error("Data unavailable. Returning to Step 1.")
             set_step(1); st.rerun()
             return
             
        optimal_df = st.session_state.optimal_df
        total_optimal = len(optimal_df)

        st.header("Step 3: Pareto Optimal Solutions")
        st.markdown(
            f"These **{total_optimal}** unique combinations represent the best possible stat lines (Base Production, Tech, Promot) "
            "for a given Price level, meaning no other combination with the same price is strictly better."
        )
        
        # Simplify the optimal DF for display
        display_cols = ['price', 'base production', 'tech', 'promot', 'W_side', 'B_side', 'B_trans_times', 'major_box']
        optimal_display_df = optimal_df[display_cols].reset_index()
        optimal_display_df.columns = ['ID', 'Price', 'Production', 'Tech', 'Promot', 'W Box', 'B Box', 'B Rotations', 'Top Box']
        
        st.dataframe(
            optimal_display_df, 
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn("Combination ID", help="ID used for plotting.", format="%d"),
                "B Rotations": st.column_config.TextColumn(help="0=0°, 1=90°, 2=180°, 3=270°")
            }
        )

        if st.session_state.step == 3:
            col_next, col_back = st.columns(2)
            with col_next:
                if st.button("3. Explore Combination Visualizer", use_container_width=True, type="primary"):
                    # Automatically select the first optimal ID for the visualizer
                    if not optimal_display_df.empty:
                        st.session_state.selected_visual_id = optimal_display_df['ID'].iloc[0]
                    set_step(4)
                    st.rerun()
            with col_back:
                if st.button("Go Back to Summary Metrics"): set_step(2); st.rerun()

        st.divider()

    # --- Step 4: Combination Visualizer & Download ---
    if st.session_state.step >= 4:
        if st.session_state.result_df is None:
             st.error("Data unavailable. Returning to Step 1.")
             set_step(1); st.rerun()
             return

        result_df = st.session_state.result_df
        combina_box_record = st.session_state.combina_box_record

        st.header("Step 4: Combination Visualizer & Data Download")
        
        col_sel, col_desc = st.columns([1, 2])
        
        # Determine the initial selection for the selectbox
        all_ids = result_df.index.tolist()
        initial_index = 0
        if 'selected_visual_id' in st.session_state and st.session_state.selected_visual_id in all_ids:
            initial_index = all_ids.index(st.session_state.selected_visual_id)
        elif not optimal_df.empty:
             # Ensure the initial selection is the first optimal result if the state hasn't been set
             initial_id = optimal_df.index[0]
             initial_index = all_ids.index(initial_id)
        
        with col_sel:
            selected_id = st.selectbox(
                "Select Combination ID to Visualize:",
                options=all_ids,
                index=initial_index,
                key='visualizer_selectbox', # Use a key to maintain state
                help="Choose an ID from the full analysis (1 to 1184)."
            )

        with col_desc:
            st.subheader("Item Key & Formulae")
            st.markdown(get_item_description())

        if selected_id:
            fig, error = plot_box(selected_id, combina_box_record)
            
            if fig:
                st.subheader(f"Visual Grid for Combination #{selected_id}")
                st.pyplot(fig, use_container_width=False)
                
                # Display detailed attributes below the plot
                st.subheader("Detailed Attributes")
                # FIX: Corrected column names. 'n' is the index, 'W_placed_i/j' do not exist.
                # 'B_placed_i/j' are the correct names for the B box coordinates.
                attributes = result_df.loc[selected_id].drop(
                    ['B_placed_i', 'B_placed_j', 'appear items', 'is optimal']
                ).to_frame().T
                st.dataframe(attributes, use_container_width=True)

            elif error:
                st.error(error)

        st.markdown("---")
        
        # --- Full Data Download ---
        st.subheader("Full Data Analysis Download")
        st.markdown("Download the complete dataset of all calculated combinations for further analysis.")
        
        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        csv = convert_df(result_df.reset_index())

        st.download_button(
            label="Download All Combinations Data (.csv)",
            data=csv,
            file_name='smartphone_inc_box_analysis_full.csv',
            mime='text/csv',
        )
        
        st.markdown("---")
        
        if st.button("Go Back to Step 3: Optimal Solutions", use_container_width=True): 
            set_step(3); st.rerun()

if __name__ == "__main__":
    smartphone_inc_analysis_page()