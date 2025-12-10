import streamlit as st
import pandas as pd
import numpy as np
from utils import run_camel_run_simulation # Import the function from the utility file

# --- Initialization ---
# Initialize session state variables for step tracking and data storage
if 'camel_step' not in st.session_state:
    st.session_state.camel_step = 1
if 'camel_sim_hist' not in st.session_state:
    st.session_state.camel_sim_hist = None
if 'camel_small_game_hist' not in st.session_state:
    st.session_state.camel_small_game_hist = None
if 'camel_num_sims' not in st.session_state:
    st.session_state.camel_num_sims = 1000

# --- Navigation Callbacks ---
def set_step(step):
    st.session_state.camel_step = step

def run_simulation_camel():
    # Only run if a valid number is provided
    num_sims = int(st.session_state.get('num_sims_input_camel', 1000))
    if num_sims < 10:
        st.error("Please enter at least 10 simulations.")
        return

    st.session_state.camel_num_sims = num_sims
    
    # 1. Initialize the progress bar with a custom text
    # The text will update live if run_camel_run_simulation updates the bar.
    progress_bar = st.progress(0, text=f"Simulation 0 of {num_sims:,} running...")

    # NOTE: To see the live simulation number, you MUST update the 
    # 'run_camel_run_simulation' function (in utils.py) to accept 'progress_bar' 
    # and call 'progress_bar.progress((i + 1) / num_sims, text=f"Simulation {i+1:,} of {num_sims:,} running...")'
    # inside its simulation loop.
    
    sim_hist, small_game_hist = run_camel_run_simulation(st.session_state.camel_num_sims, progress_bar)
    
    progress_bar.empty() # Clear the progress bar after completion

    st.session_state.camel_sim_hist = sim_hist
    st.session_state.camel_small_game_hist = small_game_hist
    set_step(2)
    st.rerun()

# --- Page Content ---
st.title("🐪 Camel Run Game Simulation Analysis (Interactive)")
st.markdown("Use the steps below to run a simulation and analyze the results progressively.")
st.divider()

# --- Step 1: Configuration and Simulation Run ---
st.subheader("Step 1: Configure & Run Simulation")
if st.session_state.camel_step >= 1:
    st.markdown("""
    **Action:** Define the number of simulation trials for the Monte Carlo analysis and execute the run.
    The simulation models 5 camels racing where they can stack and move together.
    """)
    
    # Input for number of simulations
    st.number_input(
        "Number of Simulations:",
        min_value=10,
        max_value=100000,
        value=st.session_state.camel_num_sims,
        key='num_sims_input_camel',
        step=1000,
        help="A higher number provides more accurate probability distribution."
    )
    
    if st.button("Run Simulation", key='btn_run_camel', type='primary'):
        run_simulation_camel()
    
    st.caption("👈 Click the button above to start the analysis.")

st.divider()

# --- Step 2: Display Overall Simulation History ---
st.subheader("Step 2: Review Simulation History")
if st.session_state.camel_step >= 2:
    st.success(f"Simulation Complete: {st.session_state.camel_num_sims:,} trials run.")
    st.markdown(f"**Result:** Displaying the first few rows of the simulation summary history (raw outcomes for each trial).")
    st.dataframe(st.session_state.camel_sim_hist.head())
    
    if st.button("Continue to Final Outcomes Analysis", on_click=set_step, args=[3], key='btn_step2_camel'):
        st.rerun()

st.divider()

# --- Step 3: Display Final Game Outcomes (Winner/Loser) ---
st.subheader("Step 3: Analyze Final Outcomes")
if st.session_state.camel_step >= 3:
    st.markdown("**Result:** Visualizing the frequency distribution for the winning and losing camels across all simulations.")
    sim_hist = st.session_state.camel_sim_hist
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("Winner Counts")
        winner_counts = sim_hist.groupby('winner').size().rename('Counts')
        st.dataframe(winner_counts)
        st.bar_chart(winner_counts)

    with col2:
        st.caption("Last Lose Counts")
        loser_counts = sim_hist.groupby('last lose').size().rename('Counts')
        st.dataframe(loser_counts)
        st.bar_chart(loser_counts)
        
    if st.button("Continue to In-Game Rank Statistics", on_click=set_step, args=[4], key='btn_step3_camel'):
        st.rerun()

st.divider()

# --- Step 4: Display Rank Statistics (Mean/Std Dev) ---
st.subheader("Step 4: Analyze Rank Position Averages")
if st.session_state.camel_step >= 4:
    st.markdown("**Result:** Examining the average number of times each camel held the 1st, 2nd, or 5th rank during the intermediate 'runs' of the game.")
    sim_hist = st.session_state.camel_sim_hist
    
    rank_stats = sim_hist.describe().loc[['mean', 'std']]
    rank_columns = [col for col in rank_stats.columns if 'count' in col]
    st.dataframe(rank_stats[rank_columns])

    st.caption("Average Times in Rank Position (Mean)")
    # Filter for rank columns and calculate means
    mean_rank_data = rank_stats.loc['mean', rank_columns].T
    # Clean column names for better chart display
    mean_rank_data.index = mean_rank_data.index.str.replace(' count', '').str.replace(' first', ' Rank 1').str.replace(' second', ' Rank 2').str.replace(' late', ' Rank 5')
    st.bar_chart(mean_rank_data)
        
    if st.button("Continue to In-Game Rank Frequency", on_click=set_step, args=[5], key='btn_step4_camel'):
        st.rerun()

st.divider()

# --- Step 5: Display In-Game Rank Frequency (Pivot Table) ---
st.subheader("Step 5: Review Full Rank Frequency")
if st.session_state.camel_step >= 5:
    st.markdown("**Result:** A pivot table showing the total frequency of each camel occupying a specific rank across all recorded intermediate 'runs' in the simulation.")
    small_game_hist = st.session_state.camel_small_game_hist
    
    # Recalculate the pivot table for display
    if not small_game_hist.empty:
        rank_frequency = pd.pivot_table(small_game_hist, index='name', columns='rank', aggfunc='count', fill_value=0)['big_run']
        st.dataframe(rank_frequency)
        st.bar_chart(rank_frequency.T)
    else:
        st.warning("No intermediate game history was recorded.")

    if st.button("Start New Simulation", on_click=set_step, args=[1], key='btn_step5_camel'):
        st.success("Analysis complete. Resetting to Step 1.")
        st.rerun()