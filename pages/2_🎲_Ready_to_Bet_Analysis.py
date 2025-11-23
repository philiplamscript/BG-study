import streamlit as st
import pandas as pd
import numpy as np
from utils import run_bet_simulation # Import the function from the utility file

# --- Initialization ---
# Initialize session state variables for step tracking and data storage
if 'bet_step' not in st.session_state:
    st.session_state.bet_step = 1
if 'award_his' not in st.session_state:
    st.session_state.award_his = None
if 'bet_num_sims' not in st.session_state:
    st.session_state.bet_num_sims = 10000

# --- Navigation Callbacks ---
def set_step_bet(step):
    st.session_state.bet_step = step

def run_simulation_bet():
    # Only run if a valid number is provided
    num_sims = int(st.session_state.get('num_sims_input_bet', 10000))
    if num_sims < 10:
        st.error("Please enter at least 10 simulations.")
        return
        
    st.session_state.bet_num_sims = num_sims
    with st.spinner(f"Running {st.session_state.bet_num_sims:,} simulations... This may take a moment."):
        award_his = run_bet_simulation(st.session_state.bet_num_sims)
    st.session_state.award_his = award_his
    set_step_bet(2)     
    st.rerun()

# --- Page Content ---
st.title("🎲 Ready to Bet Game Simulation Analysis (Interactive)")
st.markdown("Use the steps below to run a simulation and analyze the results progressively.")
st.divider()

# --- Step 1: Configuration and Simulation Run ---
st.subheader("Step 1: Configure & Run Simulation")
if st.session_state.bet_step >= 1:
    st.markdown("""
    **Action:** Define the number of simulation trials for the Monte Carlo analysis and execute the run.
    This simulation models "horses" moving based on two dice rolls with a crucial **consecutive move bonus** mechanic.
    """)
    
    # Input for number of simulations
    st.number_input(
        "Number of Simulations:",
        min_value=10,
        max_value=100000,
        value=st.session_state.bet_num_sims,
        key='num_sims_input_bet',
        step=1000,
        help="A higher number provides more accurate probability distribution."
    )
    
    if st.button("Run Simulation", key='btn_run_bet', type='primary'):
        run_simulation_bet()
    
    st.caption("👈 Click the button above to start the analysis.")

st.divider()

# --- Step 2: Display Overall Simulation History ---
st.subheader("Step 2: Review Simulation History")
if st.session_state.bet_step >= 2:
    st.success(f"Simulation Complete: {st.session_state.bet_num_sims:,} trials run.")
    st.markdown(f"**Result:** Displaying the first few rows of the simulation summary history (raw outcomes for each trial).")
    st.dataframe(st.session_state.award_his)
    
    if st.button("Continue to Rank Probabilities", on_click=set_step_bet, args=[3], key='btn_step2_bet'):
        st.rerun()

st.divider()

# --- Step 3: Display Rank Probability per Horse ---
st.subheader("Step 3: Analyze Rank Probability per Horse")
if st.session_state.bet_step >= 3:
    st.markdown("""
    **Result:** Calculating the probability of each 'horse' (dice sum) finishing in 1st, 2nd, or 3rd place, 
    ordered by horse number (3 through 11). The chart shows the combined probability of ranking 1st, 2nd, or 3rd.
    """)
    award_his = st.session_state.award_his
    
    # 1. Get the columns in order (horse 3 to 11)
    horse_columns = [f'horse {i} rank' for i in range(3, 12)]
    review_rank = award_his[horse_columns]
    
    # 2. Calculate cumulative probabilities
    # Note: .mean() calculates the proportion/probability directly
    rank1_prob = (review_rank == 1).mean().rename('Rank 1 Prob')
    rank1_2_prob = (review_rank <= 2).mean().rename('Rank 1-2 Prob')
    rank1_3_prob = (review_rank <= 3).mean().rename('Rank 1-3 Prob')

    # 3. Calculate incremental probabilities (as requested)
    rank2_prob = (rank1_2_prob - rank1_prob).rename('Rank 2 Prob')
    rank3_prob = (rank1_3_prob - rank1_2_prob).rename('Rank 3 Prob')

    # 4. Create the final DataFrame for the stacked chart, ordered by horse number (default index order)
    prob_df_stacked = pd.DataFrame({
        'Rank 1 Probability': rank1_prob,
        'Rank 2 Probability': rank2_prob,
        'Rank 3 Probability': rank3_prob,
        'Rank 1~2 Probability': rank1_2_prob,
        'Rank 1~3 Probability': rank1_3_prob,
        
    })
    
    # Clean up the index names (e.g., 'horse 3 rank' -> 'Horse 3')
    prob_df_stacked.index = prob_df_stacked.index.str.replace(' horse ', 'Horse ').str.replace(' rank', '')
    
    # Display the stacked probability table (showing the splits)
    st.dataframe(prob_df_stacked[['Rank 1 Probability','Rank 1~2 Probability','Rank 1~3 Probability']]) 
    
    st.caption("Stacked Probability Chart (1st, 2nd, and 3rd Place Odds)")
    # 5. Chart: Streamlit's bar_chart automatically stacks columns by default.
    st.bar_chart(prob_df_stacked[['Rank 1 Probability','Rank 2 Probability','Rank 3 Probability']])
    
    if st.button("Continue to System Metrics", on_click=set_step_bet, args=[4], key='btn_step3_bet'):
        st.rerun()

st.divider()

# --- Step 4: Display Average System Metrics ---
st.subheader("Step 4: Analyze Average System Metrics")
if st.session_state.bet_step >= 4:
    st.markdown("**Result:** Reviewing statistics on overall game length, bonus frequency, and the winning horse color distribution.")
    award_his = st.session_state.award_his
    
    # FIX: Calculate mean ONLY on numeric columns (which excludes rank1_colour)
    # Then, drop the other rank-related numeric columns that we don't need for the metrics display.
    mean_info = award_his.mean(numeric_only=True).drop(
        [col for col in award_his.columns if 'rank' in col and col in award_his.mean(numeric_only=True).index] 
    ).sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("Game Length and Dice Rolls")
        st.metric("Average Diced Times (Game Length)", f"{mean_info['diced times']:.2f}")
        st.metric("Average Boosted Times (Move > 1)", f"{mean_info['boosted times']:.2f}")
        
        # Displaying the raw numeric mean information for completeness
        st.dataframe(mean_info.to_frame(name="Average Value"))

    with col2:
        st.caption("Winning Horse Color Distribution")
        # Use .value_counts() on the categorical column for direct counts
        rank1_color_counts = award_his['rank1_colour'].value_counts().rename('Counts').sort_values(ascending=False)
        st.dataframe(rank1_color_counts)
        st.bar_chart(rank1_color_counts)

    if st.button("Start New Simulation", on_click=set_step_bet, args=[1], key='btn_step4_bet'):
        st.success("Analysis complete. Resetting to Step 1.")
        st.rerun()