import streamlit as st
import pandas as pd
import numpy as np

# Note: The simulation functions are now in utils.py and imported by the page files.

st.set_page_config(
    # Updated the browser tab title and hopefully the sidebar label for the root page
    page_title="Homepage", 
    layout="wide", 
)

st.title("🎲 Game Strategy Analysis Dashboard")
st.markdown("""
    Welcome! This application hosts **Monte Carlo simulation results** for two distinct board game mechanics. 
    Each analysis is powered by **10,000 randomized trials** to provide robust, data-driven insights into strategic betting and probability distributions.
""")

st.divider()

# Use columns for a clear, side-by-side presentation of the two analysis types
col1, col2, col3 = st.columns(3)

with col1:
    # Direct link to the page
    st.page_link("pages/1_🐪_Camel_Run_Analysis.py", label="Go to Camel Run Results", icon="🐪")

    st.subheader("1. 🐪 Camel Run Analysis")
    st.markdown("""
        ### **Game Type: Stack-Based Racing**
        
        This simulation mirrors a racing game where movement is complicated by **stacking mechanics** (camels carry other camels). The initial order and dice rolls create a highly dynamic environment.
        
        **Key Strategic Insights:**
        * **Final Outcomes:** Probability distribution for the eventual winner and the camel in last place.
        * **Intermediate Bets:** Frequency analysis of which camel holds the 1st, 2nd, or 5th rank at various points in the game—crucial for intermediate betting strategies.
    """)
    

with col2:
    # Direct link to the page
    st.page_link("pages/2_🎲_Ready_to_Bet_Analysis.py", label="Go to Ready to Bet Results", icon="🎲")

    st.subheader("2. 🎲 Ready to Bet Analysis")
    # Adding a placeholder image to illustrate the theme
    st.image("./pic/Ready Set bet - Title.jpeg")
    
    st.markdown("""
        ### **Game Type: Dice & Consecutive Bonus**
        
        This simulation models movement driven by the sum of two dice (houses 3 through 11). The critical mechanism is the **consecutive move bonus**, which drastically increases movement for houses rolled back-to-back.
        """)
    
    st.image("./pic/Ready Set bet - Race.jpeg")

    st.markdown("""
        **Key Strategic Insights:**
        * **Winning Odds:** Precise rank probability (1st, 2nd, 3rd) for each house, showing how the bonus shifts the expected distribution from a standard bell curve.
        * **System Metrics:** Analysis of average game length and the frequency of bonus (boosted) moves.
    """)
    



with col3:
    # Direct link to the page
    st.page_link("pages/3_📱_Smartphone_Inc_Analysis.py", label="Go to Smartphone Inc. Strategy", icon="📱")

    st.subheader("3. 📱 Smartphone Inc. Analysis")
    st.image("https://placehold.co/600x200/0F766E/FFFFFF?text=Strategy+Optimization", caption="Illustration of Strategic Decision Making")
    st.markdown("""
        ### **Game Type: Economic Strategy (Case Study)**
        
        This is an **exhaustive case analysis** to determine the optimal strategy for maximizing profit in the Smartphone Inc. game. It explores all combinations of Price, Technology, and Promotion tiers.
        
        **Key Strategic Insights:**
        * **Max Profit Strategy:** Identifies the exact combination of P, T, and M that yields the highest total profit.
        * **Cost/Revenue Breakdown:** Visualizes the cost-to-revenue dynamics for all 80 possible scenarios.
    """)
    
st.divider()

st.info("Navigate using the links above or the sidebar to dive into the data tables and charts.")