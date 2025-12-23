import streamlit as st
import pandas as pd
import numpy as np

# Note: The simulation functions are now in utils.py and imported by the page files.

def image_opening(link):

    """dueling different working path on streamlit cloud"""

    try:
        st.image(f"{link}",width=250)
    except:
        st.image(f"Gambling_simulate{link}",width=250)


st.set_page_config(
    # Updated the browser tab title and hopefully the sidebar label for the root page
    page_title="Homepage", 
    layout="wide", 
)

st.title("🎲 Game Strategy Analysis Dashboard")
st.markdown("""
    Welcome! This application hosts **Monte Carlo simulation results** , providing data-driven insights into strategic betting and probability distributions.
""")

st.divider()

# Use columns for a clear, side-by-side presentation of the two analysis types
col1, col2 = st.columns(2)

with col1:
    # Direct link to the page
    st.page_link("pages/1_🐪_Camel_Run_Analysis.py", label="Go to Camel Run Results", icon="🐪")

    st.subheader("1. 🐪 Camel Run Analysis")
    image_opening("pic/Camel Run - Title.jpg")

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
    image_opening("./pic/Ready Set bet - Title.jpeg")
    
    st.markdown("""
        ### **Game Type: Dice & Consecutive Bonus**
        
        This simulation models movement driven by the sum of two dice (houses 3 through 11). The critical mechanism is the **consecutive move bonus**, which drastically increases movement for houses rolled back-to-back.
        """)
    
    image_opening("./pic/Ready Set bet - Race.jpeg")

    st.markdown("""
        **Key Strategic Insights:**
        * **Winning Odds:** Precise rank probability (1st, 2nd, 3rd) for each house, showing how the bonus shifts the expected distribution from a standard bell curve.
        * **System Metrics:** Analysis of average game length and the frequency of bonus (boosted) moves.
    """)
    

    
st.divider()

st.info("Navigate using the links above or the sidebar to dive into the data tables and charts.")