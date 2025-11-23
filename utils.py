import streamlit as st
import pandas as pd
import numpy as np

import copy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- Helper Functions from 'Ready to Bet.ipynb' ---
def two_dice_sum():
    d1 = np.random.choice(range(6)) + 1 
    d2 = np.random.choice(range(6)) + 1
    return d1 + d2, d1, d2

def select_horse(value):
    return min(max(value, 3), 11)

@st.cache_data
def run_bet_simulation(num_trials):
    """
    Reruns the logic for the Ready to Bet game simulation.
    
    Args:
        num_trials (int): The number of simulations to run.
    """
    award_his = pd.DataFrame()
    for trail in range(num_trials):
        dice_his = pd.DataFrame()
        
        # Initial status setup
        int_status = [{'horse':3,'extra move':3,'colour':'Blue'},
                      {'horse':4,'extra move':3,'colour':'Blue'},
                      {'horse':5,'extra move':2,'colour':'Orange'},
                      {'horse':6,'extra move':1,'colour':'Red'},
                      {'horse':7,'extra move':0,'colour':'Black'},
                      {'horse':8,'extra move':1,'colour':'red'},
                      {'horse':9,'extra move':2,'colour':'Orange'},
                      {'horse':10,'extra move':3,'colour':'Blue'},
                      {'horse':11,'extra move':3,'colour':'Blue'},
                      ]
        position = pd.DataFrame(int_status)
        position[['current position','last diced']] = 0
    
        while position['current position'].max() < 15:
            dice_sum, d1, d2 = two_dice_sum()
            selected_horse = select_horse(dice_sum)
            selected_horse_inf = position[position['horse'] == selected_horse].iloc[0]
            # Movement logic with bonus: 1 if not last diced, 1 + extra_move if last diced
            move = 1 if selected_horse_inf['last diced'] == 0 else 1 + selected_horse_inf['extra move']
    
            record_dict = [{'selected_horse':selected_horse, 'move':move, 'dice_sum':dice_sum, 'd1':d1, 'd2':d2}]
            dice_his = pd.concat([dice_his, pd.DataFrame(record_dict)], ignore_index=True)
    
            position.loc[position['horse']==selected_horse,'current position'] += move
            position['last diced'] = 0
            position.loc[position['horse']==selected_horse,'last diced'] = 1
    
        position['rank'] = position['current position'].rank(ascending=False, method='min')
        award_dict = {}
    
        for i in range(3, 12):
            award_dict[f'horse {i} rank'] = position[position['horse']==i]['rank'].iloc[0]
        award_dict['rank1_2 diff'] = position['current position'].max() - position['current position'].nlargest(2).min()
        award_dict['rank1_3 diff'] = position['current position'].max() - position['current position'].nlargest(3).min()
        award_dict['min_position'] = position['current position'].min()
        award_dict['rank1_colour'] = position[position['rank'] == 1]['colour'].iloc[0]
        award_dict['horse moved <=3'] = position[position['current position'] <=3].count()['horse']
        award_dict['diced times'] = dice_his.shape[0]
        award_dict['boosted times'] = dice_his[dice_his['move'] > 1].shape[0]
        
        award_his = pd.concat([award_his, pd.DataFrame(award_dict, index=[trail])])

    # Add boolean columns for analysis
    award_his['horse 7 rank>=5'] = award_his['horse 7 rank'] >= 5
    award_his['2horse posit <=3'] = award_his['horse moved <=3'] >= 2
    award_his['min_position>=6'] = award_his['min_position'] >= 6
    award_his['rank1_2 diff <=1'] = award_his['rank1_2 diff'] <= 1
    award_his['rank1_3 diff <=3'] = award_his['rank1_3 diff'] <= 3
    
    return award_his

# --- Helper Functions from 'Camel run.ipynb' ---

@st.cache_data
def run_camel_run_simulation(num_sims):
    """
    Reruns the logic for the Camel Run game simulation.
    
    Args:
        num_sims (int): The number of simulations to run.
    """
    sim_hist = pd.DataFrame()
    for sim in range(num_sims):
        # init condition setup
        df = [['A',0,1],['B',0,2],['C',0,3],['D',0,4],['E',0,5]]
        Camel_list = pd.DataFrame(df,columns=['name','location','order'])
        Camel_list['moved'] = 0
        big_run = 0
        Camel_in_dice = range(len(Camel_list))
        small_game_hist = pd.DataFrame()
        
        while Camel_list['location'].max() <= 16:
            big_run = big_run + 1
            for run_th, Camel_a_run in enumerate(range(len(Camel_list))):
                # random pick A~E
                selected_camel_i = np.random.choice(Camel_in_dice)
                selected_camel = Camel_list.iloc[selected_camel_i]
                # selected, roll dice 1,2,3
                Camel_move = np.random.choice(range(3)) + 1
                
                # update location
                is_stacked = (Camel_list['location'] == selected_camel.location) & (Camel_list['order'] >= selected_camel.order)
                Camel_list.loc[is_stacked, 'location'] += Camel_move
                Camel_list.loc[is_stacked, 'order'] += 10 
                
                # Re-rank the order within each location
                Camel_list['order'] = Camel_list.groupby("location")["order"].rank(method="dense")
                
                # update for next run
                Camel_in_dice = [i for i in Camel_in_dice if i != selected_camel_i ]
                
                # detect if game end
                if Camel_list['location'].max() > 16: break
                
            # a small run record
            small_game_record = Camel_list.copy()
            # Calculate overall rank based on location (major) and order (minor)
            small_game_record['rank'] = (small_game_record['location'] * 10 + small_game_record['order']).rank(method="dense", ascending = False)
            small_game_record['big_run'] = big_run
            small_game_hist = pd.concat([small_game_hist, small_game_record], ignore_index=True)

            # reset the random pick
            Camel_in_dice = range(len(Camel_list))
            
        # simulate record
        max_loc = Camel_list['location'].max()
        winner_filter = Camel_list[Camel_list['location'] == max_loc]
        winner = winner_filter[winner_filter['order'] == winner_filter['order'].max()]['name'].iloc[0]

        min_loc = Camel_list['location'].min()
        loser_filter = Camel_list[Camel_list['location'] == min_loc]
        loser = loser_filter[loser_filter['order'] == loser_filter['order'].min()]['name'].iloc[0]

        if not small_game_hist.empty:
            pv_record = pd.pivot_table(small_game_hist, index='name', columns='rank', aggfunc='count', fill_value=0)['big_run']
        else:
            pv_record = pd.DataFrame(0, index=['A', 'B', 'C', 'D', 'E'], columns=[1, 2, 5]) 
            
        sim_record = {'winner': winner,
                      'last lose': loser,
                      'A first count': pv_record.loc['A', 1] if 1 in pv_record.columns else 0,
                      'B first count': pv_record.loc['B', 1] if 1 in pv_record.columns else 0,
                      'C first count': pv_record.loc['C', 1] if 1 in pv_record.columns else 0,
                      'D first count': pv_record.loc['D', 1] if 1 in pv_record.columns else 0,
                      'E first count': pv_record.loc['E', 1] if 1 in pv_record.columns else 0,
                      'A second count': pv_record.loc['A', 2] if 2 in pv_record.columns else 0,
                      'B second count': pv_record.loc['B', 2] if 2 in pv_record.columns else 0,
                      'C second count': pv_record.loc['C', 2] if 2 in pv_record.columns else 0,
                      'D second count': pv_record.loc['D', 2] if 2 in pv_record.columns else 0,
                      'E second count': pv_record.loc['E', 2] if 2 in pv_record.columns else 0,
                      'A late count': pv_record.loc['A', 5] if 5 in pv_record.columns else 0,
                      'B late count': pv_record.loc['B', 5] if 5 in pv_record.columns else 0,
                      'C late count': pv_record.loc['C', 5] if 5 in pv_record.columns else 0,
                      'D late count': pv_record.loc['D', 5] if 5 in pv_record.columns else 0,
                      'E late count': pv_record.loc['E', 5] if 5 in pv_record.columns else 0,
                      'max run': big_run
                      }
        sim_hist = pd.concat([sim_hist, pd.DataFrame(sim_record, index=[sim])])
        
    return sim_hist, small_game_hist


@st.cache_data
def calculate_smartphone_profits():
    """
    Performs the exhaustive case analysis for the Smartphone Inc. game
    to calculate maximum profit and associated strategies (Price, Tech, Promotion).

    The analysis covers 5 Price levels (0-4), 5 Tech levels (0-4), and 5 Promotion levels (0-4).
    """
    
    # Game parameters based on common interpretation of Smartphone Inc. mechanics
    # Price Levels P: 0, 1, 2, 3, 4
    REVENUE = {0: 10, 1: 13, 2: 16, 3: 19, 4: 22} 
    BASE_COST = 2
    # Tech/Promo Levels T, M: 0, 1, 2, 3, 4
    # Tech/Promo cost is simply the level
    
    all_strategies = []
    
    # Iterate through all combinations of (Price Level, Tech Level, Promotion Level)
    for p_level in range(5): # Price 0-4
        for t_level in range(5): # Tech 0-4
            for m_level in range(5): # Promotion 0-4
                
                # --- Cost Calculation ---
                tech_cost = t_level
                promo_cost = m_level
                cost_per_unit = BASE_COST + tech_cost + promo_cost
                
                # --- Production Capacity ---
                # Capacity increases with Tech level (Base of 3, +1 per Tech level)
                production_capacity = 3 + t_level 
                
                # --- Revenue & Profit Margin ---
                revenue_per_unit = REVENUE[p_level]
                
                # Margin: Revenue - Cost. Cannot be negative for profit calculation.
                margin_per_unit = revenue_per_unit - cost_per_unit
                
                # --- Simplified Demand Model ---
                # Demand is generally higher with lower Price (P), higher Tech (T), and higher Promotion (M).
                # This is a simplified proxy for market share (max 18 units of demand)
                raw_demand_score = (5 - p_level) * 2 + t_level + m_level
                
                # --- Sales ---
                # Sales are capped by the lower of demand or production capacity
                sales = min(raw_demand_score, production_capacity)
                
                # --- Total Profit Calculation ---
                profit = sales * margin_per_unit
                
                # Store results
                all_strategies.append({
                    'Price Level': p_level,
                    'Tech Level': t_level,
                    'Promo Level': m_level,
                    'Revenue': revenue_per_unit,
                    'Cost Per Unit': cost_per_unit,
                    'Capacity': production_capacity,
                    'Sales': sales,
                    'Profit': profit
                })

    results_df = pd.DataFrame(all_strategies)
    return results_df


### function for Smartphone inc
# --- 1. Constants and Setup (Based on Smartphone Inc. Game) ---

# Item Labels (0: Empty/Base, 1: Production, 2: Cheap Production, 3: Price Item, 4: Tech, 5: Promot)
EMPTY_LABEL = 0

# Define the two base Box W and Box B definitions (before transpose/rotation)
# Box W Definitions (Player White)
# W0: [[5, 2, 5], [0, 4, 2]] -> Transposed: [[5, 0], [2, 4], [5, 2]] (3x2)
# W1: [[0, 4, 1], [4, 3, 5]] -> Transposed: [[0, 4], [4, 3], [1, 5]] (3x2)
box_W_defs = [
    np.array([[5, 2, 5], [EMPTY_LABEL, 4, 2]]).transpose(),
    np.array([[EMPTY_LABEL, 4, 1], [4, 3, 5]]).transpose()
]

# Box B Definitions (Player Black)
# B0: [[2, 5, 4], [0, 4, 1]] -> Transposed: [[2, 0], [5, 4], [4, 1]] (3x2)
# B1: [[0, 5, 3], [3, 5, 4]] -> Transposed: [[0, 3], [5, 5], [3, 4]] (3x2)
box_B_defs = [
    np.array([[2, 5, 4], [EMPTY_LABEL, 4, 1]]).transpose(),
    np.array([[EMPTY_LABEL, 5, 3], [3, 5, 4]]).transpose()
]

# Color mapping for visualization (Matplotlib)
# Colors: White (0), Black (1), Gray (2), Red (3), Blue (4), Orange (5)
COLORS = [(1, 1, 1), (0, 0, 0), (0.5, 0.5, 0.5), (1, 0, 0), (0, 0, 1), (1, 0.5, 0)]
LEVELS = [0, 1, 2, 3, 4, 5]
ITEM_COLORMAP = ListedColormap(COLORS, N=len(LEVELS))


# --- 2. Core Box Manipulation Functions ---

def rot90_t(arr, n):
    """Rotates a NumPy array 90 degrees clockwise 'n' times."""
    for _ in range(n):
        arr = np.rot90(arr)
    return arr

def place_box(box, place, i, j):
    """
    Places a small box onto a larger 'place' grid.
    
    Args:
        box (np.array): The small box to place.
        place (np.array): The larger grid (Combina_box, 7x7).
        i (int): Column index (x-axis) for placement.
        j (int): Row index (y-axis) for placement.
        
    Returns:
        tuple: (is_replace_box, combina_box)
            is_replace_box: Grid showing overlaps (1=exposed, 2=replaced/covered).
            combina_box: The combined grid with item values.
    """
    # Initialize overlap check grid with 1s where the existing 'place' has items
    is_replace_box = np.clip(copy.deepcopy(place), 0, 1)
    combina_box = copy.deepcopy(place)
    
    # Define the slice where the box will be placed
    j_slice = slice(j, j + box.shape[0])
    i_slice = slice(i, i + box.shape[1])
    
    # 1. Update is_replace_box for overlap check
    # Add 1 where the *new* box has items (value >= 1)
    is_replace_box[j_slice, i_slice] = (
        is_replace_box[j_slice, i_slice] + np.clip(copy.deepcopy(box), 0, 1)
    )
    # The resulting values are: 0 (empty), 1 (one item), 2 (overlap/replaced)
    
    # 2. Update combina_box with the new box values where they are not empty (>= 1)
    # This represents placing the 'top' box
    mask = box >= 1
    combina_box[j_slice, i_slice][mask] = box[mask]
    
    return is_replace_box, combina_box

def record_box(replace_box, Combina_box, n, W_filp, B_filp, trans_times, i, j, major_b):
    """Calculates product attributes and records placement info."""
    record_dict = {}
    record_dict['n'] = n
    record_dict['replaced'] = (replace_box == 2).sum()
    record_dict['exposed'] = (replace_box == 1).sum()
    
    # Item counts
    record_dict['production items'] = (Combina_box == 1).sum()
    record_dict['cheap production'] = (Combina_box == 2).sum()
    record_dict['price items'] = (Combina_box == 3).sum()
    record_dict['tech'] = (Combina_box == 4).sum()
    record_dict['promot'] = (Combina_box == 5).sum()
    record_dict['appear items'] = (Combina_box > 0).sum()
    
    # Calculated attributes
    # Base Production: Production + Cheap Production + Number of replaced items (since items under the 
    # top box still provide production, but not other bonuses)
    record_dict['base production'] = record_dict['production items'] + record_dict['cheap production'] + record_dict['replaced']
    # Price: 5 (base price) + Price Items - Cheap Production
    record_dict['price'] = 5 + record_dict['price items'] - record_dict['cheap production']
    
    # Placement info
    record_dict['W_side'] = W_filp
    record_dict['B_side'] = B_filp
    record_dict['B_trans_times'] = trans_times
    record_dict['B_placed_i'] = i
    record_dict['B_placed_j'] = j
    record_dict['major_box'] = major_b # 'w' if W is on top, 'b' if B is on top
    
    return record_dict

@st.cache_data
def generate_combinations_data():
    """Generates all valid combinations and their resulting attributes."""
    
    n = 1
    result_data = []
    combina_box_record = {}
    
    # Box W is fixed at (2, 2) on the 7x7 grid.
    # Box B placement coordinates (i, j) range from (0 to 3) and (0 to 4).
    
    # W_filp (0 or 1): which base W box definition is used
    for W_filp in range(2):
        box_W_ues = box_W_defs[W_filp]
        
        # B_filp (0 or 1): which base B box definition is used
        for B_filp in range(2):
            
            # trans_times (0 to 3): 0, 90, 180, 270 degree rotation
            for trans_times in range(4):
                box_B_ues = rot90_t(box_B_defs[B_filp], trans_times)
                
                # i (0 to 3), j (0 to 4): placement coordinates for the B box
                for i in range(4):
                    for j in range(5):
                        
                        # --- Overlap Check (Independent of placement order) ---
                        # We calculate the overlap count once. W is the base grid.
                        Combina_box_temp = np.full((7, 7), EMPTY_LABEL)
                        _, Combina_box_W_base = place_box(box_W_ues, Combina_box_temp, 2, 2)
                        replace_box_check, _ = place_box(box_B_ues, Combina_box_W_base, i, j)
                        
                        overlap_count = (replace_box_check == 2).sum()

                        # The notebook enforces an overlap count (replaced) in [1, 4]
                        if overlap_count not in [1, 2, 3, 4]: 
                            continue # Skip invalid overlap counts (0 or 5+)


                        # --- First Combination: B box on top (Major = 'b') ---
                        
                        # Place W first (bottom layer, centered)
                        Combina_box = np.full((7, 7), EMPTY_LABEL)
                        _, Combina_box_W_base = place_box(box_W_ues, Combina_box, 2, 2)
                        
                        # Place B second (top layer)
                        replace_box_B_top, Combina_box_B_top = place_box(box_B_ues, Combina_box_W_base, i, j)
                        
                        major_b = 'b'
                        record_dict_b = record_box(replace_box_B_top, Combina_box_B_top, n, W_filp, B_filp, trans_times, i, j, major_b)
                        result_data.append(record_dict_b)
                        combina_box_record[n] = Combina_box_B_top
                        n += 1
                        
                        # --- Second Combination: W box on top (Major = 'w') ---

                        # Place B first (bottom layer)
                        Combina_box = np.full((7, 7), EMPTY_LABEL)
                        _, Combina_box_B_base = place_box(box_B_ues, Combina_box, i, j)

                        # Place W second (top layer)
                        replace_box_W_top, Combina_box_W_top = place_box(box_W_ues, Combina_box_B_base, 2, 2)
                        
                        major_b = 'w'
                        record_dict_w = record_box(replace_box_W_top, Combina_box_W_top, n, W_filp, B_filp, trans_times, i, j, major_b)
                        result_data.append(record_dict_w)
                        combina_box_record[n] = Combina_box_W_top
                        n += 1
                        
    result_df = pd.DataFrame(result_data).set_index('n')
    
    # Calculate optimality based on Pareto front
    result_df = calculate_optimality(result_df)
    
    return result_df, combina_box_record

def calculate_optimality(result_df):
    """Adds an 'is optimal' column to the DataFrame based on Pareto efficiency."""
    
    review_df = result_df[['price', 'base production', 'tech', 'promot']].copy()
    result_df['is optimal'] = False
    
    for case_n in review_df.index:
        case_row = review_df.loc[case_n]
        
        # Filter for rows with the same price
        same_price_df = review_df[review_df['price'] == case_row['price']]
        
        # Check if any other combination with the same price strictly dominates this one:
        # Dominance means: (P >= P_self) AND (BP >= BP_self) AND (T >= T_self) AND (Pr >= Pr_self) 
        # AND at least one inequality is STRICTLY better (>).
        
        # Identify dominating candidates (must be better or equal on all axes)
        is_better_or_equal = ((same_price_df['base production'] >= case_row['base production']) &
                              (same_price_df['tech'] >= case_row['tech']) &
                              (same_price_df['promot'] >= case_row['promot']) &
                              (same_price_df.index != case_n)) # Exclude self
        
        # Identify if any of those dominating candidates is strictly better on at least one axis
        is_strictly_better = ((same_price_df['base production'] > case_row['base production']) |
                              (same_price_df['tech'] > case_row['tech']) |
                              (same_price_df['promot'] > case_row['promot']))
        
        # If there is at least one combination that is better or equal on all axes 
        # AND strictly better on at least one, then the current case is dominated (NOT optimal).
        is_dominated = (is_better_or_equal & is_strictly_better).any()
        
        if not is_dominated:
            result_df.loc[case_n, 'is optimal'] = True
            
    return result_df

@st.cache_data
def get_optimal_pareto_front(result_df):
    """
    Filters the DataFrame to return only the unique optimal combinations 
    (those on the Pareto front).
    """
    # Use the pre-calculated 'is optimal' column
    optimal_df = result_df[result_df['is optimal']].copy()
    
    # Drop duplicates based on the four main attributes, keeping the one with the lowest 'n'
    unique_optimal_df = optimal_df.drop_duplicates(
        subset=['price', 'base production', 'tech', 'promot'], 
        keep='first'
    )
    
    # Sort for cleaner presentation
    return unique_optimal_df.sort_values(
        ['price', 'base production', 'tech', 'promot'], 
        ascending=[True, False, False, False]
    )

def plot_box(box_n, combina_box_record):
    """Generates a Matplotlib figure for a specific box combination."""
    if box_n not in combina_box_record:
        return None, f"Combination #{box_n} not found."
        
    box = combina_box_record[box_n]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Display the box
    ax.imshow(box, cmap=ITEM_COLORMAP, vmin=0, vmax=5.9, interpolation='nearest')
    
    # Set titles and labels
    ax.set_title(f"Combination #{box_n}", fontsize=16, fontweight='bold')
    
    # Grid lines
    ax.set_xticks(np.arange(-.5, box.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, box.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    
    # Remove major ticks/labels
    ax.tick_params(which="minor", size=0)
    ax.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False)
    
    # Custom legend for the item types
    legend_handles = [plt.Rectangle((0,0),1,1, fc=color, ec="black") for color in COLORS[1:]]
    legend_labels = ['Production (1)', 'Cheap Production (2)', 'Price Item (3)', 'Tech (4)', 'Promot (5)']
    
    # Place legend below the plot
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=3, title='Item Type (Code)', fontsize='small')
    fig.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust layout to fit legend
    
    return fig, None

def get_item_description():
    """Returns a string describing the item codes."""
    return (
        "**Item Codes:**\n"
        "* 1: **Production**\n"
        "* 2: **Cheap Production** (Gives +1 Prod, Costs -1 Price)\n"
        "* 3: **Price Item** (Gives +1 Price)\n"
        "* 4: **Tech**\n"
        "* 5: **Promot**\n\n"
        "**Calculated Attributes:**\n"
        "* **Price**: 5 + (Price Items) - (Cheap Production)\n"
        "* **Base Production**: (Production Items) + (Cheap Production) + (Replaced Items)\n"
    )