import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
from matplotlib import cm
from highlight_text import fig_text, ax_text
from matplotlib.colors import LinearSegmentedColormap, NoNorm
from matplotlib import cm
import matplotlib.gridspec as gridspec
import numpy as np
from mplsoccer import PyPizza, add_image, FontManager
from mplsoccer import Pitch, VerticalPitch
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
from PIL import Image
from scipy import stats
from scipy.spatial import ConvexHull
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.patheffects import withStroke
from scipy.spatial import ConvexHull, Voronoi, Delaunay
from scipy.ndimage import convolve1d
from matplotlib.animation import FuncAnimation
from matplotlib.patheffects import withStroke


MS_DT = 0.04 # Metrica Sports delta time
MS_LAG_SMOOTH = 20 # n° samples as smoothing windows
PLAYER_MAX_SPEED = 12 #[m/s]

WALK_JOG_THRESHOLD = 2 # Speed threshold for walking
JOG_RUN_THRESHOLD = 4 # Speed threshold for jogging
RUN_SPRINT_THRESHOLD = 7 # Speed threshold for sprint

SPRINTS_WINDOW = 1/MS_DT



def clean_columns_metrica(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and formats column names for Metrica Sports tracking data.

    Steps:
    -------
    1. Removes the first row.
    2. Sets the second row as the new column headers.
    3. Drops the second row after renaming.
    4. Modifies column names: 
       - Adds 'x' and 'y' suffixes for player and ball columns.
       - Keeps other column names unchanged.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw tracking data with multi-row headers.

    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame with properly formatted column names.
    """

    # Remove the first row and reset index
    df = df.iloc[1:].reset_index(drop=True)

    # Set new column headers from the first row and remove it
    df.columns = df.iloc[0]

    df = df.iloc[1:].reset_index(drop=True)

    # Generate new column names
    new_columns = []
    for col in df.columns:
        if pd.notna(col) and (col.startswith('Player') or col == 'Ball'):
            new_columns.extend([f"{col}x", f"{col}y"])  # Add x & y suffixes
        elif pd.notna(col):
            new_columns.append(col)  # Keep other column names unchanged

    # Assign modified column names
    df.columns = new_columns

    return df


def player_possession(frame_data: pd.DataFrame) -> pd.DataFrame:
    """
    Determines which player has possession of the ball in a given frame.

    Steps:
    -------
    1. Extracts ball coordinates.
    2. Computes the Euclidean distance between each player and the ball.
    3. Assigns possession to the closest player if within a set threshold.
    4. Returns updated DataFrame with a `has_possession` column.

    Parameters:
    -----------
    frame_data : pd.DataFrame
        DataFrame containing tracking data for a single frame with columns ['team', 'player', 'x', 'y'].

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with a new boolean column `has_possession`, indicating ball possession.
        For each frame, there will be at max only one player with possession.
    """

    # Extract ball position
    ball_data = frame_data[frame_data['team'] == 'ball']
    if ball_data.empty:
        return frame_data.assign(has_possession=False)

    ball_x, ball_y = ball_data[['x', 'y']].values[0]

    # Compute distances of all players from the ball
    players_data = frame_data[frame_data['player'] != 0].copy()
    players_data['distance'] = np.hypot(players_data['x'] - ball_x, players_data['y'] - ball_y)

    # Identify the closest player within the threshold distance
    MIN_THRESHOLD_DISTANCE = 1.0  # Minimum distance for possession (meters)
    players_data['has_possession'] = False

    min_distance_index = players_data['distance'].idxmin()
    if players_data.loc[min_distance_index, 'distance'] <= MIN_THRESHOLD_DISTANCE:
        players_data.at[min_distance_index, 'has_possession'] = True

    # Merge ball and player data, ensuring correct index order
    frame_data = pd.concat([ball_data, players_data]).sort_index()
    frame_data['has_possession'] = frame_data['has_possession'].fillna(False)

    return frame_data


def calculate_team_stats(team_df: pd.DataFrame, team_name: str) -> tuple:
    """
    Calculates team statistics such as pitch width held and defensive line height for a frame

    Steps:
    -------
    1. Excludes goalkeepers and get outfield player data.
    2. Computes pitch width as the difference between the maximum and minimum y-coordinates.
    3. Determines the defensive line based on the last outfield player’s x-position.
    4. Calculates the distance between the goalkeeper and the last defender as defensive line height

    Parameters:
    -----------
    team_df : pd.DataFrame
        DataFrame containing player positions.
    team_name : str
        Name of the team ('home' or 'away').

    Returns:
    --------
    tuple
        (pitch width, last defender's x-coordinate, distance between goalkeeper and last defender).
    """

    # Define goalkeeper IDs (assuming home = 11, away = 25)
    keeper_id = 11 if team_name == 'home' else 25

    # Separate goalkeeper data and exclude from team_df
    keeper_df = team_df[team_df['player'] == keeper_id]
    team_df = team_df[team_df['player'] != keeper_id]

    # Compute pitch width (distance between farthest players in y-axis)
    width = team_df['y'].max() - team_df['y'].min()

    # Determine the last defender's x-coordinate
    last_defender_x = team_df.sort_values(by='x', ascending=(team_name == 'home')).iloc[-1]['x']

    # Compute distance between goalkeeper and last defender
    def_len = abs(keeper_df['x'].values[0] - last_defender_x)

    return width, last_defender_x, def_len


def team_possession(frame: int, possession_df: pd.DataFrame) -> str:
    """
    Determines which team has possession of the ball in a given frame.

    Steps:
    -------
    1. Checks if the frame falls within any possession interval.
    2. If found, returns the corresponding team in possession.
    3. If the frame matches any `poss_end`, possession is "Neutral".
    4. Otherwise, possession is "Neutral".

    Parameters:
    -----------
    frame : int
        The current frame number.
    possession_df : pd.DataFrame
        DataFrame containing possession intervals with columns ['poss_start', 'poss_end', 'Team'].

    Returns:
    --------
    str
        The team in possession ('home', 'away', or 'Neutral').
    """

    # Check if the frame is within any possession interval
    in_possession = possession_df[(possession_df['poss_start'] <= frame) & (frame < possession_df['poss_end'])]

    if not in_possession.empty:
        return in_possession['Team'].values[0]  # Return the team in possession

    # Check if the frame is exactly at a possession end
    if frame in possession_df['poss_end'].values:
        return "Neutral"

    return "Neutral"  # Default case if no match


def generate_possession_frames(possession_df: pd.DataFrame) -> list:
    """
    Generates a list of frames during which possession is held for home/away team.
    Parameters:
    -----------
    possession_df : pd.DataFrame
        DataFrame containing possession intervals with columns ['poss_start', 'poss_end'] for a particular team.

    Returns:
    --------
    list
        List of frames during which possession is held.
    """
    possession_frames = []

    # Iterate over each row in the possession DataFrame
    for _, row in possession_df.iterrows():
        # Generate frames for the current possession interval
        frames = list(range(row['poss_start'], row['poss_end']))
        # Extend the main list with the generated frames
        possession_frames.extend(frames)

    return possession_frames


def find_stats_home(df: pd.DataFrame, gk: int = 11) -> tuple:
    """
    Finds the average defensive and attacking line positions for the home team.

    Steps:
    -------
    1. Filters out the goalkeeper data.
    2. Finds the lowest and highest x-coordinates per frame. Lowest means max as home team always 
       attack from right to left
    3. Computes the average of these coordinates across all frames.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing player positions.
    gk : int, optional
        Goalkeeper ID, by default 11.

    Returns:
    --------
    tuple
        (average lowest x-coordinate, average highest x-coordinate).
    """
    df["player"] = df["player"].astype(int)
    filtered_df = df[df['player'] != gk]

    # Find the lowest and highest x-coordinates per frame
    lowest_x_per_frame = filtered_df.loc[filtered_df.groupby('Frame')['x'].idxmax()]
    highest_x_per_frame = filtered_df.loc[filtered_df.groupby('Frame')['x'].idxmin()]

    # Compute the averages
    avg_lowest_x = lowest_x_per_frame['x'].mean()
    avg_highest_x = highest_x_per_frame['x'].mean()

    return avg_lowest_x, avg_highest_x


def find_stats_away(df: pd.DataFrame, gk: int = 25) -> tuple:
    """
    Finds the average defensive and attacking line positions for the away team.

    Steps:
    -------
    1. Filters out the goalkeeper data.
    2. Finds the lowest and highest x-coordinates per frame.
    3. Computes the average of these coordinates across all frames.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing player positions.
    gk : int, optional
        Goalkeeper ID, by default 25.

    Returns:
    --------
    tuple
        (average lowest x-coordinate, average highest x-coordinate).
    """
    df["player"] = df["player"].astype(int)
    filtered_df = df[df['player'] != gk]

    # Find the lowest and highest x-coordinates per frame
    lowest_x_per_frame = filtered_df.loc[filtered_df.groupby('Frame')['x'].idxmin()]
    highest_x_per_frame = filtered_df.loc[filtered_df.groupby('Frame')['x'].idxmax()]

    # Compute the averages
    avg_lowest_x = lowest_x_per_frame['x'].mean()
    avg_highest_x = highest_x_per_frame['x'].mean()

    return avg_lowest_x, avg_highest_x


def create_stat_table(df: pd.DataFrame) -> None:
    """
    Draws a table of the stats from the given DataFrame and saves it as a PNG file.

    Steps:
    -------
    1. Plots a table with the DataFrame data.
    2. Customizes the table's appearance (background color, font color, font size).
    3. Saves the table as a PNG file.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the statistics to be displayed in the table.
        Here, it will be the team stats for ON/OFF possession.

    Returns:
    --------
    None
    """
    fig, ax = plt.subplots(figsize=(3.5, 0.8))  # Adjust figure size as needed
    ax.axis('off')  # Hide the axes
    fig.set_facecolor("#dedede")

    # Create a table and add it to the plot
    table = ax.table(
        cellText=df.values,  # Data from the DataFrame
        colLabels=df.columns,  # Column headers
        loc='center',  # Position the table in the center
        cellLoc='center'  # Center-align the text in cells
    )
    
    # Change background and font colors
    for key, cell in table.get_celld().items():
        cell.set_facecolor("#dedede")  # Set background color
        cell.set_text_props(color='black', fontname='Century Gothic') 

    table.auto_set_font_size(False)
    table.set_fontsize(12)  # Set font size
    table.scale(1.2, 1.2)  # Scale table size

    # Save the table as a PNG file
    plt.savefig('images/table.png', bbox_inches='tight', dpi=300) 
    plt.close()



























































































































