from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
from highlight_text import fig_text
from mplsoccer import Pitch
from scipy.spatial import Delaunay
import matplotlib.patheffects as path_effects 
from scipy.spatial import ConvexHull 
from mplsoccer import add_image
from scipy.ndimage import convolve1d


from .helpers import clean_columns_metrica, player_possession, calculate_team_stats, team_possession,generate_possession_frames,find_stats_home,find_stats_away,create_stat_table

from .plotters import plot_base_viz,plot_convexhull_viz,plot_voronoi_viz,plot_delaunay_viz


# Constants
MS_DT = 0.04  # Metrica Sports delta time
MS_LAG_SMOOTH = 20  # n° samples as smoothing windows 
PLAYER_MAX_SPEED = 12  # [m/s]
WALK_JOG_THRESHOLD = 2  # Speed threshold for walking
JOG_RUN_THRESHOLD = 4  # Speed threshold for jogging 
RUN_SPRINT_THRESHOLD = 7  # Speed threshold for sprint
SPRINTS_WINDOW = int(1/MS_DT)
SPRINTS_WINDOW = int(SPRINTS_WINDOW) # Window size for sprints calculation

# Function to format the tracking data
def get_tidy_data(
    home_team_file: str, 
    away_team_file: str, 
    provider: str = "Metrica",
    add_velocity: bool = True,
    pitch_long: int = 100, 
    pitch_width: int = 100
    ) -> Optional[pd.DataFrame]:
    """
    Clean and process tracking data from provider files.

    Steps:
    -------
    1. Validate the provider.
    2. Process home team data.
    3. Process away team data.
    4. Combine home and away team data.
    5. Transform data to row format.
    6. Process ball data.
    7. Combine player and ball data.
    8. Transform coordinates for the second half.
    9. Add velocity data if requested.
    10. Convert time to minutes.

    Args:
        home_team_file: Path to home team data
        away_team_file: Path to away team data
        provider: Data provider name
        add_velocity: Whether to calculate velocity
        pitch_long: Pitch length
        pitch_width: Pitch width
        
    Returns:
        Processed DataFrame or None if provider not supported
    """
    if provider != "Metrica":
        print("Currently only the data format of Metrica Sports provider is supported")
        return None

    # Process home team data
    track_home = pd.read_csv(home_team_file).pipe(clean_columns_metrica).astype(float)
    track_home["team"] = "home"

    # Process away team data  
    track_away = pd.read_csv(away_team_file).pipe(clean_columns_metrica).astype(float)
    track_away["team"] = "away"

    # Combine data
    track_data_long = pd.concat([track_home, track_away], ignore_index=True)
    track_data_long = track_data_long.rename(columns={
        "Time [s]": "time",
        "Ballx": 'xBall', 
        "Bally": 'yBall'
    })
    track_data_long['second'] = np.floor(track_data_long['time'])

    # Transform to row format by iterating over player columns
    new_df = pd.DataFrame()
    for col in track_data_long.columns:
        if col.endswith('x'):
            player_num = col.replace('Player', '').replace('x', '')
            player_y_col = f'Player{player_num}y'
            
            temp_df = track_data_long[['Period', 'Frame', 'time', col, player_y_col, 'team', 'second']].copy()
            temp_df['player'] = player_num
            temp_df = temp_df.rename(columns={col: 'x', player_y_col: 'y'})
            new_df = pd.concat([new_df, temp_df], ignore_index=True)

    # Process ball data
    ball_df = track_data_long[['Period', 'Frame', 'time', 'xBall', 'yBall', 'team', 'second']].copy()
    ball_df['player'] = 0
    ball_df['team'] = "ball"
    ball_df = ball_df.rename(columns={'xBall': 'x', 'yBall': 'y'})

    # Combine and clean
    new_df = pd.concat([new_df, ball_df], ignore_index=True)
    new_df = (new_df
              .sort_values(['Frame', 'player', 'x', 'y'], na_position='last')
              .drop_duplicates(subset=['Frame', 'player'], keep='first')
              .dropna(subset=['x']))

    # Transform coordinates for 2nd half
    new_df['y'] = 1 - new_df['y']
    new_df['x'] = np.where(new_df['Period'] == 1, 
                          pitch_long * new_df['x'], 
                          pitch_long * (1 - new_df['x']))
    new_df['y'] = np.where(new_df['Period'] == 1,
                          pitch_width * new_df['y'],
                          pitch_width * (1 - new_df['y']))
    new_df[['x', 'y']] = new_df[['x', 'y']].round(2)

    # Add velocity if requested
    if add_velocity:
        new_df["dx"] = new_df.groupby("player")["x"].diff(MS_LAG_SMOOTH)
        new_df["dy"] = new_df.groupby("player")["y"].diff(MS_LAG_SMOOTH) 
        new_df["v_mod"] = np.sqrt(new_df["dx"]**2 + new_df["dy"]**2)
        new_df["speed"] = np.minimum(new_df["v_mod"] / (MS_DT * MS_LAG_SMOOTH), PLAYER_MAX_SPEED)
    
    # Convert time to minutes
    new_df['minutes'] = new_df['time'] // 60 + (new_df['time'] % 60) / 100
    new_df['player'] = new_df['player'].astype(int)

    return new_df

# Get the frames where a particular team has possesion
def possesion_frames(events_df: pd.DataFrame, target_team: str) -> pd.DataFrame:
    """
    Analyze possession phases and return frames where target team has possession.
    
    Args:
        events_df: DataFrame containing match events
        target_team: Team name to analyze possession for ('Home' or 'Away')
        
    Returns:
        DataFrame containing possession start/end frames for target team
    """
    # Validate inputs
    if not isinstance(events_df, pd.DataFrame):
        raise TypeError("events_df must be a pandas DataFrame")
    if target_team not in ['Home', 'Away']:
        raise ValueError("target_team must be either 'Home' or 'Away'")

    # Constants 
    poss_start_events = ['PASS', 'RECOVERY', 'SET PIECE', 'SHOT']
    poss_change_events = ["BALL LOST", "BALL OUT", "SHOT", "PASS"]
    excluded_events = ["CHALLENGE", "CARD"]
    DELAY_FIX = 0.1  # Time delay constant

    # Clean and prepare events data 
    events = (events_df
    .sort_values(by=['Period', 'Start Time [s]'])
    .reset_index(drop=True)
    .pipe(lambda df: df[~df['Type'].isin(excluded_events)])) # Remove excluded events

    # Calculate possession transitions
    events['inverted_time_ball_recovery'] = np.where(
        (events['Type'] == "BALL LOST") & 
        (events['Type'].shift(1) == "RECOVERY") & 
        (events['Team'] != events['Team'].shift(1)),
        1, 0
    )

    # Adjust start times for ball recovery events
    events['Start Time [s]'] = np.where(
        events['inverted_time_ball_recovery'] == 1,
        events['Start Time [s]'].shift(1) - DELAY_FIX, 
        events['Start Time [s]']
    )

    # Sort after time adjustments
    events = events.sort_values(by=['Period', 'Start Time [s]']).reset_index(drop=True)

    # Calculate wrong ball lost
    events['wrong_ball_lost'] = np.where(
        (events['Type'] == "BALL LOST") & (events['Subtype'] == "theft") &
        ( ((events['Team'] != events['Team'].shift(1)) & 
         (events['Team'] != events['Team'].shift(-1))) |
        ((events['Team'] == events['Team'].shift(-1)) & 
        (events['Type'].shift(-1) == "RECOVERY")) ),
        1, 0
    )

    # Calculate wrong ball recovery
    events['wrong_recover'] = np.where(
        (
            (events['Type']== "RECOVERY") &
            (events['Type'].shift(-1) == "BALL LOST") &
            (events['Type'].shift(1) == "FAULT RECEIVED") &
            (events['Team'] != events['Team'].shift(1)) &
            (events['Team'] == events['Team'].shift(-1))
        )  |    (
            (events['Type']== "RECOVERY") &
            (events['Type'].shift(1) == "BALL LOST") &
            (events['Type'].shift(2) == "FAULT RECEIVED") &
            (events['Team'] != events['Team'].shift(2)) &
            (events['Team'] == events['Team'].shift(1))
        )   ,
        1, 0
    )

    # Calculate possession start
    events['poss_start'] = np.where(
        (  (  events['Team'] != events['Team'].shift(1)   ) &
            (  events['Type'].isin(poss_start_events)  ) &
            (       (events['Type'].shift(1).isin(poss_change_events + ["RECOVERY"]))  | (events['Subtype'].shift(1).isin(poss_change_events)))       ) 
            |
            ( (events['Subtype'] == "KICK OFF") & (~pd.isna(events['Subtype']) ) ) &
        ~((events['Type'] == "RECOVERY") & (events['Team'].shift(1) == events['Team'].shift(-1)) & (events['Team'] != events['Team'].shift(-1))),
        1, 0
    )

    # Calculate possession end
    events['poss_end'] = np.where(
        (events['Team'] != events['Team'].shift(-1)) & (events['Type'].shift(-1).isin(poss_start_events)) &
        ((events['Type'].isin(poss_change_events + ["RECOVERY"])) | (events['Subtype'].isin(poss_change_events))),
        1, 0
    )

    # Alter poss start and end for ball out events
    events['poss_start'] = np.where(events['Type'].shift(1) == "BALL OUT" , 1, events['poss_start'])
    events['poss_end'] = np.where(events['Type'] == "BALL OUT", 1, events['poss_end'])

    # Alter poss start and end  
    events['poss_end'] = np.where((events['poss_end'] == 0) & (events['Team'] != events['Team'].shift(-1)) & (~pd.isna(events['Team'].shift(-1))), 1, events['poss_end'])
    events['poss_start'] = np.where((events['poss_start'] == 0) & (events['Team'] != events['Team'].shift(1)) & (~pd.isna(events['Team'].shift(1))), 1, events['poss_start'])
    events['poss_end'] = np.where((events['poss_end'] == 1) & (events['Team'] == events['Team'].shift(-1)) & (~pd.isna(events['Team'].shift(-1))), 0, events['poss_end'])
    events['poss_start'] = np.where((events['poss_start'] == 1) & (events['Team'] == events['Team'].shift(1)) & (~pd.isna(events['Team'].shift(1))), 0, events['poss_start'])
    events['frame'] = np.where(events['Type'] == "ball out", events['Start Frame'], events['End Frame'])

    # Handle unique frame cases
    unique_frame_cases = events[
        (events['poss_start'] == 1) & (events['poss_end'] == 1) & (events['Team'] == target_team)
    ].copy()
    unique_frame_cases['frame'] = unique_frame_cases['Start Frame']
    unique_frame_cases['poss_end'] = 0
    unique_frame_cases['Start Time [s]'] = unique_frame_cases['Start Time [s]'] - DELAY_FIX / 100

    # Process possessions
    events['frame'] = np.where((events['poss_start'] == 1) & (events['poss_end'] == 1), events['End Frame'], events['frame'])
    events['poss_start'] = np.where((events['poss_start'] == 1) & (events['poss_end'] == 1), 0, events['poss_start'])
    poss_processed = pd.concat([events, unique_frame_cases]).sort_values(by='Start Time [s]').reset_index(drop=True)


    # Prepare output
    output = pd.DataFrame({
        'Team': target_team,
        'poss_start': poss_processed[poss_processed['Team'] == target_team]['frame'][poss_processed['poss_start'] == 1].values,
        'poss_end': poss_processed[poss_processed['Team'] == target_team]['frame'][poss_processed['poss_end'] == 1].values
    })

    return output # Return final output df

# Function to plot single frame
def liga_frame(
    data: pd.DataFrame,
    target_frame: int,
    poss_data: pd.DataFrame,
    method: str = "base", 
    pitch_fill: str = "black", 
    pitch_lines_col: str = "#7E7D7D",
    pitch_type: str = 'opta',
    save: bool = True,
    home_team_col: str = "#0A97B0", 
    away_team_col: str = "#A04747"
    ) -> None:
    """
    Visualize tracking data for a specific frame with various visualization methods.
    
    Parameters:
    -----------
    data : DataFrame
        Tracking data 
    target_frame : int
        The specific frame number to visualize
    poss_data : DataFrame
        Possession data to determine which team has the ball
    method : str, optional
        Visualization method: 'base', 'convexhull', 'delaunay', or 'voronoi'
    pitch_fill : str, optional
        Background color of the pitch
    pitch_lines_col : str, optional
        Color of the pitch lines
    pitch_type : str, optional
        Type of pitch layout
    save : bool, optional
        Whether to save the visualization
    home_team_col : str, optional
        Color for the home team
    away_team_col : str, optional
        Color for the away team
        
    Returns:
    --------
    None, displays and optionally saves the visualization
    """

    frame_df = data[data["Frame"] == target_frame]
    frame_df = frame_df.groupby('Frame').apply(player_possession).reset_index(drop=True)
    frame_df['edgecolor'] = frame_df['has_possession'].apply(lambda x: "yellow" if x else "white")
    frame_df['edge_lw'] = frame_df['has_possession'].apply(lambda x: 1.8 if x else 0.5)

    team_colors = {
        'ball': 'white',
        'away': away_team_col,
        'home': home_team_col
    }

    team_markers = {
        'ball': 'o', 
        'away': 'o',  
        'home': 'o'  
    }

    # Map the team values to colors and markers
    frame_df['color'] = frame_df['team'].map(team_colors)
    frame_df['marker'] = frame_df['team'].map(team_markers)

    # Create the pitch
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = plt.subplot(111)

    fig.set_facecolor(pitch_fill)
    ax.patch.set_facecolor(pitch_fill)
    pitch = Pitch(pitch_color=pitch_fill,
                pitch_type='opta',
                goal_type='box',
                linewidth=0.85,
                line_color=pitch_lines_col)
    pitch.draw(ax=ax)

    # Plot the scatter points with different colors and markers
    if method == "base":
        plot_base_viz(frame_df, team_colors)
           
    elif method == "convexhull":
        plot_convexhull_viz(frame_df,pitch, ax,team_colors)
        
    elif method == "delaunay":
        plot_delaunay_viz(frame_df,pitch,ax,team_colors)
        
    elif method == "voronoi":
        plot_voronoi_viz(frame_df,pitch,ax,team_colors)
        

    # Plot ball
    team_df = frame_df[frame_df['team'] == "ball"]
    plt.scatter(team_df['x'], team_df['y'], s=50, alpha=1, facecolors='none', edgecolors="yellow", marker="o", linewidths=1.5,zorder=2)

  
    #  Find defensive line and pitch width holded
    home_width, home_defensive_line,home_def_line_height = calculate_team_stats(frame_df[frame_df['team'] == 'home'], 'home')
    away_width, away_defensive_line,away_def_line_height = calculate_team_stats(frame_df[frame_df['team'] == 'away'], 'away')
    ax.plot([home_defensive_line, home_defensive_line], [0,100], lw=1, color=home_team_col,  linestyle='--',zorder=1)
    ax.plot([away_defensive_line, away_defensive_line], [0,100], lw=1, color=away_team_col,  linestyle='--',zorder=1)
    
    max_arrow_length = 10 
    # Add player number and speed with direction
    for index, row in frame_df.iterrows():
        dx = row['dx']
        dy = row['dy']
        speed = row['speed']*0.35
        if row['player'] != 0:
            plt.text(row['x'], row['y'], str(row['player']),font="Century Gothic" ,fontsize=9, ha='center', va='center', weight="bold",color='white',zorder=3)
            dx = dx*speed
            dy= dy*speed
            plt.arrow(row['x'], row['y'], dx, dy, head_width=0.5, head_length=0.5,
                    fc='white', ec='white', zorder=2)
                    
        else:
            magnitude = np.sqrt(dx**2 + dy**2)
            plt.arrow(row['x'], row['y'], dx/magnitude*2, dy/magnitude*2, head_width=0.5, head_length=0.5, fc='yellow', ec='yellow',zorder=2)


    # Set legend with custom font
    legend_font = {
        'family': 'Century Gothic', 
        'size': 12,     
    }

    # Add legend 
    plt.legend(
        title='Team',
        bbox_to_anchor=(1.045, 0.8),
        loc='center',
        handletextpad=0.5,
        labelspacing=1.0,
        prop=legend_font ,
        title_fontproperties=legend_font , # Apply custom font properties
        borderaxespad=0.3,            # Padding between the legend and the axes
        borderpad=0.17,                # Padding between the legend border and content
        frameon=True                  # Show a border around the legend
    )


    # Add clock time
    minute_value = frame_df['minutes'].values[0]
    minutes = int(minute_value)
    seconds = int((minute_value - minutes) * 60)
    str_text = f"Time - <{minutes:02d}:{seconds:02d}>"

    fig_text(
        x = 0.42, y = 0.79, 
        s = str_text,highlight_textprops=[{'color':'#FFD230', 'weight':'bold'}],
        va = 'bottom', ha = 'left',fontname ="Century Gothic", weight = 'bold',
        fontsize = 25,color ='white',path_effects=[path_effects.Stroke(linewidth=0.8, foreground="#BD8B00"), path_effects.Normal()]
    )


     # Find current ball possesion
    possession = team_possession(target_frame, poss_data)
    if possession == "Home":
        color_now = home_team_col
    elif possession == "Away":
        color_now = away_team_col
    else:
        color_now = "white"
    str_text1 = f"Current Ball Possesion: <{possession}> "
    fig_text(
        x=0.41, y=0.77,
        highlight_textprops=[
            {'color': color_now, 'weight': 'bold'}
        ],
        s=str_text1,
        va='bottom', ha='left',
        fontname="Century Gothic", weight='bold',
        fontsize=12, color='white'
    )

    str_t = f"Viz by : @Ligandro22"
    fig_text(
        x=0.77, y=0.74,
        s=str_t,
        va='bottom', ha='left',
        fontname="STXihei", weight='bold',
        fontsize=8, color='white'
    )



    str_text2 = f"Pitch width holded by <Away> : {away_width:.2f} | <Home> : {home_width:.2f} "
    fig_text(
        x=0.34, y=0.748,
        highlight_textprops=[
            {'color': away_team_col, 'weight': 'bold'},  # Properties for <Home>
            {'color': home_team_col, 'weight': 'bold'}   # Properties for <Away>
        ],
        s=str_text2,
        va='bottom', ha='left',
        fontname="Century Gothic", weight='bold',
        fontsize=11, color='white'
    )

    str_text2 = f"Defensive Line Height <Away --> : {away_def_line_height:.2f} | <Home -- > : {home_def_line_height:.2f} "
    fig_text(
        x=0.32, y=0.23,
        highlight_textprops=[
            {'color': away_team_col, 'weight': 'bold'},  # Properties for <Home>
            {'color': home_team_col, 'weight': 'bold'}   # Properties for <Away>
        ],
        s=str_text2,
        va='bottom', ha='left',
        fontname="Century Gothic", weight='bold',
        fontsize=11, color='white'
    )

    str_t = f"Viz by : @Ligandro22"
    fig_text(
        x=0.77, y=0.74,
        s=str_t,
        va='bottom', ha='left',
        fontname="STXihei", weight='bold',
        fontsize=8, color='white'
    )


    if save == True :
        plt.savefig(f"images/frame_{target_frame}.jpg",dpi =500, bbox_inches='tight')

    plt.show()
                                
# Function to animate the frames
def liga_animate(
    tidy_data: pd.DataFrame, 
    poss_data: pd.DataFrame, 
    frame_start: int, 
    frame_end: int, 
    mode: str = 'base', 
    video_writer: str = "gif",
    pitch_fill: str = "black", 
    pitch_lines_col: str = "#7E7D7D", 
    pitch_type: str = 'opta', 
    save: bool = True, 
    home_team_col: str = "#0A97B0", 
    away_team_col: str = "#A04747"
    ) -> None:
    """
    Generates an animation of a range of frames for a football match.

    Steps:
    -------
    1. Filters data to include only frames within the specified range.
    2. Determines ball possession for each frame and sets edge colors accordingly.
    3. Maps teams to their respective colors and marker styles.
    4. Initializes the pitch for visualization.
    5. Calls the appropriate animation function based on the selected mode.
    6. Choose mode of saving the animation (gif or mp4). For mp4 format, ffmpeg is required.
    
    Parameters:
    -----------
    tidy_data : pd.DataFrame
        Processed tracking data containing player positions and other match details.
    poss_data : pd.DataFrame
        Data containing possession frames for the teams
    frame_start : int
        The starting frame number for the animation.
    frame_end : int
        The ending frame number for the animation.
    mode : str, optional
        Visualization mode (default is 'base'). Other options are 'convexhull', 'delaunay', and 'voronoi'.
    video_writer : str, optional
        Format for saving the animation (default is 'gif'). Other option is 'mp4'.
    pitch_fill : str, optional
        Color of the pitch background (default is 'black').
    pitch_lines_col : str, optional
        Color of the pitch lines (default is '#7E7D7D').
    pitch_type : str, optional
        Type of pitch to be drawn (default is 'opta').
    save : bool, optional
        Whether to save the animation (default is True).
    home_team_col : str, optional
        Color representing the home team (default is '#0A97B0').
    away_team_col : str, optional
        Color representing the away team (default is '#A04747').

    Returns:
    --------
    None. Saves the video file if save set to True at videos/animation.mp4 or videos/animation.gif
    """

    # Filter data for the selected frame range
    frame_df = tidy_data[(tidy_data["Frame"] > frame_start) & (tidy_data["Frame"] < frame_end)]
    
    # Determine ball possession for each frame
    frame_df = frame_df.groupby('Frame').apply(player_possession).reset_index(drop=True)
    
    # Assign edge color and linewidth based on possession
    frame_df['edgecolor'] = frame_df['has_possession'].apply(lambda x: "yellow" if x else "white")
    frame_df['edge_lw'] = frame_df['has_possession'].apply(lambda x: 1.8 if x else 0.5)

    # Define team colors for visualization
    team_colors = {
        'ball': 'white',
        'away': away_team_col,
        'home': home_team_col
    }

    # Define marker styles for teams
    team_markers = {
        'ball': 'o',  
        'away': 'o', 
        'home': 'o'   
    }

    # Map team names to colors and markers
    frame_df['color'] = frame_df['team'].map(team_colors)
    frame_df['marker'] = frame_df['team'].map(team_markers)

    # Create the pitch figure
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    fig.set_facecolor(pitch_fill)  # Set background color
    ax.patch.set_facecolor(pitch_fill)

    # Initialize the pitch
    pitch = Pitch(
        pitch_color=pitch_fill, pitch_type=pitch_type,
        goal_type='box', linewidth=0.85, line_color=pitch_lines_col
    )
    pitch.draw(ax=ax)


    # Function to update the plot for each frame by plotting voronoi plots
    def animate_voronoi(frame: int):
        # Clear the current axis to prepare for the new frame
        ax.clear()
        pitch.draw(ax=ax)
        frame_data = frame_df[frame_df['Frame'] == frame] 

        # Plot the ball's position
        team_df = frame_data[frame_data['team'] == "ball"]
        plt.scatter(team_df['x'], team_df['y'], s=50, alpha=1, facecolors='none', edgecolors="yellow", marker="o", linewidths=1.5,zorder=2)
        
        # Filter out the ball data 
        tracking_full = frame_data[frame_data['team'] != "ball"]
        tracking_full = tracking_full[['x', 'y', 'team']]

        # Separate home and away team data
        tracking_home = tracking_full[tracking_full['team']=="home"]
        tracking_away = tracking_full[tracking_full['team']=="away"]

        # Extract x, y coordinates and team information for Voronoi calculation
        X = tracking_full.x
        Y = tracking_full.y
        Team = tracking_full['team'].map({'home': 0, 'away': 1})  # Map teams to numerical values (0 for home, 1 for away)

        # Calculate Voronoi regions for home and away teams
        vor_away,vor_home = pitch.voronoi(X, Y, Team)

        # Plot Voronoi polygons for the team
        pitch.polygon(vor_home, fc=home_team_col, ax=ax, ec='white', lw=3, alpha=0.4)
        pitch.polygon(vor_away, fc=away_team_col, ax=ax, ec='white', lw=3, alpha=0.4)

        # Add player numbers and movement direction
        for _, row in frame_data.iterrows():
            dx, dy, speed = row['dx'], row['dy'], row['speed'] * 0.35
            if row['player'] != 0: # Exclude ball
                plt.text(
                    row['x'], row['y'], str(row['player']),
                    fontname="Century Gothic", fontsize=9, ha='center', va='center', 
                    weight="bold", color='white', zorder=4
                )
                plt.arrow(row['x'], row['y'], dx * speed, dy * speed, head_width=0.5, head_length=0.5,
                        fc='white', ec='white', zorder=2)
            else:
                # Keep ball speed constant for better visualization
                magnitude = np.sqrt(dx**2 + dy**2) or 1  # Avoid division by zero
                plt.arrow(row['x'], row['y'], dx / magnitude * 2, dy / magnitude * 2,
                        head_width=0.5, head_length=0.5, fc='yellow', ec='yellow', zorder=2)


        # Plot players
        frame_data = frame_data[frame_data['team'] != "ball"]
        for team, team_col in [('away', away_team_col), ('home',home_team_col)]:
            team_df = frame_data[frame_data['team'] == team]
            pitch.scatter(team_df['x'], team_df['y'], ax=ax, s=250, edgecolor=team_df['edgecolor'],linewidths=team_df['edge_lw'], marker="o", c=team_col, zorder=3,label=team)


        # Compute defensive line and pitch width
        home_width, home_def_line, home_def_line_height = calculate_team_stats(frame_data[frame_data['team'] == 'home'], 'home')
        away_width, away_def_line, away_def_line_height = calculate_team_stats(frame_data[frame_data['team'] == 'away'], 'away')

        # Plot defensive lines
        ax.plot([home_def_line, home_def_line], [0, 100], lw=1, color=team_colors["home"], linestyle='--', zorder=1)
        ax.plot([away_def_line, away_def_line], [0, 100], lw=1, color=team_colors["away"], linestyle='--', zorder=1)

        # Set custom legend
        plt.legend(
            title='Team', bbox_to_anchor=(1.045, 0.8), loc='center', handletextpad=0.5,
            labelspacing=1.0, prop={'family': 'Century Gothic', 'size': 12}, 
            title_fontproperties={'family': 'Century Gothic', 'size': 12},
            borderaxespad=0.3, borderpad=0.17, frameon=True
        )

        # Display match time
        minute_value = frame_data['minutes'].values[0]
        minutes = int(minute_value)
        seconds = int((minute_value - minutes) * 60)
        str_text = f"Time - <{minutes:02d}:{seconds:02d}>"

        fig_text(
            x = 0.425, y = 0.79, 
            s = str_text,highlight_textprops=[{'color':'#FFD230', 'weight':'bold'}],
            va = 'bottom', ha = 'left',fontname ="Century Gothic", weight = 'bold',
            fontsize = 25,color ='white',path_effects=[path_effects.Stroke(linewidth=0.8, foreground="#BD8B00"), path_effects.Normal()]
        )
        

        # Determine current ball possession
        possession = team_possession(frame, poss_data)
        color_now = team_colors.get(possession.lower(), "white")
        fig_text(
            x=0.41, y=0.77, s=f"Current Ball Possession: <{possession}>",
            highlight_textprops=[{'color': color_now, 'weight': 'bold'}],
            va='bottom', ha='left', fontname="Century Gothic", weight='bold',
            fontsize=12, color='white'
        )

        # Display pitch width stats
        fig_text(
            x=0.34, y=0.748, s=f"Pitch width held by <Away> : {away_width:.2f} | <Home> : {home_width:.2f}",
            highlight_textprops=[{'color': team_colors["away"], 'weight': 'bold'}, 
                                {'color': team_colors["home"], 'weight': 'bold'}],
            va='bottom', ha='left', fontname="Century Gothic", weight='bold',
            fontsize=11, color='white'
        )

        # Display defensive line height
        fig_text(
            x=0.32, y=0.23, s=f"Defensive Line Height <Away --> : {away_def_line_height:.2f} | <Home -- > : {home_def_line_height:.2f}",
            highlight_textprops=[{'color': team_colors["away"], 'weight': 'bold'}, 
                                {'color': team_colors["home"], 'weight': 'bold'}],
            va='bottom', ha='left', fontname="Century Gothic", weight='bold',
            fontsize=11, color='white'
        )

        # Add visualization credit
        fig_text(
            x=0.77, y=0.74, s="Viz by: @Ligandro22",
            va='bottom', ha='left', fontname="STXihei", weight='bold',
            fontsize=8, color='white'
        )


    # Function to update the plot for each frame by plotting delaunay plots
    def animate_delaunay(frame: int):
        # Clear the current axis to prepare for the new frame
        ax.clear()
        pitch.draw(ax=ax)
        frame_data = frame_df[frame_df['Frame'] == frame] 

        # Plot the ball's position
        team_df = frame_data[frame_data['team'] == "ball"]
        plt.scatter(team_df['x'], team_df['y'], s=50, alpha=1, facecolors='none', edgecolors="yellow", marker="o", linewidths=1.5,zorder=2)
        
        # Filter out the ball data 
        tracking_full = frame_data[frame_data['team'] != "ball"]
        tracking_full = tracking_full[['x', 'y', 'team']]

        # Separate home and away team data
        tracking_home = tracking_full[tracking_full['team']=="home"]
        tracking_away = tracking_full[tracking_full['team']=="away"]

        # Convert to arrays for Delauny calculation
        points_home = tracking_home[['x', 'y']].values
        del_home= Delaunay(tracking_home[['x', 'y']])

        points_away= tracking_away[['x', 'y']].values
        del_away= Delaunay(tracking_away[['x', 'y']])

        # Draw Delauny triangles
        for i in del_home.simplices:
            plt.plot(points_home[i, 0], points_home[i, 1], home_team_col, zorder = 1)

        for i in del_away.simplices:
            plt.plot(points_away[i, 0], points_away[i, 1], away_team_col, zorder = 1)

        # Add player numbers and movement direction
        for _, row in frame_data.iterrows():
            dx, dy, speed = row['dx'], row['dy'], row['speed'] * 0.35
            if row['player'] != 0: # Exclude ball
                plt.text(
                    row['x'], row['y'], str(row['player']),
                    fontname="Century Gothic", fontsize=9, ha='center', va='center', 
                    weight="bold", color='white', zorder=4
                )
                plt.arrow(row['x'], row['y'], dx * speed, dy * speed, head_width=0.5, head_length=0.5,
                        fc='white', ec='white', zorder=2)
            else:
                # Keep ball speed constant for better visualization
                magnitude = np.sqrt(dx**2 + dy**2) or 1  # Avoid division by zero
                plt.arrow(row['x'], row['y'], dx / magnitude * 2, dy / magnitude * 2,
                        head_width=0.5, head_length=0.5, fc='yellow', ec='yellow', zorder=2)


        # Plot players
        frame_data = frame_data[frame_data['team'] != "ball"]
        for team, team_col in [('away', away_team_col), ('home',home_team_col)]:
            team_df = frame_data[frame_data['team'] == team]
            pitch.scatter(team_df['x'], team_df['y'], ax=ax, s=250, edgecolor=team_df['edgecolor'],linewidths=team_df['edge_lw'], marker="o", c=team_col, zorder=3,label=team)


        # Compute defensive line and pitch width
        home_width, home_def_line, home_def_line_height = calculate_team_stats(frame_data[frame_data['team'] == 'home'], 'home')
        away_width, away_def_line, away_def_line_height = calculate_team_stats(frame_data[frame_data['team'] == 'away'], 'away')

        # Plot defensive lines
        ax.plot([home_def_line, home_def_line], [0, 100], lw=1, color=team_colors["home"], linestyle='--', zorder=1)
        ax.plot([away_def_line, away_def_line], [0, 100], lw=1, color=team_colors["away"], linestyle='--', zorder=1)

        # Set custom legend
        plt.legend(
            title='Team', bbox_to_anchor=(1.045, 0.8), loc='center', handletextpad=0.5,
            labelspacing=1.0, prop={'family': 'Century Gothic', 'size': 12}, 
            title_fontproperties={'family': 'Century Gothic', 'size': 12},
            borderaxespad=0.3, borderpad=0.17, frameon=True
        )

        # Display match time
        minute_value = frame_data['minutes'].values[0]
        minutes = int(minute_value)
        seconds = int((minute_value - minutes) * 60)
        str_text = f"Time - <{minutes:02d}:{seconds:02d}>"

        fig_text(
            x = 0.425, y = 0.79, 
            s = str_text,highlight_textprops=[{'color':'#FFD230', 'weight':'bold'}],
            va = 'bottom', ha = 'left',fontname ="Century Gothic", weight = 'bold',
            fontsize = 25,color ='white',path_effects=[path_effects.Stroke(linewidth=0.8, foreground="#BD8B00"), path_effects.Normal()]
        )
        

        # Determine current ball possession
        possession = team_possession(frame, poss_data)
        color_now = team_colors.get(possession.lower(), "white")
        fig_text(
            x=0.41, y=0.77, s=f"Current Ball Possession: <{possession}>",
            highlight_textprops=[{'color': color_now, 'weight': 'bold'}],
            va='bottom', ha='left', fontname="Century Gothic", weight='bold',
            fontsize=12, color='white'
        )

        # Display pitch width stats
        fig_text(
            x=0.34, y=0.748, s=f"Pitch width held by <Away> : {away_width:.2f} | <Home> : {home_width:.2f}",
            highlight_textprops=[{'color': team_colors["away"], 'weight': 'bold'}, 
                                {'color': team_colors["home"], 'weight': 'bold'}],
            va='bottom', ha='left', fontname="Century Gothic", weight='bold',
            fontsize=11, color='white'
        )

        # Display defensive line height
        fig_text(
            x=0.32, y=0.23, s=f"Defensive Line Height <Away --> : {away_def_line_height:.2f} | <Home -- > : {home_def_line_height:.2f}",
            highlight_textprops=[{'color': team_colors["away"], 'weight': 'bold'}, 
                                {'color': team_colors["home"], 'weight': 'bold'}],
            va='bottom', ha='left', fontname="Century Gothic", weight='bold',
            fontsize=11, color='white'
        )

        # Add visualization credit
        fig_text(
            x=0.77, y=0.74, s="Viz by: @Ligandro22",
            va='bottom', ha='left', fontname="STXihei", weight='bold',
            fontsize=8, color='white'
        )


    # Function to update the plot for each frame by plotting convex hulls
    def animate_convex(frame: int):
        ax.clear()
        pitch.draw(ax=ax)
        frame_data = frame_df[frame_df['Frame'] == frame]


        team_df = frame_data[frame_data['team'] == "ball"]
        plt.scatter(team_df['x'], team_df['y'], s=50, alpha=1, facecolors='none', edgecolors="yellow", marker="o", linewidths=1.5,zorder=2)   

        for team in ['away', 'home']:
            team_df = frame_data[frame_data['team'] == team]
            if not team_df.empty:
                # Exclude players with numbers 11 and 25 for convex hull calculation
                team_df['player'] = team_df['player'].astype(int)
                hull_df = team_df[~team_df['player'].isin([11, 25])]
                if not hull_df.empty:
                    hull = pitch.convexhull(hull_df['x'], hull_df['y'])
                    if team == 'away':
                        poly = pitch.polygon(hull, ax=ax, edgecolor='red', facecolor=away_team_col, alpha=0.3, zorder=-1)
                    elif team == 'home':
                        poly = pitch.polygon(hull, ax=ax, edgecolor='blue', facecolor=home_team_col, alpha=0.3, zorder=-1)

            # Include all players in the scatter plot
            if team == 'away':
                scatter = pitch.scatter(team_df['x'], team_df['y'], ax=ax, s=250, edgecolor=team_df['edgecolor'],linewidths=team_df['edge_lw'], marker="o", c=away_team_col, zorder=3,label=team)
            elif team == 'home':
                scatter = pitch.scatter(team_df['x'], team_df['y'], ax=ax, s=250, edgecolor=team_df['edgecolor'], linewidths=team_df['edge_lw'],marker="o", c=home_team_col, zorder=3,label=team)
            elif team == 'ball':
            # Special handling for the ball
                scatter=plt.scatter(team_df['x'], team_df['y'], s=50, alpha=1, facecolors='none', edgecolors="yellow", marker="o", linewidths=1.5,zorder=3)

        
        # Add player numbers and movement direction
        for _, row in frame_data.iterrows():
            dx, dy, speed = row['dx'], row['dy'], row['speed'] * 0.35
            if row['player'] != 0: # Exclude ball
                plt.text(
                    row['x'], row['y'], str(row['player']),
                    fontname="Century Gothic", fontsize=9, ha='center', va='center', 
                    weight="bold", color='white', zorder=4
                )
                plt.arrow(row['x'], row['y'], dx * speed, dy * speed, head_width=0.5, head_length=0.5,
                        fc='white', ec='white', zorder=2)
            else:
                # Keep ball speed constant for better visualization
                magnitude = np.sqrt(dx**2 + dy**2) or 1  # Avoid division by zero
                plt.arrow(row['x'], row['y'], dx / magnitude * 2, dy / magnitude * 2,
                        head_width=0.5, head_length=0.5, fc='yellow', ec='yellow', zorder=2)

        # Compute defensive line and pitch width
        home_width, home_def_line, home_def_line_height = calculate_team_stats(frame_data[frame_data['team'] == 'home'], 'home')
        away_width, away_def_line, away_def_line_height = calculate_team_stats(frame_data[frame_data['team'] == 'away'], 'away')

        # Plot defensive lines
        ax.plot([home_def_line, home_def_line], [0, 100], lw=1, color=team_colors["home"], linestyle='--', zorder=1)
        ax.plot([away_def_line, away_def_line], [0, 100], lw=1, color=team_colors["away"], linestyle='--', zorder=1)

        # Set custom legend
        plt.legend(
            title='Team', bbox_to_anchor=(1.045, 0.8), loc='center', handletextpad=0.5,
            labelspacing=1.0, prop={'family': 'Century Gothic', 'size': 12}, 
            title_fontproperties={'family': 'Century Gothic', 'size': 12},
            borderaxespad=0.3, borderpad=0.17, frameon=True
        )

        # Display match time
        minute_value = frame_data['minutes'].values[0]
        minutes = int(minute_value)
        seconds = int((minute_value - minutes) * 60)
        str_text = f"Time - <{minutes:02d}:{seconds:02d}>"

        fig_text(
            x = 0.425, y = 0.79, 
            s = str_text,highlight_textprops=[{'color':'#FFD230', 'weight':'bold'}],
            va = 'bottom', ha = 'left',fontname ="Century Gothic", weight = 'bold',
            fontsize = 25,color ='white',path_effects=[path_effects.Stroke(linewidth=0.8, foreground="#BD8B00"), path_effects.Normal()]
        )
        

        # Determine current ball possession
        possession = team_possession(frame, poss_data)
        color_now = team_colors.get(possession.lower(), "white")
        fig_text(
            x=0.41, y=0.77, s=f"Current Ball Possession: <{possession}>",
            highlight_textprops=[{'color': color_now, 'weight': 'bold'}],
            va='bottom', ha='left', fontname="Century Gothic", weight='bold',
            fontsize=12, color='white'
        )

        # Display pitch width stats
        fig_text(
            x=0.34, y=0.748, s=f"Pitch width held by <Away> : {away_width:.2f} | <Home> : {home_width:.2f}",
            highlight_textprops=[{'color': team_colors["away"], 'weight': 'bold'}, 
                                {'color': team_colors["home"], 'weight': 'bold'}],
            va='bottom', ha='left', fontname="Century Gothic", weight='bold',
            fontsize=11, color='white'
        )

        # Display defensive line height
        fig_text(
            x=0.32, y=0.23, s=f"Defensive Line Height <Away --> : {away_def_line_height:.2f} | <Home -- > : {home_def_line_height:.2f}",
            highlight_textprops=[{'color': team_colors["away"], 'weight': 'bold'}, 
                                {'color': team_colors["home"], 'weight': 'bold'}],
            va='bottom', ha='left', fontname="Century Gothic", weight='bold',
            fontsize=11, color='white'
        )

        # Add visualization credit
        fig_text(
            x=0.77, y=0.74, s="Viz by: @Ligandro22",
            va='bottom', ha='left', fontname="STXihei", weight='bold',
            fontsize=8, color='white'
        )

    # Function to update the plot for each frame by plotting base plots
    def animate_base( frame: int):
        """
        Animates a single frame in the football match visualization.

        Steps:
        -------
        1. Clears the plot and redraws the pitch.
        2. Plots player positions and ball with appropriate colors and markers.
        3. Adds player numbers and movement arrows.
        4. Computes and displays defensive line height and pitch width for each team.
        5. Displays time, possession, and visualization credits.

        Parameters:
        -----------
        frame : int
            Current frame number to be visualized.
        team_colors : dict
            Dictionary mapping teams ('home', 'away', 'ball') to their respective colors.

        Returns:
        --------
        None
        """

        # Clear previous frame and redraw pitch
        ax.clear()
        pitch.draw(ax=ax)

        # Filter data for the current frame
        frame_data = frame_df[frame_df['Frame'] == frame]

        # Plot players and ball positions
        for team, marker in team_markers.items():
            team_df = frame_data[frame_data['team'] == team]
            if team == 'ball':
                plt.scatter(
                    team_df['x'], team_df['y'], s=50, alpha=1, 
                    facecolors='none', edgecolors="yellow", marker=marker, linewidths=1.5, zorder=3
                )
            else:
                # For player plot with custom colors and markers
                plt.scatter(
                    team_df['x'], team_df['y'], s=250, alpha=1, 
                    c=team_df['color'], edgecolors=team_df['edgecolor'], marker=marker, 
                    linewidths=team_df['edge_lw'], label=team, zorder=3
                )

        # Add player numbers and movement direction
        for _, row in frame_data.iterrows():
            dx, dy, speed = row['dx'], row['dy'], row['speed'] * 0.35
            if row['player'] != 0: # Exclude ball
                plt.text(
                    row['x'], row['y'], str(row['player']),
                    fontname="Century Gothic", fontsize=9, ha='center', va='center', 
                    weight="bold", color='white', zorder=4
                )
                plt.arrow(row['x'], row['y'], dx * speed, dy * speed, head_width=0.5, head_length=0.5,
                        fc='white', ec='white', zorder=2)
            else:
                # Keep ball speed constant for better visualization
                magnitude = np.sqrt(dx**2 + dy**2) or 1  # Avoid division by zero
                plt.arrow(row['x'], row['y'], dx / magnitude * 2, dy / magnitude * 2,
                        head_width=0.5, head_length=0.5, fc='yellow', ec='yellow', zorder=2)

        # Compute defensive line and pitch width
        home_width, home_def_line, home_def_line_height = calculate_team_stats(frame_data[frame_data['team'] == 'home'], 'home')
        away_width, away_def_line, away_def_line_height = calculate_team_stats(frame_data[frame_data['team'] == 'away'], 'away')

        # Plot defensive lines
        ax.plot([home_def_line, home_def_line], [0, 100], lw=1, color=team_colors["home"], linestyle='--', zorder=1)
        ax.plot([away_def_line, away_def_line], [0, 100], lw=1, color=team_colors["away"], linestyle='--', zorder=1)

        # Set custom legend
        plt.legend(
            title='Team', bbox_to_anchor=(1.045, 0.8), loc='center', handletextpad=0.5,
            labelspacing=1.0, prop={'family': 'Century Gothic', 'size': 12}, 
            title_fontproperties={'family': 'Century Gothic', 'size': 12},
            borderaxespad=0.3, borderpad=0.17, frameon=True
        )

        # Display match time
        minute_value = frame_data['minutes'].values[0]
        minutes = int(minute_value)
        seconds = int((minute_value - minutes) * 60)
        str_text = f"Time - <{minutes:02d}:{seconds:02d}>"

        fig_text(
            x = 0.425, y = 0.79, 
            s = str_text,highlight_textprops=[{'color':'#FFD230', 'weight':'bold'}],
            va = 'bottom', ha = 'left',fontname ="Century Gothic", weight = 'bold',
            fontsize = 25,color ='white',path_effects=[path_effects.Stroke(linewidth=0.8, foreground="#BD8B00"), path_effects.Normal()]
        )
        

        # Determine current ball possession
        possession = team_possession(frame, poss_data)
        color_now = team_colors.get(possession.lower(), "white")
        fig_text(
            x=0.41, y=0.77, s=f"Current Ball Possession: <{possession}>",
            highlight_textprops=[{'color': color_now, 'weight': 'bold'}],
            va='bottom', ha='left', fontname="Century Gothic", weight='bold',
            fontsize=12, color='white'
        )

        # Display pitch width stats
        fig_text(
            x=0.34, y=0.748, s=f"Pitch width held by <Away> : {away_width:.2f} | <Home> : {home_width:.2f}",
            highlight_textprops=[{'color': team_colors["away"], 'weight': 'bold'}, 
                                {'color': team_colors["home"], 'weight': 'bold'}],
            va='bottom', ha='left', fontname="Century Gothic", weight='bold',
            fontsize=11, color='white'
        )

        # Display defensive line height
        fig_text(
            x=0.32, y=0.23, s=f"Defensive Line Height <Away --> : {away_def_line_height:.2f} | <Home -- > : {home_def_line_height:.2f}",
            highlight_textprops=[{'color': team_colors["away"], 'weight': 'bold'}, 
                                {'color': team_colors["home"], 'weight': 'bold'}],
            va='bottom', ha='left', fontname="Century Gothic", weight='bold',
            fontsize=11, color='white'
        )

        # Add visualization credit
        fig_text(
            x=0.77, y=0.74, s="Viz by: @Ligandro22",
            va='bottom', ha='left', fontname="STXihei", weight='bold',
            fontsize=8, color='white'
        )


    # Create the animation
    if mode =="base":
        ani = FuncAnimation(fig, animate_base,frames=frame_df['Frame'].unique(), repeat=False)
    elif mode =="voronoi":
        ani = FuncAnimation(fig, animate_voronoi,frames=frame_df['Frame'].unique(), repeat=False)
    elif mode =="delaunay":
        ani = FuncAnimation(fig, animate_delaunay,frames=frame_df['Frame'].unique(), repeat=False)
    elif mode =="convexhull":
        ani = FuncAnimation(fig, animate_convex,frames=frame_df['Frame'].unique(), repeat=False)
    else:
        print("Error : Enter correct mode")

    if video_writer =="gif":
        ani.save('videos/animation.gif', writer='pillow', fps=25, dpi=100)
    elif video_writer =="mp4":
        ani.save('videos/animation.mp4', writer='ffmpeg', fps=25, bitrate=2000, dpi=100)


# Function to plot on/off ball possession positioning of teams
def liga_plot_poss(
    tidy_data: pd.DataFrame, 
    poss_data: pd.DataFrame, 
    target_team: str, 
    pitch_fill: str = "black", 
    pitch_lines_col: str = "#7E7D7D",
    pitch_type: str = 'opta', 
    save: bool = True,
    home_team_col: str = "#0A97B0", 
    away_team_col: str = "#A04747"
    ) -> None:
    """
    Plot the on/off ball possession positioning of teams.

    Steps:
    -------
    1. Find dataframe where a team is on/off possesion .
    2. Calculate average player positions for in possession/out possession
    3. Find defensive line avg and high line avg.
    4. Initializes the pitch for visualization.
    5. Plot convex hull and player positions for on/off possession.
    6. Create table of stats and plot it onto the table
    
    Parameters:
    -----------
    tidy_data : pd.DataFrame
        Processed tracking data containing player positions and other match details.
    poss_data : pd.DataFrame
        Data containing possession frames for the teams
    target_team : str
        Team name to analyze ('home' or 'away').
    pitch_fill : str, optional
        Background color of the pitch (default is 'black').
    pitch_lines_col : str, optional
        Color of the pitch lines (default is '#7E7D7D').
    pitch_type : str, optional
        Type of pitch layout (default is 'opta').
    save : bool, optional
        Whether to save the plot (default is True). Saves at images/{team}_possession.png
    home_team_col : str, optional
        Color representing the home team (default is '#0A97B0').
    away_team_col : str, optional
        Color representing the away team (default is '#A04747').
    Returns:
        None. Displays and optionally saves the plot.
    """
    # Subset data of particular team
    stats_df = tidy_data[tidy_data["team"] == target_team].copy()

    # Find frames where a team has possession
    home_poss = poss_data[poss_data["Team"] == "Home"] 
    away_poss = poss_data[poss_data["Team"] == "Away"]

    # Store possession frames as list
    possession_frames_home = generate_possession_frames(home_poss)
    possession_frames_away = generate_possession_frames(away_poss)

    # Set new column to denote in possession or out possession, 
    # we are considering team to be out of possesion only if the oppossing team has possession
    if target_team == "home":
        stats_df['in_possession'] = stats_df['Frame'].isin(possession_frames_home).astype(int)
        stats_df['out_possession'] = stats_df['Frame'].isin(possession_frames_away).astype(int)
    else:
        stats_df['in_possession'] = stats_df['Frame'].isin(possession_frames_away).astype(int)
        stats_df['out_possession'] = stats_df['Frame'].isin(possession_frames_home).astype(int)

    # Get dataframe where team is in/out possession
    team_inposs = stats_df[stats_df["in_possession"] == 1]
    team_outposs = stats_df[stats_df["out_possession"] == 1]

    # Find average player positions for in possession/out possession
    avg_positions_ip = team_inposs.groupby('player')[['x', 'y']].mean().reset_index()
    avg_positions_op = team_outposs.groupby('player')[['x', 'y']].mean().reset_index()

    # Keep only the players who started the match
    avg_positions_ip = avg_positions_ip.sort_values(by="player").head(11)
    avg_positions_op = avg_positions_op.sort_values(by="player").head(11)

    # Find defensive line avg and high line avg
    if target_team == "home":
        team_df_line_inposs, team_high_xip = find_stats_home(team_inposs, gk=11)
        team_df_line_oposs, team_high_xop = find_stats_home(team_outposs, gk=11)
    else:
        team_df_line_inposs, team_high_xip = find_stats_away(team_inposs, gk=25)
        team_df_line_oposs, team_high_xop = find_stats_away(team_outposs, gk=25)

    # Create the pitch
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = plt.subplot(111)

    fig.set_facecolor(pitch_fill)
    ax.patch.set_facecolor(pitch_fill)
    pitch = Pitch(pitch_color=pitch_fill,
                pitch_type=pitch_type,
                goal_type='box',
                linewidth=0.85,
                line_color=pitch_lines_col)
    pitch.draw(ax=ax)

    # Set team color
    color = home_team_col if target_team == "home" else away_team_col

    # Plot in-possession positions
    plt.scatter(avg_positions_ip['x'], avg_positions_ip['y'], s=250, alpha=1, c=color, edgecolors="white", marker="o", linewidths=1.6, zorder=3)

    # Add player numbers
    for index, row in avg_positions_ip.iterrows():
        plt.text(row['x'], row['y'], str(int(row['player'])), font="Century Gothic", fontsize=9, ha='center', va='center', weight="bold", color='white', zorder=3)

    # Calculate and plot convex hull for in-possession
    hull_df_ip = avg_positions_ip[~avg_positions_ip['player'].isin([11, 25])]
    if not hull_df_ip.empty:
        hull_ip = pitch.convexhull(hull_df_ip['x'], hull_df_ip['y'])
        hull_spatial_ip = ConvexHull(hull_df_ip[['x', 'y']])
        pitch.polygon(hull_ip, ax=ax, edgecolor=color, facecolor=color, alpha=0.3, zorder=-1)

    # Plot defensive line for in-possession
    ax.plot([team_df_line_inposs, team_df_line_inposs], [0, 100], lw=1, color=color, linestyle='--', zorder=1)

    # Calculate spread and area for in-possession
    ip_area = hull_spatial_ip.area
    points_ip = hull_df_ip[['x', 'y']].values
    ip_spread = max(np.linalg.norm(points_ip[i] - points_ip[j]) for i in range(len(points_ip)) for j in range(i + 1, len(points_ip)))

    # Plot out-possession positions
    plt.scatter(avg_positions_op['x'], avg_positions_op['y'], s=250, alpha=1, c="white", edgecolors=color, marker="o", linewidths=1.6, zorder=3)

    # Add player numbers
    for index, row in avg_positions_op.iterrows():
        plt.text(row['x'], row['y'], str(int(row['player'])), font="Century Gothic", fontsize=9, ha='center', va='center', weight="bold", color='black', zorder=3)

    # Calculate and plot convex hull for out-possession
    hull_df_op = avg_positions_op[~avg_positions_op['player'].isin([11, 25])]
    if not hull_df_op.empty:
        hull_op = pitch.convexhull(hull_df_op['x'], hull_df_op['y'])
        hull_spatial_op = ConvexHull(hull_df_op[['x', 'y']])
        pitch.polygon(hull_op, ax=ax, edgecolor="white", facecolor="white", alpha=0.3, zorder=-1)

    # Plot defensive line for out-possession
    ax.plot([team_df_line_oposs, team_df_line_oposs], [0, 100], lw=1, color="white", linestyle='--', zorder=1)

    # Calculate spread and area for out-possession
    op_area = hull_spatial_op.area
    points_op = hull_df_op[['x', 'y']].values
    op_spread = max(np.linalg.norm(points_op[i] - points_op[j]) for i in range(len(points_op)) for j in range(i + 1, len(points_op)))


    # Connect players
    for player in avg_positions_ip['player']:
        ip_pos = avg_positions_ip[avg_positions_ip['player'] == player]
        op_pos = avg_positions_op[avg_positions_op['player'] == player]
        plt.plot([ip_pos['x'].values[0], op_pos['x'].values[0]],
                [ip_pos['y'].values[0], op_pos['y'].values[0]],
                color='gray', linestyle='--', linewidth=1, zorder=2)

    # Add direction of play text and arrow for home team
    if target_team == "home":  
        fig_text(
            x = 0.503, y = 0.237, 
            s = "Direction of Play",
            va = 'bottom', ha = 'left',fontname ="Century Gothic",
            fontsize = 9,color ='white'
        )
        arrow_location = (48, -2.8)
        plt.arrow(
            arrow_location[0], arrow_location[1],
            -3, 0,  # Adjust the arrow direction to make it horizontal
            shape='full', color='white', linewidth=4,
            head_width=0.2, head_length=0.2
        )
    # Add direction of play text and arrow for away team
    else:            
        fig_text(
            x = 0.42, y = 0.237, 
            s = "Direction of Play",
            va = 'bottom', ha = 'left',fontname ="Century Gothic",
            fontsize = 9,color ='white'
        )
        arrow_location = (52, -2.8)
        plt.arrow(
            arrow_location[0], arrow_location[1],
            3, 0,  # Adjust the arrow direction to make it horizontal
            shape='full', color='white', linewidth=4,
            head_width=0.2, head_length=0.2
        )

    # Add text for on/off ball average positioning
    str_text = f"<ON>/<OFF> Ball avg. positioning -<{target_team.capitalize()} Team>"
    fig_text(
        x = 0.14, y = 0.745, 
        s = str_text,highlight_textprops=[{'color':color, 'weight':'bold'},
        {'color':"white", 'weight':'bold'},{'color':color, 'weight':'bold'}],
        va = 'bottom', ha = 'left',fontname ="Century Gothic", weight = 'bold',
        fontsize = 13,color ='white'
    )

    # Add visualization credit
    str_t = f"Viz by : @Ligandro22"
    fig_text(
        x=0.77, y=0.74,
        s=str_t,
        va='bottom', ha='left',
        fontname="STXihei", weight='bold',
        fontsize=8, color='white'
    )
 
    # Add table over the plot
    # Define columns for the DataFrame
    columns = ['Metric', 'OFF', 'ON']
    df = pd.DataFrame(columns=columns)

    # Data rows for the DataFrame
    data_rows = [
        {'Metric': "Def Line x", 'OFF': team_df_line_oposs, 'ON': team_df_line_inposs},
        {'Metric':  "Att Line x", 'OFF': team_high_xop, 'ON': team_high_xip},
        {'Metric': "Spread", 'OFF': op_spread, 'ON': ip_spread},
        {'Metric': "Area", 'OFF': op_area, 'ON': ip_area}
    ]

    # Append each row to the DataFrame using pd.concat
    for row in data_rows:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Round the specified columns
    columns_to_round = ['OFF', 'ON']
    df[columns_to_round] = df[columns_to_round].round(2)

    # Calculate the difference percentage and format it
    df["DIFF %"] = (((df['ON'] - df['OFF']) / df['OFF']) * 100).round(2)
    if target_team == "away": # left to right so no need to adjust +/- signs
        df["DIFF %"] = df["DIFF %"].apply(lambda x: f"+{x}" if x > 0 else f"{x}")
    else:  # home : right to left so need to adjust +/- signs
        df.loc[:1, "DIFF %"] = df.loc[:1, "DIFF %"].apply(lambda x: f"-{abs(x)}" if x > 0 else f"+{abs(x)}")
        df.loc[2:, "DIFF %"] = df.loc[2:, "DIFF %"].apply(lambda x: f"+{x}" if x > 0 else f"{x}")

    # Create the statistics table
    create_stat_table(df)

    # Overlay the table onto the pitch
    im1 = plt.imread("images/table.png")
    if target_team == "home":
        ax_image = add_image(im1, fig, left=0.144, bottom=0.57, width=0.23, height=0.23)
    else:
        ax_image = add_image(im1, fig, left=0.65, bottom=0.57, width=0.23, height=0.23)
   
    if save == True:
        plt.savefig(f'images/{target_team}_possession.png', bbox_inches='tight', dpi=300) 


# Plot Heatmap for player in-possession and off-possession positions
def liga_player_heatmap(
    tidy_data: pd.DataFrame, 
    poss_data: pd.DataFrame, 
    target_team: str, 
    target_player: int, 
    pitch_fill: str = "black", 
    pitch_lines_col: str = "#7E7D7D",
    pitch_type: str = 'opta', 
    save: bool = True,
    home_team_col: str = "#0A97B0", 
    away_team_col: str = "#A04747"
    ) -> None:
    """
    Plot heatmaps for a player's in-possession and off-possession positions.

    Steps:
    -------
    1. Subset data for the target team and player.
    2. Determine possession frames for the team.
    3. Create heatmaps for in-possession and off-possession positions.
    4. Add direction of play and visualization credits.

    Parameters:
    -----------
    tidy_data : pd.DataFrame
        Processed tracking data containing player positions and other match details.
    poss_data : pd.DataFrame
        Data containing possession frames for the teams.
    target_team : str
        Team name to analyze ('home' or 'away').
    target_player : int
        Player number to analyze.
    pitch_fill : str, optional
        Background color of the pitch (default is 'black').
    pitch_lines_col : str, optional
        Color of the pitch lines (default is '#7E7D7D').
    pitch_type : str, optional
        Type of pitch layout (default is 'opta').
    save : bool, optional
        Whether to save the plot (default is True).
    home_team_col : str, optional
        Color representing the home team (default is '#0A97B0').
    away_team_col : str, optional
        Color representing the away team (default is '#A04747').

    Returns:
    --------
    None. Displays and optionally saves the heatmaps.
    """
    # Subset data of particular team                  
    stats_df = tidy_data[tidy_data["team"] == target_team].copy()
        
    # Find frames where a team has possession
    home_poss = poss_data[poss_data["Team"] == "Home"] 
    away_poss = poss_data[poss_data["Team"] == "Away"]

    # Store possession frames as list
    possession_frames_home = generate_possession_frames(home_poss)
    possession_frames_away = generate_possession_frames(away_poss)

    # Set new column to denote in possession or out possession
    if target_team == "home":
        stats_df['in_possession'] = np.where(stats_df['Frame'].isin(possession_frames_home), 1, 0)
        stats_df['out_possession'] = np.where(stats_df['Frame'].isin(possession_frames_away), 1, 0)
    else:
        stats_df['in_possession'] = np.where(stats_df['Frame'].isin(possession_frames_away), 1, 0)
        stats_df['out_possession'] = np.where(stats_df['Frame'].isin(possession_frames_home), 1, 0)

    # Get dataframe where team is in/out possession
    player_ip = stats_df[(stats_df["in_possession"] == 1) & (stats_df["player"] == target_player)]
    player_op = stats_df[(stats_df["out_possession"] == 1) & (stats_df["player"] == target_player)]

    # Define color based on team
    color = home_team_col if target_team == "home" else away_team_col

    # Define custom colormap
    from matplotlib.colors import LinearSegmentedColormap  
    req_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors", ['black', color], N=10)

    # Function to plot heatmap
    def plot_heatmap(player_data: pd.DataFrame, title: str) -> None:
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = plt.subplot(111)

        fig.set_facecolor(pitch_fill)
        ax.patch.set_facecolor(pitch_fill)

        pitch = Pitch(pitch_color=pitch_fill, pitch_type=pitch_type, goal_type='box', linewidth=0.85, line_color=pitch_lines_col)
        pitch.draw(ax=ax)

        pitch.kdeplot(player_data.x, player_data.y, ax=ax, fill=True, levels=50, thresh=0, cut=4, zorder=-1, cmap=req_cmap)

        fig_text(x=0.145, y=0.747, s=title, highlight_textprops=[{'color': color, 'weight': 'bold'}], 
                 va='bottom', ha='left', fontname="Century Gothic", 
                 weight='bold', fontsize=13, color='white')

        fig_text(x=0.77, y=0.74, s="Viz by : @Ligandro22", 
                va='bottom', ha='left',
                fontname="STXihei", weight='bold', 
                fontsize=8, color='white')

        if target_team == "home":  
            fig_text(x=0.503, y=0.237, s="Direction of Play", va='bottom', ha='left', 
                     fontname="Century Gothic", fontsize=9, color='white')
            plt.arrow(48, -2.8, -3, 0, shape='full', color='white', 
                    linewidth=4, head_width=0.2, head_length=0.2)
        else:            
            fig_text(x=0.42, y=0.237, s="Direction of Play", va='bottom', ha='left', 
                    fontname="Century Gothic", fontsize=9, color='white')
            plt.arrow(52, -2.8, 3, 0, shape='full', color='white', 
                     linewidth=4, head_width=0.2, head_length=0.2)


    # Plot in-possession heatmap Your existing code
    plot_heatmap(player_ip, f"In Possession Heatmap\nPlayer Number : <{target_player}>")

    # Plot off-possession heatmap        # Add a horizontal arrow at the specified location
    plot_heatmap(player_op, f"Off Possession Heatmap\nPlayer Number : <{target_player}>")


# Calculate player running data
def player_movement_stats(tidy_data: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Calculate player running statistics including distance covered in different speed ranges.

    Parameters:
    -----------
    tidy_data : pd.DataFrame
        Processed tracking data containing player positions and other match details.
    save : bool, optional
        Whether to save the output tables as images (default is True).

    Returns:
    --------
    pd.DataFrame
        DataFrame containing player running statistics.
    """
    range_names = ["walking_km", "jogging_km", "running_km", "sprinting_km"]
    
    # Remove ball data
    non_ball_df = tidy_data[tidy_data['team'] != 'ball']
    non_ball_df = non_ball_df.reset_index()

    # Find avg_speed, distance and number of frames by player
    player_df = non_ball_df.groupby(['team', 'player']).agg(
        n_samples=pd.NamedAgg(column='speed', aggfunc='count'),
        distance_km=pd.NamedAgg(column='v_mod', aggfunc=lambda x: (x / MS_LAG_SMOOTH).sum() / 1000),
        avg_speed_m_s=pd.NamedAgg(column='speed', aggfunc='mean')
    ).reset_index()

    # Calculate distances for different speed ranges
    for i, name in enumerate(range_names, 1):
        if i == 1:
            temp = non_ball_df[non_ball_df['speed'] < WALK_JOG_THRESHOLD].groupby(['team', 'player'])['v_mod'].sum() / (MS_LAG_SMOOTH * 1000)
        elif i == 2:
            temp = non_ball_df[(non_ball_df['speed'] >= WALK_JOG_THRESHOLD) & (non_ball_df['speed'] < JOG_RUN_THRESHOLD)].groupby(['team', 'player'])['v_mod'].sum() / (MS_LAG_SMOOTH * 1000)
        elif i == 3:
            temp = non_ball_df[(non_ball_df['speed'] >= JOG_RUN_THRESHOLD) & (non_ball_df['speed'] < RUN_SPRINT_THRESHOLD)].groupby(['team', 'player'])['v_mod'].sum() / (MS_LAG_SMOOTH * 1000)
        elif i == 4:
            temp = non_ball_df[non_ball_df['speed'] >= RUN_SPRINT_THRESHOLD].groupby(['team', 'player'])['v_mod'].sum() / (MS_LAG_SMOOTH * 1000)

        # Add to player_df for corresponding speed range
        player_df[name] = temp.reset_index(drop=True)

    # Calculate additional columns
    player_df['minutes_played'] = player_df['n_samples'] * MS_DT / 60
    player_df['avg_speed_km_h'] = player_df['avg_speed_m_s'] / 1000 * 3600

    # Sort the data
    output = player_df.sort_values(by=['team', 'minutes_played'], ascending=[True, False])

    output = output[['team', 'player', 'distance_km',
                     'walking_km', 'jogging_km', 'running_km', 'sprinting_km',
                     'minutes_played', 'avg_speed_m_s', 'avg_speed_km_h']]
    
    # Convert to float and round decimals
    output = output.astype({col: 'float' for col in output.columns if col not in ['team', 'player']}).round({col: 2 for col in output.columns if col not in ['team', 'player']})
    
    # Find home and away data sorted by most distance ran
    home_table = output[output["team"] == "home"]
    home_table = home_table.sort_values(by="minutes_played", ascending=False)
    away_table = output[output["team"] == "away"]
    away_table = away_table.sort_values(by="minutes_played", ascending=False)

    # Plot home table
    fig, ax = plt.subplots(figsize=(16, 3.8)) 
    ax.axis('off')  
    fig.set_facecolor("#dedede")

    table = ax.table(
        cellText=home_table.values,  
        colLabels=home_table.columns, 
        loc='center',
        cellLoc='center'  
    )

    for key, cell in table.get_celld().items():
        cell.set_facecolor("#dedede") 
        cell.set_text_props(color='black', fontname='Century Gothic') 

    table.auto_set_font_size(False)
    table.set_fontsize(12)  
    table.scale(1.2, 1.2) 

    if save:
        plt.savefig('images/home_player_table.png', bbox_inches='tight', dpi=300) 
        
    # Plot away table
    fig, ax = plt.subplots(figsize=(16, 3.8))  
    ax.axis('off')  
    fig.set_facecolor("#dedede")

    table = ax.table(
        cellText=away_table.values,  
        colLabels=away_table.columns, 
        loc='center',  
        cellLoc='center'  
    )

    for key, cell in table.get_celld().items():
        cell.set_facecolor("#dedede")  
        cell.set_text_props(color='black', fontname='Century Gothic') 

    table.auto_set_font_size(False)
    table.set_fontsize(12) 
    table.scale(1.2, 1.2)  
    
    if save:
        plt.savefig('images/away_player_table.png', bbox_inches='tight', dpi=300) 

    return output


# Plot graph of player speed
def liga_player_stats_graph(player_stats: pd.DataFrame, team: str, save: bool = True) -> None:
    """
    Plot a graph of player speed statistics for a given team.

    Parameters:
    -----------
    player_stats : pd.DataFrame
        DataFrame containing player statistics including distance covered in different speed ranges.
    team : str
        Team name to plot the statistics for ('home' or 'away').
    save : bool, optional
        Whether to save the plot as an image (default is True).

    Returns:
    --------
    None. Displays and optionally saves the plot.
    """
    df = player_stats[player_stats["team"] == team].sort_values(by="distance_km", ascending=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(13, 8))

    # Set background color to black
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Define the segments for the stacked bar chart
    categories = ['walking_km', 'jogging_km', 'running_km', 'sprinting_km']

    # Plot each segment horizontally
    left = np.zeros(len(df))

    for category in categories:
        ax.barh(df["player"], df[category], left=left, label=category.replace('_km', ''), zorder=2)
        left += df[category]

    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7, zorder=-1)

    ax.set_yticks(df["player"])

    # Customize fonts
    ax.set_title(f'Player distance covered in km - {team.capitalize()}', font="Century Gothic", fontsize=24, fontweight='bold', color='white')

    plt.xlabel('Distance (km)', font='Century Gothic', fontsize=14, fontweight='bold', color="white")
    plt.ylabel('Player', font='Century Gothic', fontsize=14, fontweight='bold', color="white")

    plt.xticks(fontname="Century Gothic", fontsize=12, color="white")
    plt.yticks(fontname="Century Gothic", fontsize=12, color="white")

    ax.tick_params(axis='both', colors='white')
    ax.legend(title='Speed Range', loc='upper right', fontsize=10, facecolor='white', edgecolor='white')

    str_t = "Viz by : @Ligandro22"
    fig_text(
        x=0.78, y=0.923,
        s=str_t,
        va='bottom', ha='left',
        fontname="STXihei", weight='bold',
        fontsize=10, color='white'
    )

    if save:
        plt.savefig(f'images/{team}_player_speed_graph.png', bbox_inches='tight', dpi=300)

    # Display the plot
    plt.tight_layout()
    plt.show()


# Find sprints data for player
def sprints_info(tidy_data: pd.DataFrame, target_team: str = "home") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find sprints data for players in a team.

    Steps:
    -------
    1. Subset data for the target team.
    2. Remove rows with no recorded speed.
    3. Determine if each row is a sprint based on speed threshold.
    4. Group by player and team, then apply convolution to identify sprints.
    5. Determine start and end of sprints.
    6. Concatenate results and subset frames where sprint starts or ends.
    7. Remove wrong sprints by pairing start and end frames.
    8. Calculate number of sprints by each player.

    Parameters:
    -----------
    tidy_data : pd.DataFrame
        Processed tracking data containing player positions and other match details.
    target_team : str, optional
        Team name to analyze ('home' or 'away').

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        DataFrame containing number of sprints by each player and DataFrame containing sprint start/end frames.
    """
    # Subset data for team
    filtered_data = tidy_data[tidy_data['team'] == target_team]

    # Remove rows with no recorded speed
    conv_data = filtered_data[filtered_data['speed'].notna()]

    # Determine if each row is a sprint
    conv_data['is_sprint'] = np.where(conv_data['speed'] >= RUN_SPRINT_THRESHOLD, 1, 0)

    # Group by player and team, then apply convolution
    grouped = conv_data.groupby(['player', 'team'])
    result = []

    for (player, team), group in grouped:
        group = group.reset_index(drop=True)
        conv_result = convolve1d(group['is_sprint'], np.ones(SPRINTS_WINDOW), mode='constant', cval=0.0)
        group['conv'] = np.round(conv_result).astype(int)

        # Determine start and end of sprints
        group['start'] = np.where((group['conv'] == SPRINTS_WINDOW) & (group['conv'].shift(1) != SPRINTS_WINDOW), 1, 0)
        group['end'] = np.where((group['conv'] == 0) & (group['conv'].shift(SPRINTS_WINDOW) == SPRINTS_WINDOW), 1, 0)
        group['n'] = range(1, len(group) + 1)

        # Select relevant columns
        result.append(group[['player', 'team', 'start', 'end', 'n']])

    # Concatenate results
    conv = pd.concat(result).reset_index(drop=True)
    
    # Subset frames where sprint starts or ends
    sprints_data = conv[(conv["start"] == 1) | (conv["end"] == 1) ]
    
    # Remove wrong sprints by pairing one row with next as sprints always have a start frame and end frame
    sprints_data =sprints_data.reset_index(drop=True)
    
    # We know, a sprint will have 2 rows to denote start and end
    sprints_data['keep'] = True

    # Iterate through the DataFrame to find unpaired starts
    for i in range(len(sprints_data) - 1):
        if sprints_data.iloc[i]['start'] == 1 and sprints_data.iloc[i]['end'] == 0:
            # Check if the next row is a valid end
            if not (sprints_data.iloc[i + 1]['start'] == 0 and sprints_data.iloc[i + 1]['end'] == 1):
                sprints_data.at[i, 'keep'] = False

    # Check the last row separately
    if len(sprints_data) % 2 != 0 and sprints_data.iloc[-1]['start'] == 1 and sprints_data.iloc[-1]['end'] == 0:
        sprints_data.at[-1, 'keep'] = False
    

    # Corrected data
    sprints_data = sprints_data[sprints_data["keep"] == True]

    # Calculate number of sprints by each player
    player_sprints = (
    conv.groupby('player')
           .agg(no_of_sprints=('start', 'sum'))
           .reset_index()
           .sort_values(by='no_of_sprints', ascending=False)
    )
    return player_sprints ,sprints_data  


# Plot sprints for a specific player
def liga_plot_player_sprints(
    tidy_data: pd.DataFrame,
    sprints_data: pd.DataFrame,
    target_team: str,
    target_player: int,
    home_team_col: str = "#0A97B0",
    away_team_col: str = "#A04747",
    pitch_fill: str = "black",
    pitch_lines_col: str = "#7E7D7D",
    pitch_type: str = 'opta',
    save: bool = False
    ) -> None:
    """
    Plot sprints for a specific player.

    Steps:
    -------
    1. Determine the color based on the team.
    2. Find frames where the player starts and ends sprints.
    3. Create a DataFrame for player sprints.
    4. Create a list of frames where the player is sprinting.
    5. Subset the tracking data for the required frames.
    6. Create the pitch and plot player positions.
    7. Add text and markers for sprint start and end frames.
    8. Add direction of play and visualization credits.
    9. Save the plot if required.

    Parameters:
    -----------
    tidy_data : pd.DataFrame
        Processed tracking data containing player positions and other match details.
    sprints_data : pd.DataFrame
        Data containing sprint start/end frames for players.
    target_team : str
        Team name to analyze ('home' or 'away').
    target_player : int
        Player number to analyze.
    home_team_col : str, optional
        Color representing the home team (default is '#0A97B0').
    away_team_col : str, optional
        Color representing the away team (default is '#A04747').
    pitch_fill : str, optional
        Background color of the pitch (default is 'black').
    pitch_lines_col : str, optional
        Color of the pitch lines (default is '#7E7D7D').
    pitch_type : str, optional
        Type of pitch layout (default is 'opta').
    save : bool, optional
        Whether to save the plot (default is False).

    Returns:
    --------
    None. Displays and optionally saves the plot.
    """
    color_main = home_team_col if target_team == "home" else away_team_col

    # Find frames where player starts and ends sprints
    player_starts = sprints_data[(sprints_data["player"] == target_player) & (sprints_data["start"] == 1)]
    player_ends = sprints_data[(sprints_data["player"] == target_player) & (sprints_data["end"] == 1)]

    # Create player sprints DataFrame
    player_sprints = player_starts.copy()
    player_sprints.rename(columns={'n': 'frame_start'}, inplace=True)
    player_sprints["frame_end"] = player_ends["n"].values

    # Create a list of frames where the player is sprinting
    frame_list = player_sprints.apply(lambda row: list(range(row['frame_start'], row['frame_end'] + 1)), axis=1)
    total_req_frames = [frame for sublist in frame_list for frame in sublist]

    # Subset the tracking data for the required frames
    player_df = tidy_data[tidy_data["player"] == target_player]
    player_df = player_df[player_df["Frame"].isin(total_req_frames)]

    # Create the pitch
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = plt.subplot(111)

    fig.set_facecolor(pitch_fill)
    ax.patch.set_facecolor(pitch_fill)
    pitch = Pitch(pitch_color=pitch_fill, pitch_type=pitch_type, goal_type='box', linewidth=0.85, line_color=pitch_lines_col)
    pitch.draw(ax=ax)

    plt.scatter(player_df['x'], player_df['y'], s=0.5, alpha=1, c=color_main, marker=".", zorder=3)

    for index, row in player_df.iterrows():
        if row['Frame'] in list(player_starts["n"]):
            plt.text(row['x'], row['y'] + 1.2, f"{row['minutes']:.2f}", font="Century Gothic", fontsize=7,
                     ha='center', va='center', weight="bold", color='white', zorder=3)
            plt.scatter(row['x'], row['y'], s=7, alpha=1, c="white", marker="o", zorder=3)
        if row['Frame'] in list(player_ends["n"]):
            plt.scatter(row['x'], row['y'], s=5, alpha=1, c=color_main, marker="x", zorder=3)

    str_text = f"Sprints by Player Number <{target_player}>"
    fig_text(x=0.145, y=0.765, s=str_text, highlight_textprops=[{'color': color_main, 'weight': 'bold'}],
             va='bottom', ha='left', fontname="Century Gothic", weight='bold', fontsize=13, color='white')

    str_text = "Label denotes Time when sprints start"
    fig_text(x=0.145, y=0.745, s=str_text, va='bottom', ha='left', fontname="Century Gothic", fontsize=10, color='white')

    str_t = "Viz by : @Ligandro22"
    fig_text(x=0.77, y=0.74, s=str_t, va='bottom', ha='left', fontname="STXihei", weight='bold', fontsize=8, color='white')

    if target_team == "home":
        fig_text(x=0.503, y=0.237, s="Direction of Play", va='bottom', ha='left', fontname="Century Gothic", fontsize=9, color='white')
        arrow_location = (48, -2.8)
        plt.arrow(arrow_location[0], arrow_location[1], -3, 0, shape='full', color='white', linewidth=4, head_width=0.2, head_length=0.2)
    else:
        fig_text(x=0.42, y=0.237, s="Direction of Play", va='bottom', ha='left', fontname="Century Gothic", fontsize=9, color='white')
        arrow_location = (52, -2.8)
        plt.arrow(arrow_location[0], arrow_location[1], 3, 0, shape='full', color='white', linewidth=4, head_width=0.2, head_length=0.2)

    if save:
        plt.savefig(f'images/player_{target_player}_sprints.png', bbox_inches='tight', dpi=300)

    plt.show()


# Function to generate animation of range of sprint frames
def liga_sprint_animate(
    tidy_data: pd.DataFrame,
    poss_data : pd.DataFrame,
    target_player: int,
    frame_start: int,
    frame_end: int,
    save: bool = True,
    video_writer: str = "gif",
    pitch_fill: str = "black",
    pitch_lines_col: str = "#7E7D7D",
    pitch_type: str = 'opta',
    home_team_col: str = "#0A97B0",
    away_team_col: str = "#A04747"
    ) -> None:
    """
    Generate an animation of a range of frames highlighting a player's sprints.

    Parameters:
    -----------
    tidy_data : pd.DataFrame
        Processed tracking data containing player positions and other match details.
    poss_data : pd.DataFrame
        Data containing possession frames for the teams.
    target_player : int
        Player number to highlight in the animation.
    frame_start : int
        The starting frame number for the animation.
    frame_end : int
        The ending frame number for the animation.
    save : bool, optional
        Whether to save the animation (default is True)
    video_writer : str, optional
        Format for saving the animation (default is 'gif'). Other option is 'mp4'.
    pitch_fill : str, optional
        Background color of the pitch (default is 'black').
    pitch_lines_col : str, optional
        Color of the pitch lines (default is '#7E7D7D').
    pitch_type : str, optional
        Type of pitch layout (default is 'opta').
    home_team_col : str, optional
        Color representing the home team (default is '#0A97B0').
    away_team_col : str, optional
        Color representing the away team (default is '#A04747').

    Returns:
    --------
    None. Saves the animation as a video file.
    """
    # Subset frame range data
    frame_df = tidy_data[(tidy_data["Frame"] > frame_start) & (tidy_data["Frame"] < frame_end)]
    player_sprint_frames = frame_df[frame_df["player"] == target_player]
    frame_df = frame_df.groupby('Frame').apply(player_possession).reset_index(drop=True)  # Determine ball possession
    frame_df['edgecolor'] = frame_df['has_possession'].apply(lambda x: "yellow" if x else "white")
    frame_df['edge_lw'] = frame_df['has_possession'].apply(lambda x: 1.8 if x else 0.5)

    team_colors = {
        'ball': 'white',
        'away': away_team_col,
        'home': home_team_col
    }

    team_markers = {
        'ball': 'o',
        'away': 'o',
        'home': 'o'
    }

    # Map the team values to colors and markers
    frame_df['color'] = frame_df['team'].map(team_colors)
    frame_df['marker'] = frame_df['team'].map(team_markers)

    # Create the pitch
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    fig.set_facecolor(pitch_fill)
    ax.patch.set_facecolor(pitch_fill)
    pitch = Pitch(pitch_color=pitch_fill, pitch_type=pitch_type, goal_type='box', linewidth=0.85, line_color=pitch_lines_col)
    pitch.draw(ax=ax)

    # Function to update the plot for each frame
    def sprint_plot(frame: int) -> None:
        ax.clear()
        pitch.draw(ax=ax)
        frame_data = frame_df[frame_df['Frame'] == frame]
        plt.scatter(player_sprint_frames['x'], player_sprint_frames['y'], s=25, alpha=1, facecolors='white', edgecolors="green", marker=".", linewidths=1.5, zorder=3)

        for team, marker in team_markers.items():
            team_df = frame_data[frame_data['team'] == team]
            if team == 'ball':
                plt.scatter(team_df['x'], team_df['y'], s=50, alpha=1, facecolors='none', edgecolors="yellow", marker=marker, linewidths=1.5, zorder=3)
            else:
                plt.scatter(team_df['x'], team_df['y'], s=250, alpha=1, c=team_df['color'], edgecolors=team_df['edgecolor'], marker=marker, linewidths=team_df['edge_lw'], label=team, zorder=3)

        # Add player number and speed with direction
        for _, row in frame_data.iterrows():
            dx, dy, speed = row['dx'], row['dy'], row['speed'] * 0.35
            if row['player'] != 0:
                plt.text(row['x'], row['y'], str(row['player']), font="Century Gothic", fontsize=9, ha='center', va='center', weight="bold", color='white', zorder=4)
                plt.arrow(row['x'], row['y'], dx * speed, dy * speed, head_width=0.5, head_length=0.5, fc='white', ec='white', zorder=2)
            else:
                magnitude = np.sqrt(dx**2 + dy**2)
                plt.arrow(row['x'], row['y'], dx / magnitude * 2, dy / magnitude * 2, head_width=0.5, head_length=0.5, fc='yellow', ec='yellow', zorder=2)

        # Add legend with custom font
        plt.legend(
            title='Team',
            bbox_to_anchor=(1.045, 0.8),
            loc='center',
            handletextpad=0.5,
            labelspacing=1.0,
            prop={'family': 'Century Gothic', 'size': 12},
            title_fontproperties={'family': 'Century Gothic', 'size': 12},
            borderaxespad=0.3,
            borderpad=0.17,
            frameon=True
        )

        # Add clock time
        minute_value = frame_data['minutes'].values[0]
        minutes = int(minute_value)
        seconds = int((minute_value - minutes) * 60)
        str_text = f"Time - <{minutes:02d}:{seconds:02d}>"

        fig_text(
            x=0.425, y=0.76,
            s=str_text, highlight_textprops=[{'color': '#FFD230', 'weight': 'bold'}],
            va='bottom', ha='left', fontname="Century Gothic", weight='bold',
            fontsize=25, color='white', path_effects=[path_effects.Stroke(linewidth=0.8, foreground="#BD8B00"), path_effects.Normal()]
        )

        # Find current ball possession
        possession = team_possession(frame, poss_data)
        color_now = home_team_col if possession == "Home" else away_team_col if possession == "Away" else "white"
        str_text1 = f"Current Ball Possession: <{possession}> "
        fig_text(
            x=0.41, y=0.74,
            highlight_textprops=[{'color': color_now, 'weight': 'bold'}],
            s=str_text1,
            va='bottom', ha='left',
            fontname="Century Gothic", weight='bold',
            fontsize=12, color='white'
        )

        str_t = "Viz by : @Ligandro22"
        fig_text(
            x=0.77, y=0.74,
            s=str_t,
            va='bottom', ha='left',
            fontname="STXihei", weight='bold',
            fontsize=8, color='white'
        )

    ani = FuncAnimation(fig, sprint_plot, frames=frame_df['Frame'].unique(), repeat=False)
    if save:
        if video_writer =="gif":
            ani.save(f'videos/{target_player}_sprint_animation.gif', writer='pillow', fps=25, dpi=100)
        elif video_writer =="mp4":
            ani.save(f'videos/{target_player}_sprint_animation.mp4', writer='ffmpeg', fps=25, bitrate=2000, dpi=100)










































































































































































































































































































































































































