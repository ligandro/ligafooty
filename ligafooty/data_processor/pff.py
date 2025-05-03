# data_processor/metrica_processor.py
import pandas as pd
import numpy as np
from ligafooty.utils.constants import MS_LAG_SMOOTH, MS_DT, PLAYER_MAX_SPEED, WALK_JOG_THRESHOLD, JOG_RUN_THRESHOLD, RUN_SPRINT_THRESHOLD, SPRINTS_WINDOW
from .base import TrackingDataProcessor
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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from kloppy import pff
import json

pd.set_option('future.no_silent_downcasting', True)

class PFFTrackingDataProcessor():
    def process(self,
        meta_data_json_file: str,
        roster_meta_data_json_file : str,
        raw_data_json_file : str,
        add_velocity: bool = True,
        pitch_long: int = 100, 
        pitch_width: int = 100):
        """
        Clean and process PFF tracking data from provider files.

        Steps:
        -------
        1. Transform data to row format.
        6. Process ball data.
        7. Combine player and ball data.
        8. Transform coordinates for the second half.
        9. Add velocity data if requested.
        10. Convert time to minute:second format

        Args:
            track_df : pd.DataFrame,  Raw tracking data with multi-row headers.
            player_df : pd.DataFrame,  Player data with player ID and name.
            add_velocity: Whether to calculate velocity
            pitch_long: Pitch length
            pitch_width: Pitch width
            
        Returns:
            Processed tracking data DataFrame 
            Goalkeeper shirt numbers
            Team mapping dictionary
            Player DataFrame
    
        """
        # Read and clean the data
        dataset = pff.load_tracking(meta_data=meta_data_json_file,
                     roster_meta_data=roster_meta_data_json_file,
                     raw_data = raw_data_json_file,
                     # Optional Parameters
                     coordinates = "pff",
                     sample_rate = None,
                     limit = None)
        
        track_df =dataset.to_df()

        with open(roster_meta_data_json_file, "r") as file:
            data = json.load(file)

        player_df = pd.DataFrame([{
            'player_id': item['player']['id'],
            'nickname': item['player']['nickname'],
            'position': item['positionGroupType'],
            'shirt_number': item['shirtNumber'],
            'team_name': item['team']['name'],
            'team_id': item['team']['id'],
            'started' : item['started']
        } for item in data])


        # Process  data
        tidy_data= self._clean_columns(track_df) 

        
        # Remove players not in frame
        tidy_data =tidy_data.dropna(subset=['x', 'y'])

        # Map PFF coordinates to pitch coordinates
        tidy_data["x"] = ((tidy_data["x"] - (-52.5)) / (52.5 - (-52.5))) * pitch_long
        tidy_data["y"] = ((tidy_data["y"] - (-34)) / (34 - (-34))) * pitch_width

        # Inverse x,y for 2nd half period
        tidy_data['x'] = np.where(tidy_data['Period'] == 1, tidy_data['x'],  (100 - tidy_data['x']))
        tidy_data['y'] = np.where(tidy_data['Period'] == 1, tidy_data['y'], (100 - tidy_data['y']))

        tidy_data['x'] = tidy_data['x'] .round(2)
        tidy_data['y'] = tidy_data['y'] .round(2)

        # Add velocity if requested
        if add_velocity:
            tidy_data["dx"] = tidy_data.groupby("player_id")["x"].diff(MS_LAG_SMOOTH)
            tidy_data["dy"] = tidy_data.groupby("player_id")["y"].diff(MS_LAG_SMOOTH) 
            tidy_data["v_mod"] = np.sqrt(tidy_data["dx"]**2 + tidy_data["dy"]**2)
            tidy_data["speed"] = np.minimum(tidy_data["v_mod"] / (MS_DT * MS_LAG_SMOOTH), PLAYER_MAX_SPEED)
        
        # Convert time to minutes
        tidy_data['minutes'] = tidy_data['time'] // 60 + (tidy_data['time'] % 60) / 100
        
        # Now, to get player and team names, we need to merge with player_df

        # Convert datatype
        player_df["player_id"] = player_df["player_id"].astype(int)

        # Find Golakeeper shirt numbers
        gk_shirt_numbers = player_df.loc[(player_df["position"] == "GK") & (player_df["started"] == True), ["team_name", "shirt_number"]]

        # Merge player data 
        tidy_data =  tidy_data.merge(player_df[["player_id","nickname","shirt_number","team_name"]], 
                                    how='left', left_on='player_id', 
                                        right_on='player_id')

        # Set ball shirt number to 0
        tidy_data.loc[tidy_data["player_id"] == 0, "shirt_number"] = 0  

        # Map team names from team_id
        mapping = player_df.drop_duplicates("team_id").set_index("team_id")["team_name"]
        tidy_data["possession"] = tidy_data["possession"].map(mapping)

        # Map team names to home and away
        unique_teams = tidy_data["team_name"].dropna().unique()
        team_mapping = {unique_teams[0]: "home", unique_teams[1]: "away"}

        tidy_data["team_name"] = tidy_data["team_name"].map(team_mapping)
        tidy_data["possession"] = tidy_data["possession"].map(team_mapping)
        
        # Convert datatype
        tidy_data["team_name"] = tidy_data["team_name"].astype("string")
        tidy_data["possession"] = tidy_data["possession"].astype("string")
        tidy_data["shirt_number"] = tidy_data["shirt_number"].astype("int")
        
        # Rename columns
        tidy_data = tidy_data.rename(columns={"team_name": "team",
                            "shirt_number": "player"})

        # Set ball team  to "ball"
        tidy_data.loc[tidy_data["player"] == 0, "team"] = "ball"  

        # Drop DUPLICATE columns
        tidy_data = tidy_data.drop_duplicates(subset=['player_id', 'Frame'])
        tidy_data =tidy_data.sort_values(by=['Frame']) # Sort by Frame

        tidy_data['time'] = pd.to_timedelta(tidy_data['time'])

        # Format as "MM:SS"
        tidy_data['time'] = tidy_data['time'].dt.components.minutes.astype(str).str.zfill(2) + ":" + \
                            tidy_data['time'].dt.components.seconds.astype(str).str.zfill(2)

        # Map team_name of Gk to home and away
        gk_shirt_numbers["team_name"] = gk_shirt_numbers["team_name"].map(team_mapping)

        return tidy_data,gk_shirt_numbers,team_mapping,player_df


    def _clean_columns(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and formats column names for PFF tracking data.

        Steps:
        -------
        1. Finds non player column names
        2. Finds the player column names which are in the format of 'player_id_x' and 'player_id_y'.
        3. Creates a new DataFrame by appending new row for each player x and y coordinates per frame
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
        
        # Identify non-player columns (e.g., 'period_id', 'timestamp', etc.)
        non_player_cols = [col for col in df.columns if not any(c.isdigit() for c in col)]
        
        # Identify unique player IDs by extracting numeric prefixes from column names
        player_ids = sorted(set(col.split('_')[0] for col in df.columns if any(c.isdigit() for c in col)))
        
        # Create a new DataFrame to store reshaped data
        new_rows = []

        for _, row in df.iterrows():
            # Add a row for each player
            for player_id in player_ids:
                player_x = row.get(f"{player_id}_x", None)
                player_y = row.get(f"{player_id}_y", None)

                new_row = {col: row[col] for col in non_player_cols}  # Copy non-player columns
                new_row["player_id"] = int(player_id)  # Convert to integer for consistency
                new_row["x"] = player_x
                new_row["y"] = player_y

                new_rows.append(new_row)

            # Add a row for the ball with player_id = 0
            new_row = {col: row[col] for col in non_player_cols}  # Copy non-player columns
            new_row["player_id"] = 0  # Assign player_id as 0 for the ball
            new_row["x"] = row.get("ball_x", None)
            new_row["y"] = row.get("ball_y", None)

            new_rows.append(new_row)

        # Convert list of dictionaries to DataFrame
        df_final = pd.DataFrame(new_rows)

        # Convert column datatype
        df_final["player_id"] = df_final["player_id"].astype(int)
        
        # Remove uneccessary columns
        df_final = df_final.drop(columns=["ball_x", "ball_y", "ball_z", "ball_speed"])
        
        # Rename columns
        df_final = df_final.rename(columns={"period_id": "Period", "frame_id": "Frame", "timestamp": "time",
                                "ball_owning_team_id": "possession"})
        return df_final

    

    

    def _player_possession(self,frame_data: pd.DataFrame) -> pd.DataFrame:
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

        # Find coordinates of the ball
        ball_data = frame_data[frame_data['team'] == 'ball']
        if ball_data.empty:
            return frame_data.assign(has_possession=False)

        ball_x = ball_data['x'].values[0]
        ball_y = ball_data['y'].values[0]

        # Calculate distances for each player to the ball
        players_data = frame_data[frame_data['player'] != 0]
        players_data['distance'] = np.sqrt((players_data['x'] - ball_x)**2 + (players_data['y'] - ball_y)**2)

        # Determine the player with the smallest distance to the ball
        min_distance_index = players_data['distance'].idxmin()
        min_distance = players_data.loc[min_distance_index, 'distance']
        
        MIN_THRESHOLD_DISTANCE = 1  # Set min distance as 1 meter
        players_data['has_possession'] = False
        if min_distance <= MIN_THRESHOLD_DISTANCE:  # Assign ball possession only if min distance is within threshold
            players_data.loc[min_distance_index, 'has_possession'] = True

        # Combine the ball data with the players data
        frame_data = pd.concat([ball_data, players_data]).sort_index()
        frame_data['has_possession'] = frame_data['has_possession'].fillna(False)

        return frame_data

    def plot_single_frame(
        self,
        tidy_data: pd.DataFrame,
        target_frame: int,
        gk_df: pd.DataFrame,
        method: str = "base", 
        text_font: str="Century Gothic",
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
        tidy_data : DataFrame
            Tracking data 
        target_frame : int
            The specific frame number to visualize
        gk_df : DataFrame
            Data about Goalkeeper numbers
        method : str, optional
            Visualization method: 'base', 'convexhull', 'delaunay', or 'voronoi'
        text_font : str,optional
            Text Font
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
        frame_df = tidy_data[tidy_data["Frame"] == target_frame]
        frame_df = frame_df.groupby('Frame').apply(self._player_possession).reset_index(drop=True)
        if 'has_possession' not in frame_df.columns:
            frame_df['has_possession'] = False  # Default value

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
                    pitch_type=pitch_type,
                    goal_type='box',
                    linewidth=0.85,
                    line_color=pitch_lines_col)
        pitch.draw(ax=ax)

        # Plot the scatter points with different colors and markers
        if method == "base":
            self._plot_base_viz(frame_df, team_colors,gk_df)
            
        elif method == "convexhull":
            self._plot_convexhull_viz(frame_df,pitch,ax,team_colors,gk_df)
            
        elif method == "delaunay":
            self._plot_delaunay_viz(frame_df,pitch,ax,team_colors,gk_df)
            
        elif method == "voronoi":
            self._plot_voronoi_viz(frame_df,pitch,ax,team_colors,gk_df)
            

        # Plot ball
        team_df = frame_df[frame_df['team'] == "ball"]
        plt.scatter(team_df['x'], team_df['y'], s=50, alpha=1, facecolors='none', edgecolors="yellow", marker="o", linewidths=1.5,zorder=2)

    
        #  Find defensive line and pitch width holded
        home_width, home_defensive_line,home_def_line_height = self._calculate_team_stats_pff(frame_df[frame_df['team'] == 'home'], 'home',gk_df)
        away_width, away_defensive_line,away_def_line_height = self._calculate_team_stats_pff(frame_df[frame_df['team'] == 'away'], 'away',gk_df)
        ax.plot([home_defensive_line, home_defensive_line], [0,100], lw=1, color=home_team_col,  linestyle='--',zorder=1)
        ax.plot([away_defensive_line, away_defensive_line], [0,100], lw=1, color=away_team_col,  linestyle='--',zorder=1)
        
        max_arrow_length = 10 
        # Add player number and speed with direction
        for index, row in frame_df.iterrows():
            dx = row['dx']
            dy = row['dy']
            speed = row['speed']*0.35
            if row['player'] != 0:
                plt.text(row['x'], row['y'], str(row['player']),font=text_font ,fontsize=9, ha='center', va='center', weight="bold",color='white',zorder=3)
                dx = dx*speed
                dy= dy*speed
                plt.arrow(row['x'], row['y'], dx, dy, head_width=0.5, head_length=0.5,
                        fc='white', ec='white', zorder=2)
                        
            else:
                magnitude = np.sqrt(dx**2 + dy**2)
                plt.arrow(row['x'], row['y'], dx/magnitude*2, dy/magnitude*2, head_width=0.5, head_length=0.5, fc='yellow', ec='yellow',zorder=2)

        legend_font = {
            'family': text_font,
            'size': 12,
        }

        # Define custom legend elements
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Home',
                markerfacecolor=home_team_col, markersize=14),
            Line2D([0], [0], marker='o', color='w', label='Away',
                markerfacecolor=away_team_col, markersize=14),
            Line2D([0], [0], marker='o', color='w', label='Ball',
                markerfacecolor='black', markersize=10, markeredgecolor='yellow',markeredgewidth=2.5)
        ]

        # Add custom legend
        ax.legend(
            handles=legend_elements,
            title='Team',
            bbox_to_anchor=(1.045, 0.8),
            loc='center',
            handletextpad=0.5,
            labelspacing=1.0,
            prop=legend_font,
            title_fontproperties=legend_font,
            borderaxespad=0.3,
            borderpad=0.17,
            frameon=True
        )
        
        # Add clock time
        time_value = frame_df['time'].values[0]

        if frame_df["Period"].values[0] == 2:
            minutes, seconds = map(int, time_value.split(":"))
            minutes += 45
            time_value = f"{minutes:02}:{seconds:02}"

        str_text = f"Time - <{time_value}>"

        fig_text(
            x = 0.42, y = 0.79, 
            s = str_text,highlight_textprops=[{'color':'#FFD230', 'weight':'bold'}],
            va = 'bottom', ha = 'left',fontname =text_font, weight = 'bold',
            fontsize = 25,color ='white',path_effects=[path_effects.Stroke(linewidth=0.8, foreground="#BD8B00"), path_effects.Normal()]
        )


        # Find current ball possesion
        possession = frame_df.loc[0,"possession"]
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
            fontname=text_font, weight='bold',
            fontsize=12, color='white'
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
            fontname=text_font, weight='bold',
            fontsize=11, color='white'
        )

        str_text2 = f"Defensive Line Height <Away> : {away_def_line_height:.2f} | <Home> : {home_def_line_height:.2f} "
        fig_text(
            x=0.32, y=0.23,
            highlight_textprops=[
                {'color': away_team_col, 'weight': 'bold'},  # Properties for <Home>
                {'color': home_team_col, 'weight': 'bold'}   # Properties for <Away>
            ],
            s=str_text2,
            va='bottom', ha='left',
            fontname=text_font, weight='bold',
            fontsize=11, color='white'
        )

        if save == True :
            plt.savefig(f"images/frame_{target_frame}.jpg",dpi =500, bbox_inches='tight')

        plt.show()
    



    def _calculate_team_stats_pff(self,team_df: pd.DataFrame, team_name: str,gk_df :pd.DataFrame) -> tuple:
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
        gk_df : DataFrame
            Data about Goalkeeper numbers
        Returns:
        --------
        tuple
            (pitch width, last defender's x-coordinate, distance between goalkeeper and last defender).
        """
        if team_name == 'home':
            keeper_no = int(gk_df[gk_df['team_name'] == 'home']['shirt_number'].values[0])
            keeper_df = team_df[team_df['player'] ==keeper_no]
            team_df = team_df[team_df['player'] != keeper_no] 
        elif team_name == 'away':
            keeper_no = int(gk_df[gk_df['team_name'] == 'away']['shirt_number'].values[0])
            keeper_df = team_df[team_df['player'] ==keeper_no] 
            team_df = team_df[team_df['player'] != keeper_no]  
            
        # Calculate width (distance between two farthest players in y-axis)
        min_y = team_df['y'].min()
        max_y = team_df['y'].max()
        width = max_y - min_y

        # Calculate defensive line (x-coordinate of the last player)
        if team_name == 'home':
            sorted_players = team_df.sort_values(by='x', ascending=True)  # Sort by x ascending for home
        else:
            sorted_players = team_df.sort_values(by='x', ascending=False)  # Sort by x descending for away

        def_len=  abs(keeper_df['x'] - sorted_players.iloc[-1]['x'] ) # Distance b/w keeper and last defender

        return width, sorted_players.iloc[-1]['x'], def_len.values[0] 
    
    def _plot_base_viz(
        self,
        frame_df: pd.DataFrame,
        team_colors: dict,
        gk_df: pd.DataFrame
        ) -> None:
        """
        Draw the base visualization for player positions on a pitch.

        Parameters:
        -----------
        frame_df : pd.DataFrame
            DataFrame containing tracking data for a specific frame.
        team_colors : dict
            Dictionary mapping team names to color for plotting.
        gk_df : DataFrame
            Data about Goalkeeper numbers
        Returns:
        --------
        None, plots the player positions on the pitch.
        """
        for team, color in team_colors.items():
            if team != 'ball':
                team_df = frame_df[frame_df['team'] == team]
                plt.scatter(
                    team_df['x'], team_df['y'], 
                    s=250, alpha=1, c=color, 
                    edgecolors=team_df['edgecolor'], marker="o", 
                    linewidths=team_df['edge_lw'], label=team, zorder=3
                )

    def _plot_convexhull_viz(
        self,
        frame_df: pd.DataFrame,
        pitch, ax,
        team_colors: dict,
        gk_df: pd.DataFrame
        ) -> None:
        """
        Draws the convex hull visualization for player positions on a pitch.

        Parameters:
        ----------
        frame_df : pd.DataFrame
            DataFrame containing tracking data for a specific frame.
        pitch : mplsoccer.Pitch
            The pitch object used for plotting.
        ax : matplotlib.axes.Axes
            The axes object where the visualization will be drawn.
        team_colors : dict
            Dictionary mapping team names to color for plotting.
        gk_df : DataFrame
            Data about Goalkeeper numbers
        Returns:
        --------
        None, plots the player positions on the pitch.
        """
        gk_numbers = list(map(int, gk_df["shirt_number"]))
        # Iterate over each team
        for team, color in team_colors.items():
            if team != 'ball':
                team_df = frame_df[frame_df['team'] == team] # Filter out the team data
                if not team_df.empty:
                    # Exclude goalkeepers for convex hull calculation
                    hull_df = team_df[~team_df['player'].isin(gk_numbers)]

                    if not hull_df.empty:
                        hull = pitch.convexhull(hull_df['x'], hull_df['y']) # Calculate hull data
                        if team == 'away':
                            poly = pitch.polygon(hull, ax=ax, edgecolor='red', facecolor=color, alpha=0.3, zorder=-1) # Plot the polygon
                        elif team == 'home':
                            poly = pitch.polygon(hull, ax=ax, edgecolor='blue', facecolor=color, alpha=0.3, zorder=-1)
                        pitch.scatter(team_df['x'], team_df['y'], ax=ax, s=250, edgecolor=team_df['edgecolor'],
                            linewidths=team_df['edge_lw'], marker="o", c=color, zorder=3,label=team)
    def _plot_voronoi_viz(
        self,
        frame_df: pd.DataFrame,
        pitch, ax,
        team_colors: dict,
        gk_df: pd.DataFrame
        ) -> None:
        """
        Draws the voronoi visualization for player positions on a pitch.

        Parameters:
        ----------
        frame_df : pd.DataFrame
            DataFrame containing tracking data for a specific frame.
        pitch : mplsoccer.Pitch
            The pitch object used for plotting.
        ax : matplotlib.axes.Axes
            The axes object where the visualization will be drawn.
        team_colors : dict
            Dictionary mapping team names to color for plotting.
        gk_df : DataFrame
            Data about Goalkeeper numbers
        Returns:
        --------
        None, plots the player positions on the pitch.
        """
        # Exclude the ball data, keeping only player positions
        tracking_full = frame_df[frame_df['team'] != "ball"][['x', 'y', 'team']]

        X = tracking_full.x
        Y = tracking_full.y
        Team = tracking_full['team'].map({'home': 0, 'away': 1}) # Convert team names to numerical values (home = 0, away = 1)
    
        vor_away,vor_home = pitch.voronoi(X, Y, Team) # Compute Voronoi regions for each team

        # Plot Voronoi regions for home and away teams
        pitch.polygon(vor_home, fc=team_colors["home"], ax=ax, ec='white', lw=3, alpha=0.4) 
        pitch.polygon(vor_away, fc=team_colors["away"], ax=ax, ec='white', lw=3, alpha=0.4)

        # Plot players
        frame_data = frame_df[frame_df['team'] != "ball"] # Get player data only
        for team, team_col in team_colors.items() :# Iterate for home and away
            if team != "ball":
                team_df = frame_data[frame_data['team'] == team]
                pitch.scatter(team_df['x'], team_df['y'], ax=ax, s=250, edgecolor=team_df['edgecolor'],linewidths=team_df['edge_lw'], marker="o", c=team_col, zorder=3,label=team)



    def _plot_delaunay_viz(
        self,
        frame_df: pd.DataFrame,
        pitch, ax,
        team_colors: dict,
        gk_df: pd.DataFrame
        ) -> None:
        """
        Draws the voronoi visualization for player positions on a pitch.

        Parameters:
        ----------
        frame_df : pd.DataFrame
            DataFrame containing tracking data for a specific frame.
        pitch : mplsoccer.Pitch
            The pitch object used for plotting.
        ax : matplotlib.axes.Axes
            The axes object where the visualization will be drawn.
        team_colors : dict
            Dictionary mapping team names to color for plotting.
        gk_df : DataFrame
            Data about Goalkeeper numbers

        Returns:
        --------
        None, plots the player positions on the pitch.
        """
        gk_numbers = list(map(int, gk_df["shirt_number"]))

        # Exclude the ball data, keeping only player positions
        tracking_full = frame_df[frame_df['team'] != "ball"][['x', 'y', 'team', 'player']]
        
        tracking_home = tracking_full[tracking_full['team']=="home"]
        tracking_home = tracking_home[~tracking_home['player'].isin(gk_numbers)]  # Remove goalkeeper

        tracking_away = tracking_full[tracking_full['team']=="away"]
        tracking_away = tracking_away[~tracking_away['player'].isin(gk_numbers)] # Remove goalkeeper

        # Convert to arrays for Delauny calculation
        points_home = tracking_home[['x', 'y']].values
        del_home= Delaunay(tracking_home[['x', 'y']])

        points_away= tracking_away[['x', 'y']].values
        del_away= Delaunay(tracking_away[['x', 'y']])
        
        # Draw Delauny triangles for home and away teams
        plt.plot(points_home[del_home.simplices, 0], points_home[del_home.simplices, 1], team_colors["home"], zorder=1)
        plt.plot(points_away[del_away.simplices, 0], points_away[del_away.simplices, 1], team_colors["away"], zorder=1)


        # Plot players
        frame_data = frame_df[frame_df['team'] != "ball"] # Get player data only
        for team, team_col in team_colors.items() :# Iterate for home and away
            if team != "ball":
                team_df = frame_data[frame_data['team'] == team]
                pitch.scatter(team_df['x'], team_df['y'], ax=ax, s=250, edgecolor=team_df['edgecolor'],linewidths=team_df['edge_lw'], marker="o", c=team_col, zorder=3,label=team)


                                
    # Function to animate the frames
    def animate_frames(
        self,
        tidy_data: pd.DataFrame, 
        gk_df: pd.DataFrame,  
        frame_start: int, 
        frame_end: int, 
        mode: str = 'base', 
        text_font: str="Century Gothic",
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
        gk_df : pd.DataFrame
            Data containing goalkeeper numbers.
        frame_start : int
            The starting frame number for the animation.
        frame_end : int
            The ending frame number for the animation.
        mode : str, optional
            Visualization mode (default is 'base'). Other options are 'convexhull', 'delaunay', and 'voronoi'.
        text_font : str, optional
            Font for text in the visualization (default is "Century Gothic")
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
        frame_df = frame_df.groupby('Frame').apply(self._player_possession).reset_index(drop=True)        
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
                        fontname=text_font, fontsize=9, ha='center', va='center', 
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
            home_width, home_def_line, home_def_line_height = self._calculate_team_stats_pff(frame_data[frame_data['team'] == 'home'], 'home',gk_df)
            away_width, away_def_line, away_def_line_height = self._calculate_team_stats_pff(frame_data[frame_data['team'] == 'away'], 'away',gk_df)

            # Plot defensive lines
            ax.plot([home_def_line, home_def_line], [0, 100], lw=1, color=team_colors["home"], linestyle='--', zorder=1)
            ax.plot([away_def_line, away_def_line], [0, 100], lw=1, color=team_colors["away"], linestyle='--', zorder=1)

            legend_font = {
                'family': text_font,
                'size': 12,
            }

            # Define custom legend elements
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Home',
                    markerfacecolor=home_team_col, markersize=14),
                Line2D([0], [0], marker='o', color='w', label='Away',
                    markerfacecolor=away_team_col, markersize=14),
                Line2D([0], [0], marker='o', color='w', label='Ball',
                    markerfacecolor='black', markersize=10, markeredgecolor='yellow',markeredgewidth=2.5)
            ]

            # Add custom legend
            ax.legend(
                handles=legend_elements,
                title='Team',
                bbox_to_anchor=(1.045, 0.8),
                loc='center',
                handletextpad=0.5,
                labelspacing=1.0,
                prop=legend_font,
                title_fontproperties=legend_font,
                borderaxespad=0.3,
                borderpad=0.17,
                frameon=True
            )

            # Add clock time
            time_value = frame_data['time'].values[0]

            if frame_data["Period"].values[0] == 2:
                minutes, seconds = map(int, time_value.split(":"))
                minutes += 45
                time_value = f"{minutes:02}:{seconds:02}"

            str_text = f"Time - <{time_value}>"


            fig_text(
                x = 0.425, y = 0.79, 
                s = str_text,highlight_textprops=[{'color':'#FFD230', 'weight':'bold'}],
                va = 'bottom', ha = 'left',fontname =text_font, weight = 'bold',
                fontsize = 25,color ='white',path_effects=[path_effects.Stroke(linewidth=0.8, foreground="#BD8B00"), path_effects.Normal()]
            )
            

            # Determine current ball possession
            possession = frame_data["possession"].unique()
            possession =possession[0]
            color_now = team_colors.get(possession.lower(), "white")
            fig_text(
                x=0.41, y=0.77, s=f"Current Ball Possession: <{possession}>",
                highlight_textprops=[{'color': color_now, 'weight': 'bold'}],
                va='bottom', ha='left', fontname=text_font, weight='bold',
                fontsize=12, color='white'
            )

            # Display pitch width stats
            fig_text(
                x=0.34, y=0.748, s=f"Pitch width held by <Away> : {away_width:.2f} | <Home> : {home_width:.2f}",
                highlight_textprops=[{'color': team_colors["away"], 'weight': 'bold'}, 
                                    {'color': team_colors["home"], 'weight': 'bold'}],
                va='bottom', ha='left', fontname=text_font, weight='bold',
                fontsize=11, color='white'
            )

            # Display defensive line height
            fig_text(
                x=0.32, y=0.23, s=f"Defensive Line Height <Away --> : {away_def_line_height:.2f} | <Home -- > : {home_def_line_height:.2f}",
                highlight_textprops=[{'color': team_colors["away"], 'weight': 'bold'}, 
                                    {'color': team_colors["home"], 'weight': 'bold'}],
                va='bottom', ha='left', fontname=text_font, weight='bold',
                fontsize=11, color='white'
            )


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
                        fontname=text_font, fontsize=9, ha='center', va='center', 
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
            home_width, home_def_line, home_def_line_height = self._calculate_team_stats_pff(frame_data[frame_data['team'] == 'home'], 'home',gk_df)
            away_width, away_def_line, away_def_line_height = self._calculate_team_stats_pff(frame_data[frame_data['team'] == 'away'], 'away',gk_df)

            # Plot defensive lines
            ax.plot([home_def_line, home_def_line], [0, 100], lw=1, color=team_colors["home"], linestyle='--', zorder=1)
            ax.plot([away_def_line, away_def_line], [0, 100], lw=1, color=team_colors["away"], linestyle='--', zorder=1)

            legend_font = {
                'family': text_font,
                'size': 12,
            }

            # Define custom legend elements
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Home',
                    markerfacecolor=home_team_col, markersize=14),
                Line2D([0], [0], marker='o', color='w', label='Away',
                    markerfacecolor=away_team_col, markersize=14),
                Line2D([0], [0], marker='o', color='w', label='Ball',
                    markerfacecolor='black', markersize=10, markeredgecolor='yellow',markeredgewidth=2.5)
            ]

            # Add custom legend
            ax.legend(
                handles=legend_elements,
                title='Team',
                bbox_to_anchor=(1.045, 0.8),
                loc='center',
                handletextpad=0.5,
                labelspacing=1.0,
                prop=legend_font,
                title_fontproperties=legend_font,
                borderaxespad=0.3,
                borderpad=0.17,
                frameon=True
            )
            # Add clock time
            time_value = frame_data['time'].values[0]

            if frame_data["Period"].values[0] == 2:
                minutes, seconds = map(int, time_value.split(":"))
                minutes += 45
                time_value = f"{minutes:02}:{seconds:02}"

            str_text = f"Time - <{time_value}>"


            fig_text(
                x = 0.425, y = 0.79, 
                s = str_text,highlight_textprops=[{'color':'#FFD230', 'weight':'bold'}],
                va = 'bottom', ha = 'left',fontname =text_font, weight = 'bold',
                fontsize = 25,color ='white',path_effects=[path_effects.Stroke(linewidth=0.8, foreground="#BD8B00"), path_effects.Normal()]
            )
            

            # Determine current ball possession
            possession = frame_data["possession"].unique()
            possession =possession[0]
            color_now = team_colors.get(possession.lower(), "white")
            fig_text(
                x=0.41, y=0.77, s=f"Current Ball Possession: <{possession}>",
                highlight_textprops=[{'color': color_now, 'weight': 'bold'}],
                va='bottom', ha='left', fontname=text_font, weight='bold',
                fontsize=12, color='white'
            )

            # Display pitch width stats
            fig_text(
                x=0.34, y=0.748, s=f"Pitch width held by <Away> : {away_width:.2f} | <Home> : {home_width:.2f}",
                highlight_textprops=[{'color': team_colors["away"], 'weight': 'bold'}, 
                                    {'color': team_colors["home"], 'weight': 'bold'}],
                va='bottom', ha='left', fontname=text_font, weight='bold',
                fontsize=11, color='white'
            )

            # Display defensive line height
            fig_text(
                x=0.32, y=0.23, s=f"Defensive Line Height <Away --> : {away_def_line_height:.2f} | <Home -- > : {home_def_line_height:.2f}",
                highlight_textprops=[{'color': team_colors["away"], 'weight': 'bold'}, 
                                    {'color': team_colors["home"], 'weight': 'bold'}],
                va='bottom', ha='left', fontname=text_font, weight='bold',
                fontsize=11, color='white'
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
                        fontname=text_font, fontsize=9, ha='center', va='center', 
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
            home_width, home_def_line, home_def_line_height = self._calculate_team_stats_pff(frame_data[frame_data['team'] == 'home'], 'home',gk_df)
            away_width, away_def_line, away_def_line_height = self._calculate_team_stats_pff(frame_data[frame_data['team'] == 'away'], 'away',gk_df)

            # Plot defensive lines
            ax.plot([home_def_line, home_def_line], [0, 100], lw=1, color=team_colors["home"], linestyle='--', zorder=1)
            ax.plot([away_def_line, away_def_line], [0, 100], lw=1, color=team_colors["away"], linestyle='--', zorder=1)

            legend_font = {
                'family': text_font,
                'size': 12,
            }

            # Define custom legend elements
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Home',
                    markerfacecolor=home_team_col, markersize=14),
                Line2D([0], [0], marker='o', color='w', label='Away',
                    markerfacecolor=away_team_col, markersize=14),
                Line2D([0], [0], marker='o', color='w', label='Ball',
                    markerfacecolor='black', markersize=10, markeredgecolor='yellow',markeredgewidth=2.5)
            ]

            # Add custom legend
            ax.legend(
                handles=legend_elements,
                title='Team',
                bbox_to_anchor=(1.045, 0.8),
                loc='center',
                handletextpad=0.5,
                labelspacing=1.0,
                prop=legend_font,
                title_fontproperties=legend_font,
                borderaxespad=0.3,
                borderpad=0.17,
                frameon=True
            )
            # Add clock time
            time_value = frame_data['time'].values[0]

            if frame_data["Period"].values[0] == 2:
                minutes, seconds = map(int, time_value.split(":"))
                minutes += 45
                time_value = f"{minutes:02}:{seconds:02}"

            str_text = f"Time - <{time_value}>"


            fig_text(
                x = 0.425, y = 0.79, 
                s = str_text,highlight_textprops=[{'color':'#FFD230', 'weight':'bold'}],
                va = 'bottom', ha = 'left',fontname =text_font, weight = 'bold',
                fontsize = 25,color ='white',path_effects=[path_effects.Stroke(linewidth=0.8, foreground="#BD8B00"), path_effects.Normal()]
            )
            

            # Determine current ball possession
            possession = frame_data["possession"].unique()
            possession =possession[0]
            color_now = team_colors.get(possession.lower(), "white")
            fig_text(
                x=0.41, y=0.77, s=f"Current Ball Possession: <{possession}>",
                highlight_textprops=[{'color': color_now, 'weight': 'bold'}],
                va='bottom', ha='left', fontname=text_font, weight='bold',
                fontsize=12, color='white'
            )

            # Display pitch width stats
            fig_text(
                x=0.34, y=0.748, s=f"Pitch width held by <Away> : {away_width:.2f} | <Home> : {home_width:.2f}",
                highlight_textprops=[{'color': team_colors["away"], 'weight': 'bold'}, 
                                    {'color': team_colors["home"], 'weight': 'bold'}],
                va='bottom', ha='left', fontname=text_font, weight='bold',
                fontsize=11, color='white'
            )

            # Display defensive line height
            fig_text(
                x=0.32, y=0.23, s=f"Defensive Line Height <Away --> : {away_def_line_height:.2f} | <Home -- > : {home_def_line_height:.2f}",
                highlight_textprops=[{'color': team_colors["away"], 'weight': 'bold'}, 
                                    {'color': team_colors["home"], 'weight': 'bold'}],
                va='bottom', ha='left', fontname=text_font, weight='bold',
                fontsize=11, color='white'
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
                        fontname=text_font, fontsize=9, ha='center', va='center', 
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
            home_width, home_def_line, home_def_line_height = self._calculate_team_stats_pff(frame_data[frame_data['team'] == 'home'], 'home',gk_df)
            away_width, away_def_line, away_def_line_height = self._calculate_team_stats_pff(frame_data[frame_data['team'] == 'away'], 'away',gk_df)

            # Plot defensive lines
            ax.plot([home_def_line, home_def_line], [0, 100], lw=1, color=team_colors["home"], linestyle='--', zorder=1)
            ax.plot([away_def_line, away_def_line], [0, 100], lw=1, color=team_colors["away"], linestyle='--', zorder=1)

            legend_font = {
                'family': text_font,
                'size': 12,
            }

            # Define custom legend elements
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Home',
                    markerfacecolor=home_team_col, markersize=14),
                Line2D([0], [0], marker='o', color='w', label='Away',
                    markerfacecolor=away_team_col, markersize=14),
                Line2D([0], [0], marker='o', color='w', label='Ball',
                    markerfacecolor='black', markersize=10, markeredgecolor='yellow',markeredgewidth=2.5)
            ]

            # Add custom legend
            ax.legend(
                handles=legend_elements,
                title='Team',
                bbox_to_anchor=(1.045, 0.8),
                loc='center',
                handletextpad=0.5,
                labelspacing=1.0,
                prop=legend_font,
                title_fontproperties=legend_font,
                borderaxespad=0.3,
                borderpad=0.17,
                frameon=True
            )
            # Add clock time
            time_value = frame_data['time'].values[0]

            if frame_data["Period"].values[0] == 2:
                minutes, seconds = map(int, time_value.split(":"))
                minutes += 45
                time_value = f"{minutes:02}:{seconds:02}"

            str_text = f"Time - <{time_value}>"


            fig_text(
                x = 0.425, y = 0.79, 
                s = str_text,highlight_textprops=[{'color':'#FFD230', 'weight':'bold'}],
                va = 'bottom', ha = 'left',fontname =text_font, weight = 'bold',
                fontsize = 25,color ='white',path_effects=[path_effects.Stroke(linewidth=0.8, foreground="#BD8B00"), path_effects.Normal()]
            )
            

            # Determine current ball possession
            possession = frame_data["possession"].unique()
            possession =possession[0]
            color_now = team_colors.get(possession.lower(), "white")
            fig_text(
                x=0.41, y=0.77, s=f"Current Ball Possession: <{possession}>",
                highlight_textprops=[{'color': color_now, 'weight': 'bold'}],
                va='bottom', ha='left', fontname=text_font, weight='bold',
                fontsize=12, color='white'
            )

            # Display pitch width stats
            fig_text(
                x=0.34, y=0.748, s=f"Pitch width held by <Away> : {away_width:.2f} | <Home> : {home_width:.2f}",
                highlight_textprops=[{'color': team_colors["away"], 'weight': 'bold'}, 
                                    {'color': team_colors["home"], 'weight': 'bold'}],
                va='bottom', ha='left', fontname=text_font, weight='bold',
                fontsize=11, color='white'
            )

            # Display defensive line height
            fig_text(
                x=0.32, y=0.23, s=f"Defensive Line Height <Away --> : {away_def_line_height:.2f} | <Home -- > : {home_def_line_height:.2f}",
                highlight_textprops=[{'color': team_colors["away"], 'weight': 'bold'}, 
                                    {'color': team_colors["home"], 'weight': 'bold'}],
                va='bottom', ha='left', fontname=text_font, weight='bold',
                fontsize=11, color='white'
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


    def _generate_possession_frames_pff(self,possession_df: pd.DataFrame) -> list:
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


    def _find_stats_home_pff(self,df: pd.DataFrame, gk_df: pd.DataFrame) -> tuple:
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
        gk_df : DataFrame
            Data about Goalkeeper numbers
        Returns:
        --------
        tuple
            (average lowest x-coordinate, average highest x-coordinate).
        """
        gk_numbers = list(map(int, gk_df["shirt_number"]))

        df.loc[:, "player"] = df["player"].astype(int)
        filtered_df = df[~df['player'].isin(gk_numbers)]

        # Find the lowest and highest x-coordinates per frame
        lowest_x_per_frame = filtered_df.loc[filtered_df.groupby('Frame')['x'].idxmax()]
        highest_x_per_frame = filtered_df.loc[filtered_df.groupby('Frame')['x'].idxmin()]

        # Compute the averages
        avg_lowest_x = lowest_x_per_frame['x'].mean()
        avg_highest_x = highest_x_per_frame['x'].mean()

        return avg_lowest_x, avg_highest_x


    def _find_stats_away_pff(self,df: pd.DataFrame,gk_df: pd.DataFrame) -> tuple:
        """
        Finds the average defensive and attacking line positions for the away team.

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
        gk_df : DataFrame
            Data about Goalkeeper numbers
        Returns:
        --------
        tuple
            (average lowest x-coordinate, average highest x-coordinate).
        """
        gk_numbers = list(map(int, gk_df["shirt_number"]))

        df.loc[:, "player"] = df["player"].astype(int)
        filtered_df = df[~df['player'].isin(gk_numbers)]

        # Find the lowest and highest x-coordinates per frame
        lowest_x_per_frame = filtered_df.loc[filtered_df.groupby('Frame')['x'].idxmin()]
        highest_x_per_frame = filtered_df.loc[filtered_df.groupby('Frame')['x'].idxmax()]

        # Compute the averages
        avg_lowest_x = lowest_x_per_frame['x'].mean()
        avg_highest_x = highest_x_per_frame['x'].mean()

        return avg_lowest_x, avg_highest_x
    
    # Function to plot on/off ball possession positioning of teams
    def plot_possession_pff(
        self,
        tidy_data: pd.DataFrame, 
        target_team: str, 
        gk_df: pd.DataFrame, 
        player_df: pd.DataFrame, 
        team_mapping : dict,
        text_font="Century Gothic",
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
        target_team : str
            Team name to analyze ('home' or 'away').
        gk_df : pd.DataFrame
            Data containing Goalkeeper shirt numbers
        player_df : pd.DataFrame
            Data containing Player details
        team_mapping : dict,
            Mapping of team names.
        text_font : str
            Text Font
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
        stats_df =tidy_data.copy()
        stats_df = stats_df[stats_df["team"] == target_team]
        
        team_inposs = stats_df[ stats_df["possession"] ==target_team ]
        team_outposs = stats_df[ stats_df["possession"] != target_team ]

        # Find average player positions for in poss/ out poss
        avg_positions_ip = team_inposs.groupby('player').agg({
            'x': 'mean',
            'y': 'mean'
        }).reset_index()

        avg_positions_op = team_outposs.groupby('player').agg({
            'x': 'mean',
            'y': 'mean'
        }).reset_index()
            
        avg_positions_ip["player"] = avg_positions_ip["player"].astype(int)
        avg_positions_op["player"] = avg_positions_op["player"].astype(int)

        # Keep only the players who started the match 
        player_df["team"] = player_df["team_name"].map(team_mapping)
        players_started = player_df[player_df["started"] == True]
        avg_positions_ip = avg_positions_ip[
            avg_positions_ip["player"].isin(players_started[players_started["team"] == target_team]["shirt_number"])
        ].sort_values(by="player")
        avg_positions_op  = avg_positions_op[
            avg_positions_op["player"].isin(players_started[players_started["team"] == target_team]["shirt_number"])
        ].sort_values(by="player")
        
        # Find defensive line avg and high line avg
        if target_team == "home":
            team_df_line_inposs,team_high_xip = self._find_stats_home_pff(team_inposs,gk_df)
            team_df_line_oposs,team_high_xop =  self._find_stats_home_pff(team_outposs,gk_df)
        else:
            team_df_line_inposs,team_high_xip = self._find_stats_away_pff(team_inposs,gk_df)
            team_df_line_oposs,team_high_xop =  self._find_stats_away_pff(team_outposs,gk_df)


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

        if target_team == "home":
            color = home_team_col
        else:
            color =away_team_col


        # In possesion plots
        plt.scatter(avg_positions_ip['x'], avg_positions_ip['y'], s=250, alpha=1, c=color, edgecolors="white", marker="o", linewidths=1.6,zorder=3)
                
        # Add player number and speed with direction
        for index, row in avg_positions_ip.iterrows():
                plt.text(row['x'], row['y'], str(int(row['player'])),font=text_font ,fontsize=9, ha='center', va='center', weight="bold",color='white',zorder=3)


        avg_positions_ip['player'] = avg_positions_ip['player'].astype(int)
        hull_df = avg_positions_ip[~avg_positions_ip['player'].isin([11, 25])]

        if not hull_df.empty:
                hull = pitch.convexhull(hull_df['x'], hull_df['y']) # Calculate hull data
                hull_spatial = ConvexHull(hull_df[['x', 'y']])
                poly = pitch.polygon(hull, ax=ax, edgecolor=color, facecolor=color, alpha=0.3, zorder=-1)

        ax.plot([team_df_line_inposs, team_df_line_inposs], [0,100], lw=1, color=color,  linestyle='--',zorder=1)
        
        # Find spread and area of convex hull
        ip_area = hull_spatial.area
        points = hull_df[['x', 'y']].values
        ip_spread = max(np.linalg.norm(points[i] - points[j]) for i in range(len(points)) for j in range(i + 1, len(points)))

        
        # Out possesion plots
        plt.scatter(avg_positions_op['x'], avg_positions_op['y'], s=250, alpha=1, c="white", edgecolors=color, marker="o", linewidths=1.6,zorder=3)
                

        # Add player number and speed with direction
        for index, row in avg_positions_op.iterrows():
                plt.text(row['x'], row['y'], str(int(row['player'])),font=text_font ,fontsize=9, ha='center', va='center', weight="bold",color='black',zorder=3)


        avg_positions_op['player'] = avg_positions_op['player'].astype(int)
        hull_df = avg_positions_op[~avg_positions_op['player'].isin([11, 25])]

        if not hull_df.empty:
                hull = pitch.convexhull(hull_df['x'], hull_df['y']) # Calculate hull data
                hull_spatial = ConvexHull(hull_df[['x', 'y']])
                poly = pitch.polygon(hull, ax=ax, edgecolor="white", facecolor="white", alpha=0.3, zorder=-1)


        ax.plot([team_df_line_oposs, team_df_line_oposs], [0,100], lw=1, color="white",  linestyle='--',zorder=1)

        # Find spread and area of convex hull
        op_area = hull_spatial.area
        points = hull_df[['x', 'y']].values
        op_spread = max(np.linalg.norm(points[i] - points[j]) for i in range(len(points)) for j in range(i + 1, len(points)))


        # Connect players
        for player in avg_positions_ip['player']:
            ip_pos = avg_positions_ip[avg_positions_ip['player'] == player]
            op_pos = avg_positions_op[avg_positions_op['player'] == player]
            plt.plot([ip_pos['x'].values[0], op_pos['x'].values[0]],
                    [ip_pos['y'].values[0], op_pos['y'].values[0]],
                    color='gray', linestyle='--', linewidth=1, zorder=2)


        if target_team == "home":  
            fig_text(
                x = 0.503, y = 0.237, 
                s = "Direction of Play",
                va = 'bottom', ha = 'left',fontname =text_font,
                fontsize = 9,color ='white'
            )
            # Your existing code
            arrow_location = (48, -2.8)

            # Add a horizontal arrow at the specified location
            plt.arrow(
                arrow_location[0], arrow_location[1],
                -3, 0,  # Adjust the arrow direction to make it horizontal
                shape='full', color='white', linewidth=4,
                head_width=0.2, head_length=0.2
            )
        else:            
            fig_text(
                x = 0.42, y = 0.237, 
                s = "Direction of Play",
                va = 'bottom', ha = 'left',fontname =text_font,
                fontsize = 9,color ='white'
            )
            # Your existing code
            arrow_location = (52, -2.8)

            # Add a horizontal arrow at the specified location
            plt.arrow(
                arrow_location[0], arrow_location[1],
                3, 0,  # Adjust the arrow direction to make it horizontal
                shape='full', color='white', linewidth=4,
                head_width=0.2, head_length=0.2
            )

        str_text = f"<ON>/<OFF> Ball avg. positioning -<{target_team.capitalize()} Team>"

        fig_text(
            x = 0.14, y = 0.745, 
            s = str_text,highlight_textprops=[{'color':color, 'weight':'bold'},
            {'color':"white", 'weight':'bold'},{'color':color, 'weight':'bold'}],
            va = 'bottom', ha = 'left',fontname =text_font, weight = 'bold',
            fontsize = 13,color ='white'
        )

        # Create and Add stats table to plot
        columns = ['Metric', 'OFF', 'ON']
        df = pd.DataFrame(columns=columns)

        # Data rows for the DataFrame
        data_rows = [
            {'Metric': "Def Line x", 'OFF': team_df_line_oposs, 'ON': team_df_line_inposs},
            {'Metric':  "Att Line x", 'OFF': team_high_xop, 'ON': team_high_xip},
            {'Metric': "Spread", 'OFF': op_spread, 'ON': ip_spread},
            {'Metric': "Area", 'OFF': op_area, 'ON': ip_area}
        ]

        df = pd.DataFrame(data_rows)

        columns_to_round = ['OFF', 'ON']  # Specify the columns you want to round
        df[columns_to_round] = df[columns_to_round].round(2)

        # Calculate the difference percentage and format it
        df["DIFF %"] = (((df['ON'] - df['OFF']) / df['OFF']) * 100).round(2)

        # Explicitly cast to string first
        df["DIFF %"] = df["DIFF %"].astype(str)

        if target_team == "away":
            # left to right so no need to adjust +/- signs
            df["DIFF %"] = df["DIFF %"].apply(lambda x: f"+{x}" if not x.startswith(('+', '-')) and float(x) > 0 else f"{x}")
        else:  
            # home: right to left so adjust +/- signs
            df.loc[:1, "DIFF %"] = df.loc[:1, "DIFF %"].apply(lambda x: f"-{abs(float(x))}" if float(x) > 0 else f"+{abs(float(x))}")
            df.loc[2:, "DIFF %"] = df.loc[2:, "DIFF %"].apply(lambda x: f"+{x}" if not x.startswith(('+', '-')) and float(x) > 0 else f"{x}")

        self._create_stat_table_pff(df,text_font) # Create table plot

        # Overlay table onto pitch
        im1 = plt.imread("images/table.png")

        if target_team == "home":
            ax_image = add_image( im1, fig, left=0.144, bottom=0.57, width=0.23, height=0.23 )  
        else: 
            ax_image = add_image( im1, fig, left=0.65, bottom=0.57, width=0.23, height=0.23 )  

        if save == True:
            plt.savefig(f'images/{target_team}_possession.png', bbox_inches='tight', dpi=300) 





    def _create_stat_table_pff(self,df: pd.DataFrame,text_font : str) -> None:
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
        text_font : str
            Font used for text in the table (default is 'Century Gothic').

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
            cell.set_text_props(color='black', fontname=text_font) 

        table.auto_set_font_size(False)
        table.set_fontsize(12)  # Set font size
        table.scale(1.2, 1.2)  # Scale table size

        # Save the table as a PNG file
        plt.savefig('images/table.png', bbox_inches='tight', dpi=300) 
        plt.close()



    # Plot Heatmap for player in-possession and off-possession positions
    def player_heatmap_pff(
        self,
        tidy_data: pd.DataFrame, 
        target_team: str, 
        target_player: int, 
        text_font="Century Gothic",
        pitch_fill: str = "black", 
        pitch_lines_col: str = "#7E7D7D",
        pitch_type: str = 'opta', 
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
        home_team_col : str, optional
            Color representing the home team (default is '#0A97B0').
        away_team_col : str, optional
            Color representing the away team (default is '#A04747').

        Returns:
        --------
        None. Displays and optionally saves the heatmaps.
        """
        # Subset data of particular team                  
        stats_df =tidy_data.copy()
        stats_df = stats_df[stats_df["team"] == target_team]
        

        # Get dataframe where team is in/out possession
        player_ip = stats_df[(stats_df["possession"] ==target_team) & (stats_df["player"] == target_player)]
        player_op = stats_df[(stats_df["possession"] != target_team) & (stats_df["player"] == target_player)]


        # Plot in-possession heatmap Your existing code
        self._plot_heatmap(player_ip,target_team, text_font,f"In Possession Heatmap\nPlayer Number : <{target_player}>",
                        pitch_fill,pitch_lines_col,pitch_type,home_team_col,away_team_col)

        # Plot off-possession heatmap        # Add a horizontal arrow at the specified location
        self._plot_heatmap(player_op, target_team,text_font, f"Off Possession Heatmap\nPlayer Number : <{target_player}>",
                        pitch_fill,pitch_lines_col,pitch_type,home_team_col,away_team_col)




    # Function to plot heatmap
    def _plot_heatmap(
        self,
        player_data: pd.DataFrame,
        target_team: str, 
        text_font : str,
        title: str,
        pitch_fill: str = "black", 
        pitch_lines_col: str = "#7E7D7D",
        pitch_type: str = 'opta', 
        home_team_col: str = "#0A97B0", 
        away_team_col: str = "#A04747") -> None:

        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = plt.subplot(111)

        fig.set_facecolor(pitch_fill)
        ax.patch.set_facecolor(pitch_fill)

        pitch = Pitch(pitch_color=pitch_fill, pitch_type=pitch_type, goal_type='box', linewidth=0.85, line_color=pitch_lines_col)
        pitch.draw(ax=ax)

        # Define color based on team
        color = home_team_col if target_team == "home" else away_team_col
        # Define custom colormap
        from matplotlib.colors import LinearSegmentedColormap  
        req_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors", ['black', color], N=10)

        pitch.kdeplot(player_data.x, player_data.y, ax=ax, fill=True, levels=50, thresh=0, cut=4, zorder=-1, cmap=req_cmap)

        fig_text(x=0.145, y=0.747, s=title, highlight_textprops=[{'color': color, 'weight': 'bold'}], 
                va='bottom', ha='left', fontname=text_font, 
                weight='bold', fontsize=13, color='white')

        if target_team == "home":  
            fig_text(x=0.503, y=0.237, s="Direction of Play", va='bottom', ha='left', 
                    fontname=text_font, fontsize=9, color='white')
            plt.arrow(48, -2.8, -3, 0, shape='full', color='white', 
                    linewidth=4, head_width=0.2, head_length=0.2)
        else:            
            fig_text(x=0.42, y=0.237, s="Direction of Play", va='bottom', ha='left', 
                    fontname=text_font, fontsize=9, color='white')
            plt.arrow(52, -2.8, 3, 0, shape='full', color='white', 
                    linewidth=4, head_width=0.2, head_length=0.2)



    # Calculate player running data
    def player_movement_stats_pff(self,tidy_data: pd.DataFrame,player_shirt_df: pd.DataFrame,save = True,text_font:str="Century Gothic") -> pd.DataFrame:
        """
        Calculate player running statistics including distance covered in different speed ranges.

        Parameters:
        -----------
        tidy_data : pd.DataFrame
            Processed tracking data containing player positions and other match details.
        player_shirt_df : pd.DataFrame
            Player shirt number data
        save : bool, optional
            Whether to save the output tables as images (default is True).
        text_font:str,optional
            Text Font

        Returns:
        --------
        pd.DataFrame
            DataFrame containing player running statistics.
        """
        range_names = ["walking_km", "jogging_km", "running_km", "sprinting_km"]
        
        # Remove ball data
        non_ball_df = tidy_data[tidy_data['team'] != 'ball']
        non_ball_df = non_ball_df.reset_index()

        # Find avg_speed,distance and no of frames by player
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
                temp = non_ball_df[(non_ball_df['speed'] >= WALK_JOG_THRESHOLD) & (non_ball_df['speed'] < JOG_RUN_THRESHOLD)].groupby(['team', 'player'])['v_mod'].sum() /(MS_LAG_SMOOTH* 1000)
            elif i == 3:
                temp = non_ball_df[(non_ball_df['speed'] >= JOG_RUN_THRESHOLD) & (non_ball_df['speed'] < RUN_SPRINT_THRESHOLD)].groupby(['team', 'player'])['v_mod'].sum() /( MS_LAG_SMOOTH* 1000)
            elif i == 4:
                temp = non_ball_df[non_ball_df['speed'] >= RUN_SPRINT_THRESHOLD].groupby(['team', 'player'])['v_mod'].sum() / (MS_LAG_SMOOTH * 1000)

            # Add to player_df for coresponding speed range
            player_df[name] = temp.reset_index(drop=True)

        # Calculate additional columns
        player_df['avg_speed_km_h'] = player_df['avg_speed_m_s'] / 1000 * 3600

        # Sort the data
        output = player_df.sort_values(by=['team'])

        output = output[['team', 'player','distance_km',
        'walking_km', 'jogging_km', 'running_km', 'sprinting_km', 'avg_speed_m_s','avg_speed_km_h']]
        
        # Convert to float and round decimals
        output = output.astype({col: 'float' for col in output.columns if col not in ['team', 'player']}).round({col: 2 for col in output.columns if col not in ['team', 'player']})
        
        # Find home and away data sorted by most distance ran
        home_table = output[output["team"] == "home" ]
        away_table = output[output["team"] == "away" ]

        # Rename "player" column to "player_id"
        home_table = home_table.rename(columns={"player": "player_id"})
        away_table = away_table.rename(columns={"player": "player_id"})

        # Map home players
        home_players = player_shirt_df[player_shirt_df["team"] == "home"]
        home_table["player"] = home_table["player_id"].astype("Int64").map(
            home_players.set_index("shirt_number")["nickname"]
        )

        # Map away players
        away_players = player_shirt_df[player_shirt_df["team"] == "away"]
        away_table["player"] = away_table["player_id"].astype("Int64").map(
            away_players.set_index("shirt_number")["nickname"]
        )

        # Drop the 'player_id' column
        home_table = home_table.drop(columns=["player_id"])
        away_table = away_table.drop(columns=["player_id"])

        # Reorder columns to move 'player' to index 1
        def move_column_to_index(df, column_name, new_index):
            cols = df.columns.tolist()
            cols.insert(new_index, cols.pop(cols.index(column_name)))
            return df[cols]

        home_table = move_column_to_index(home_table, "player", 1)
        away_table = move_column_to_index(away_table, "player", 1)

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
            cell.set_text_props(color='black', fontname=text_font) 


        table.auto_set_font_size(False)
        table.set_fontsize(12)  
        table.scale(1.2, 1.2) 

        if save == True:
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
            cell.set_text_props(color='black', fontname=text_font) 


        table.auto_set_font_size(False)
        table.set_fontsize(12) 
        table.scale(1.2, 1.2)  
        
        if save == True:
            plt.savefig('images/away_player_table.png', bbox_inches='tight', dpi=300) 

        return output





    # Plot graph of player speed
    def player_stats_graph_pff(self,player_stats: pd.DataFrame,player_shirt_df: pd.DataFrame, target_team: str, save: bool = True,text_font:str="Century Gothic") -> None:
        """
        Plot a graph of player speed statistics for a given team.

        Parameters:
        -----------
        player_stats : pd.DataFrame
            DataFrame containing player statistics including distance covered in different speed ranges.
        player_shirt_df : pd.DataFrame
            DataFrame containing player shirt numbers and names
        target_team : str
            Team name to plot the statistics for ('home' or 'away').
        save : bool, optional
            Whether to save the plot as an image (default is True).
        text_font : str,optional
            Text Font

        Returns:
        --------
        None. Displays and optionally saves the plot.
        """
        df = player_stats[player_stats["team"] == target_team].sort_values(by="distance_km", ascending=True)
    
        shirt_df = player_shirt_df[player_shirt_df["team"] == target_team]

        df["player"] = df["player"].map(
        shirt_df.set_index("shirt_number")["nickname"]
        )
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
        ax.set_title(f'Player distance covered in km - {target_team.capitalize()}', font=text_font, fontsize=24, fontweight='bold', color='white')

        plt.xlabel('Distance (km)', font=text_font, fontsize=14, fontweight='bold', color="white")
        plt.ylabel('Player', font=text_font, fontsize=14, fontweight='bold', color="white")

        plt.xticks(fontname=text_font, fontsize=12, color="white")
        plt.yticks(fontname=text_font, fontsize=12, color="white")

        ax.tick_params(axis='both', colors='white')
        ax.legend(title='Speed Range', loc='lower right', fontsize=10, facecolor='white', edgecolor='white')


        if save:
            plt.savefig(f'images/{target_team}_player_speed_graph.png', bbox_inches='tight', dpi=300)

        # Display the plot
        plt.tight_layout()
        plt.show()


