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

pd.set_option('future.no_silent_downcasting', True)

class MetricaTrackingDataProcessor(TrackingDataProcessor):
    def process(self, home_team_file, away_team_file, pitch_long=100, pitch_width=100, add_velocity=True):
        """ Args:
        home_team_file: Path to home team data
        away_team_file: Path to away team data
        pitch_long: Pitch length
        pitch_width: Pitch width
        add_velocity: Whether to calculate velocity
        """
        # Initialize parameters
        self.home_team_file = home_team_file
        self.away_team_file = away_team_file
        self.pitch_long = pitch_long
        self.pitch_width = pitch_width
        self.add_velocity = add_velocity

        # Read and clean the data
        track_home = pd.read_csv(self.home_team_file, low_memory=False)
        track_home = self._clean_columns(track_home).pipe(self._safe_to_float)
        track_home["team"] = "home"

        track_away = pd.read_csv(self.away_team_file, low_memory=False)
        track_away = self._clean_columns(track_away).pipe(self._safe_to_float)
        track_away["team"] = "away"

        # Combine home and away data
        combined = pd.concat([track_home, track_away], ignore_index=True)
        combined = combined.rename(columns={"Time [s]": "time", "Ballx": "xBall", "Bally": "yBall"})
        combined["second"] = np.floor(combined["time"])

        player_df = self._extract_player_data(combined)
        ball_df = self._extract_ball_data(combined)

        #  Combine player and ball data
        data = pd.concat([player_df, ball_df], ignore_index=True)
        data = self._transform_coordinates(data)

        if self.add_velocity:
            data = self._add_velocity(data)
        
        data["minutes"] = data["time"] // 60 + (data["time"] % 60) / 100 # Convert to minutes
        data["player"] = data["player"].astype(int)

        return data

    def _extract_player_data(self,df: pd.DataFrame) -> pd.DataFrame:
        """Extract player data from the DataFrame.

        Returns:
        Processed DataFrame
        """
        # Extract player data by identifying columns that end with 'x' and contain 'Player'
        result = []
        for col in df.columns:
            if col.endswith("x") and "Player" in col:
                player_num = col.replace("Player", "").replace("x", "")
                y_col = f"Player{player_num}y"

                temp = df[["Period", "Frame", "time", col, y_col, "team", "second"]].copy()
                temp["player"] = player_num
                temp = temp.rename(columns={col: "x", y_col: "y"})
                result.append(temp)

        return pd.concat(result, ignore_index=True)

    def _extract_ball_data(self,df: pd.DataFrame) -> pd.DataFrame:
        """Extract ball data from the DataFrame
        
        Returns:
        Processed DataFrame
        """
        ball_df = df[["Period", "Frame", "time", "xBall", "yBall", "team", "second"]].copy()
        ball_df["player"] = 0
        ball_df["team"] = "ball"
        return ball_df.rename(columns={"xBall": "x", "yBall": "y"})

    def _transform_coordinates(self,df: pd.DataFrame) -> pd.DataFrame:
        """Transform coordinates to match the pitch dimensions
        
        Returns:
        Processed DataFrame
        """
        df = df.sort_values(["Frame", "player", "x", "y"], na_position="last")
        df = df.drop_duplicates(subset=["Frame", "player"], keep="first")
        df = df.dropna(subset=["x"])

        df["y"] = 1 - df["y"]
        # Invert coordinates for 2nd half
        df["x"] = np.where(df["Period"] == 1, 
                           self.pitch_long * df["x"], 
                           self.pitch_long * (1 - df["x"]))
        df["y"] = np.where(df["Period"] == 1,
                           self.pitch_width * df["y"],
                           self.pitch_width * (1 - df["y"]))
        df[["x", "y"]] = df[["x", "y"]].round(2)
        return df

    def _add_velocity(self,df: pd.DataFrame) -> pd.DataFrame:
        """Add velocity column to the DataFrame

        Returns:
        Processed DataFrame
        """
        # Calculate velocity based on the difference in position over time
        df["dx"] = df.groupby("player")["x"].diff(MS_LAG_SMOOTH)
        df["dy"] = df.groupby("player")["y"].diff(MS_LAG_SMOOTH)
        df["v_mod"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
        # Calculate speed and limit it to PLAYER_MAX_SPEED
        df["speed"] = np.minimum(df["v_mod"] / (MS_DT * MS_LAG_SMOOTH), PLAYER_MAX_SPEED)
        return df
    

    def _clean_columns(self,df: pd.DataFrame) -> pd.DataFrame:
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
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

        new_columns = []
        for col in df.columns:
            if pd.notna(col) and (col.startswith('Player') or col == 'Ball'):
                new_columns.extend([f"{col}x", f"{col}y"])
            elif pd.notna(col):
                new_columns.append(col)
        
        df.columns = new_columns
        return df
    
    def _safe_to_float(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(pd.to_numeric, errors='coerce')
    
    # Get the frames where a particular team has possesion
    def _possesion_frames(self,events_df: pd.DataFrame, target_team: str) -> pd.DataFrame:
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
    
    def plot_single_frame(
        self,
        data: pd.DataFrame,
        target_frame: int,
        poss_data: pd.DataFrame,
        method: str = "base",
        pitch_fill: str = "black",
        pitch_lines_col: str = "#7E7D7D",
        pitch_type: str = 'opta',
        save: bool = True,
        home_team_col: str = "#0A97B0",
        away_team_col: str = "#A04747",
        text_font: str = "Century Gothic"
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
        text_font : str,optional
            Text Font
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
        text_font :str, optional
            Font for text in the visualization
        Returns:
        --------
        None, displays and optionally saves the visualization
        """
        frame_df = data[data["Frame"] == target_frame]
        frame_df = frame_df.groupby('Frame').apply(self._player_possession).reset_index(drop=True)
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
            self._plot_base_viz(frame_df, team_colors)
            
        elif method == "convexhull":
            self._plot_convexhull_viz(frame_df,pitch, ax,team_colors)
            
        elif method == "delaunay":
            self._plot_delaunay_viz(frame_df,pitch,ax,team_colors)
            
        elif method == "voronoi":
            self._plot_voronoi_viz(frame_df,pitch,ax,team_colors)
            

        # Plot ball
        team_df = frame_df[frame_df['team'] == "ball"]
        plt.scatter(team_df['x'], team_df['y'], s=50, alpha=1, facecolors='none', edgecolors="yellow", marker="o", linewidths=1.5,zorder=2)

    
        #  Find defensive line and pitch width holded
        home_width, home_defensive_line,home_def_line_height = self._calculate_team_stats_metrica(frame_df[frame_df['team'] == 'home'], 'home')
        away_width, away_defensive_line,away_def_line_height = self._calculate_team_stats_metrica(frame_df[frame_df['team'] == 'away'], 'away')
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
        minute_value = frame_df['minutes'].values[0]
        minutes = int(minute_value)
        seconds = int((minute_value - minutes) * 60)
        str_text = f"Time - <{minutes:02d}:{seconds:02d}>"

        fig_text(
            x = 0.42, y = 0.79, 
            s = str_text,highlight_textprops=[{'color':'#FFD230', 'weight':'bold'}],
            va = 'bottom', ha = 'left',fontname =text_font, weight = 'bold',
            fontsize = 25,color ='white',path_effects=[path_effects.Stroke(linewidth=0.8, foreground="#BD8B00"), path_effects.Normal()]
        )


        # Find current ball possesion
        possession = self._team_possession_metrica(target_frame, poss_data)
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
    
    def _player_possession(self, frame_data: pd.DataFrame) -> pd.DataFrame:
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
    
    
    def _team_possession_metrica(self,frame: int, possession_df: pd.DataFrame) -> str:
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



    def _calculate_team_stats_metrica(self,team_df: pd.DataFrame, team_name: str) -> tuple:
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
    
    def _plot_base_viz(
        self,
        frame_df: pd.DataFrame,
        team_colors: dict,
        ) -> None:
        """
        Draw the base visualization for player positions on a pitch.

        Parameters:
        -----------
        frame_df : pd.DataFrame
            DataFrame containing tracking data for a specific frame.
        team_colors : dict
            Dictionary mapping team names to color for plotting.

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

        Returns:
        --------
        None, plots the player positions on the pitch.
        """
        # Iterate over each team
        for team, color in team_colors.items():
            if team != 'ball':
                team_df = frame_df[frame_df['team'] == team] # Filter out the team data
                if not team_df.empty:
                    # Exclude players with numbers 11 and 25 for convex hull calculation
                    hull_df = team_df[~team_df['player'].isin([11, 25])]

                    if not hull_df.empty:
                        hull = pitch.convexhull(hull_df['x'], hull_df['y']) # Calculate hull data
                        if team == 'away':
                            poly = pitch.polygon(hull, ax=ax, edgecolor='red', facecolor=color, alpha=0.3, zorder=-1) # Plot the polygon
                            pitch.scatter(team_df['x'], team_df['y'], ax=ax, s=250, edgecolor=team_df['edgecolor'],
                            linewidths=team_df['edge_lw'], marker="o", c=color, zorder=3,label=team) # Plot the player positions
                        elif team == 'home':
                            poly = pitch.polygon(hull, ax=ax, edgecolor='blue', facecolor=color, alpha=0.3, zorder=-1)
                            pitch.scatter(team_df['x'], team_df['y'], ax=ax, s=250, edgecolor=team_df['edgecolor'],
                            linewidths=team_df['edge_lw'], marker="o", c=color, zorder=3,label=team)
    def _plot_voronoi_viz(
        self,
        frame_df: pd.DataFrame,
        pitch, ax,
        team_colors: dict,
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

        Returns:
        --------
        None, plots the player positions on the pitch.
        """
        # Exclude the ball data, keeping only player positions
        tracking_full = frame_df[frame_df['team'] != "ball"][['x', 'y', 'team', 'player']]
        
        tracking_home = tracking_full[tracking_full['team']=="home"]
        tracking_home = tracking_home[tracking_home['player'] != 11]  # Remove goalkeeper

        tracking_away = tracking_full[tracking_full['team']=="away"]
        tracking_away = tracking_away[tracking_away['player'] != 25] # Remove goalkeeper

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
        poss_data: pd.DataFrame, 
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
        poss_data : pd.DataFrame
            Data containing possession frames for the teams
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
            home_width, home_def_line, home_def_line_height = self._calculate_team_stats_metrica(frame_data[frame_data['team'] == 'home'], 'home')
            away_width, away_def_line, away_def_line_height = self._calculate_team_stats_metrica(frame_data[frame_data['team'] == 'away'], 'away')

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

            # Display match time
            minute_value = frame_data['minutes'].values[0]
            minutes = int(minute_value)
            seconds = int((minute_value - minutes) * 60)
            str_text = f"Time - <{minutes:02d}:{seconds:02d}>"

            fig_text(
                x = 0.425, y = 0.79, 
                s = str_text,highlight_textprops=[{'color':'#FFD230', 'weight':'bold'}],
                va = 'bottom', ha = 'left',fontname =text_font, weight = 'bold',
                fontsize = 25,color ='white',path_effects=[path_effects.Stroke(linewidth=0.8, foreground="#BD8B00"), path_effects.Normal()]
            )
            

            # Determine current ball possession
            possession =  self._team_possession_metrica(frame, poss_data)
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
            home_width, home_def_line, home_def_line_height = self._calculate_team_stats_metrica(frame_data[frame_data['team'] == 'home'], 'home')
            away_width, away_def_line, away_def_line_height = self._calculate_team_stats_metrica(frame_data[frame_data['team'] == 'away'], 'away')

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

            # Display match time
            minute_value = frame_data['minutes'].values[0]
            minutes = int(minute_value)
            seconds = int((minute_value - minutes) * 60)
            str_text = f"Time - <{minutes:02d}:{seconds:02d}>"

            fig_text(
                x = 0.425, y = 0.79, 
                s = str_text,highlight_textprops=[{'color':'#FFD230', 'weight':'bold'}],
                va = 'bottom', ha = 'left',fontname =text_font, weight = 'bold',
                fontsize = 25,color ='white',path_effects=[path_effects.Stroke(linewidth=0.8, foreground="#BD8B00"), path_effects.Normal()]
            )
            

            # Determine current ball possession
            possession =  self._team_possession_metrica(frame, poss_data)
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
            home_width, home_def_line, home_def_line_height = self._calculate_team_stats_metrica(frame_data[frame_data['team'] == 'home'], 'home')
            away_width, away_def_line, away_def_line_height = self._calculate_team_stats_metrica(frame_data[frame_data['team'] == 'away'], 'away')

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

            # Display match time
            minute_value = frame_data['minutes'].values[0]
            minutes = int(minute_value)
            seconds = int((minute_value - minutes) * 60)
            str_text = f"Time - <{minutes:02d}:{seconds:02d}>"

            fig_text(
                x = 0.425, y = 0.79, 
                s = str_text,highlight_textprops=[{'color':'#FFD230', 'weight':'bold'}],
                va = 'bottom', ha = 'left',fontname =text_font, weight = 'bold',
                fontsize = 25,color ='white',path_effects=[path_effects.Stroke(linewidth=0.8, foreground="#BD8B00"), path_effects.Normal()]
            )
            

            # Determine current ball possession
            possession =  self._team_possession_metrica(frame, poss_data)
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
            home_width, home_def_line, home_def_line_height = self._calculate_team_stats_metrica(frame_data[frame_data['team'] == 'home'], 'home')
            away_width, away_def_line, away_def_line_height = self._calculate_team_stats_metrica(frame_data[frame_data['team'] == 'away'], 'away')

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

            # Display match time
            minute_value = frame_data['minutes'].values[0]
            minutes = int(minute_value)
            seconds = int((minute_value - minutes) * 60)
            str_text = f"Time - <{minutes:02d}:{seconds:02d}>"

            fig_text(
                x = 0.425, y = 0.79, 
                s = str_text,highlight_textprops=[{'color':'#FFD230', 'weight':'bold'}],
                va = 'bottom', ha = 'left',fontname =text_font, weight = 'bold',
                fontsize = 25,color ='white',path_effects=[path_effects.Stroke(linewidth=0.8, foreground="#BD8B00"), path_effects.Normal()]
            )
            

            # Determine current ball possession
            possession =  self._team_possession_metrica(frame, poss_data)
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


    def _generate_possession_frames_metrica(self,possession_df: pd.DataFrame) -> list:
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


    def _find_stats_home_metrica(self,df: pd.DataFrame, gk: int = 11) -> tuple:
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
        df.loc[:, "player"] = df["player"].astype(int)
        filtered_df = df[df['player'] != gk]

        # Find the lowest and highest x-coordinates per frame
        lowest_x_per_frame = filtered_df.loc[filtered_df.groupby('Frame')['x'].idxmax()]
        highest_x_per_frame = filtered_df.loc[filtered_df.groupby('Frame')['x'].idxmin()]

        # Compute the averages
        avg_lowest_x = lowest_x_per_frame['x'].mean()
        avg_highest_x = highest_x_per_frame['x'].mean()

        return avg_lowest_x, avg_highest_x


    def _find_stats_away_metrica(self,df: pd.DataFrame, gk: int = 25) -> tuple:
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
        df.loc[:, "player"] = df["player"].astype(int)
        filtered_df = df[df['player'] != gk]

        # Find the lowest and highest x-coordinates per frame
        lowest_x_per_frame = filtered_df.loc[filtered_df.groupby('Frame')['x'].idxmin()]
        highest_x_per_frame = filtered_df.loc[filtered_df.groupby('Frame')['x'].idxmax()]

        # Compute the averages
        avg_lowest_x = lowest_x_per_frame['x'].mean()
        avg_highest_x = highest_x_per_frame['x'].mean()

        return avg_lowest_x, avg_highest_x
    
    # Function to plot on/off ball possession positioning of teams
    def plot_possession_metrica(
        self,
        tidy_data: pd.DataFrame, 
        poss_data: pd.DataFrame, 
        target_team: str, 
        pitch_fill: str = "black", 
        pitch_lines_col: str = "#7E7D7D",
        pitch_type: str = 'opta', 
        save: bool = True,
        text_font = "Century Gothic",
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
        text_font : str, optional
            Font used for text in the plot (default is 'Century Gothic').
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
        possession_frames_home = self._generate_possession_frames_metrica(home_poss)
        possession_frames_away = self._generate_possession_frames_metrica(away_poss)

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
            team_df_line_inposs, team_high_xip = self._find_stats_home_metrica(team_inposs, gk=11)
            team_df_line_oposs, team_high_xop = self._find_stats_home_metrica(team_outposs, gk=11)
        else:
            team_df_line_inposs, team_high_xip = self._find_stats_away_metrica(team_inposs, gk=25)
            team_df_line_oposs, team_high_xop = self._find_stats_away_metrica(team_outposs, gk=25)

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
            plt.text(row['x'], row['y'], str(int(row['player'])), font=text_font, fontsize=9, ha='center', va='center', weight="bold", color='white', zorder=3)

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
            plt.text(row['x'], row['y'], str(int(row['player'])), font=text_font, fontsize=9, ha='center', va='center', weight="bold", color='black', zorder=3)

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
                va = 'bottom', ha = 'left',fontname =text_font,
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
                va = 'bottom', ha = 'left',fontname =text_font,
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
            va = 'bottom', ha = 'left',fontname =text_font, weight = 'bold',
            fontsize = 13,color ='white'
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

        df = pd.DataFrame(data_rows)
        
        # Round the specified columns
        columns_to_round = ['OFF', 'ON']
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

        # Create the statistics table
        self._create_stat_table_metrica(df,text_font)

        # Overlay the table onto the pitch
        im1 = plt.imread("images/table.png")
        if target_team == "home":
            ax_image = add_image(im1, fig, left=0.144, bottom=0.57, width=0.23, height=0.23)
        else:
            ax_image = add_image(im1, fig, left=0.65, bottom=0.57, width=0.23, height=0.23)
    
        if save == True:
            plt.savefig(f'images/{target_team}_possession.png', bbox_inches='tight', dpi=300) 




    def _create_stat_table_metrica(self,df: pd.DataFrame,text_font : str) -> None:
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
    def player_heatmap_metrica(
        self,
        tidy_data: pd.DataFrame, 
        poss_data: pd.DataFrame, 
        target_team: str, 
        target_player: int, 
        text_font : str,
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
        text_font : str,
            Font used for text in the plot
        home_team_col : str, optional
            Color representing the home team (default is '#0A97B0').
        away_team_col : str, optional
            Color representing the away team (default is '#A04747').

        Returns:
        --------
        None. Displays and saves the heatmaps.
        """
        # Subset data of particular team                  
        stats_df = tidy_data[tidy_data["team"] == target_team].copy()
            
        # Find frames where a team has possession
        home_poss = poss_data[poss_data["Team"] == "Home"] 
        away_poss = poss_data[poss_data["Team"] == "Away"]

        # Store possession frames as list
        possession_frames_home = self._generate_possession_frames_metrica(home_poss)
        possession_frames_away = self._generate_possession_frames_metrica(away_poss)

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
    def player_movement_stats_metrica(self,tidy_data: pd.DataFrame,text_font : str = "Century Gothic", save: bool = True) -> pd.DataFrame:
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
            cell.set_text_props(color='black', fontname=text_font) 

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
            cell.set_text_props(color='black', fontname=text_font) 

        table.auto_set_font_size(False)
        table.set_fontsize(12) 
        table.scale(1.2, 1.2)  
        
        if save:
            plt.savefig('images/away_player_table.png', bbox_inches='tight', dpi=300) 

        return output





    # Plot graph of player speed
    def player_stats_graph_metrica(self,player_stats: pd.DataFrame, target_team: str,text_font : str = "Century Gothic", save: bool = True) -> None:
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
        df = player_stats[player_stats["team"] == target_team].sort_values(by="distance_km", ascending=True)

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
        ax.legend(title='Speed Range', loc='upper right', fontsize=10, facecolor='white', edgecolor='white')


        if save:
            plt.savefig(f'images/{target_team}_player_speed_graph.png', bbox_inches='tight', dpi=300)

        # Display the plot
        plt.tight_layout()
        plt.show()





    # Find sprints data for player
    def sprints_info_metrica(self,tidy_data: pd.DataFrame, target_team: str = "home") -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        conv_data = conv_data.copy()
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
    def plot_player_sprints_metrica(
        self,
        tidy_data: pd.DataFrame,
        sprints_data: pd.DataFrame,
        target_team: str,
        target_player: int,
        text_font : str = "Century Gothic",
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
        text_font : str, optional
            Font used for text in the plot (default is 'Century Gothic').
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
                plt.text(row['x'], row['y'] + 1.2, f"{row['minutes']:.2f}", font=text_font, fontsize=7,
                        ha='center', va='center', weight="bold", color='white', zorder=3)
                plt.scatter(row['x'], row['y'], s=7, alpha=1, c="white", marker="o", zorder=3)
            if row['Frame'] in list(player_ends["n"]):
                plt.scatter(row['x'], row['y'], s=5, alpha=1, c=color_main, marker="x", zorder=3)

        str_text = f"Sprints by Player Number <{target_player}>"
        fig_text(x=0.145, y=0.765, s=str_text, highlight_textprops=[{'color': color_main, 'weight': 'bold'}],
                va='bottom', ha='left', fontname=text_font, weight='bold', fontsize=13, color='white')

        str_text = "Label denotes Time when sprints start"
        fig_text(x=0.145, y=0.745, s=str_text, va='bottom', ha='left', fontname=text_font, fontsize=10, color='white')

        if target_team == "home":
            fig_text(x=0.503, y=0.237, s="Direction of Play", va='bottom', ha='left', fontname=text_font, fontsize=9, color='white')
            arrow_location = (48, -2.8)
            plt.arrow(arrow_location[0], arrow_location[1], -3, 0, shape='full', color='white', linewidth=4, head_width=0.2, head_length=0.2)
        else:
            fig_text(x=0.42, y=0.237, s="Direction of Play", va='bottom', ha='left', fontname=text_font, fontsize=9, color='white')
            arrow_location = (52, -2.8)
            plt.arrow(arrow_location[0], arrow_location[1], 3, 0, shape='full', color='white', linewidth=4, head_width=0.2, head_length=0.2)

        if save:
            plt.savefig(f'images/player_{target_player}_sprints.png', bbox_inches='tight', dpi=300)

        plt.show()






    # Function to generate animation of range of sprint frames
    def sprint_animate_metrica(
        self,
        tidy_data: pd.DataFrame,
        poss_data : pd.DataFrame,
        target_player: int,
        frame_start: int,
        frame_end: int,
        save: bool = True,
        text_font : str = "Century Gothic",
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
        text_font : str, optional
            Text font used in the plot (default is 'Century Gothic').
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
        frame_df = frame_df.groupby('Frame').apply(self._player_possession).reset_index(drop=True)  # Determine ball possession
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
                    plt.text(row['x'], row['y'], str(row['player']), font=text_font, fontsize=9, ha='center', va='center', weight="bold", color='white', zorder=4)
                    plt.arrow(row['x'], row['y'], dx * speed, dy * speed, head_width=0.5, head_length=0.5, fc='white', ec='white', zorder=2)
                else:
                    magnitude = np.sqrt(dx**2 + dy**2)
                    plt.arrow(row['x'], row['y'], dx / magnitude * 2, dy / magnitude * 2, head_width=0.5, head_length=0.5, fc='yellow', ec='yellow', zorder=2)


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
            minute_value = frame_data['minutes'].values[0]
            minutes = int(minute_value)
            seconds = int((minute_value - minutes) * 60)
            str_text = f"Time - <{minutes:02d}:{seconds:02d}>"

            fig_text(
                x=0.425, y=0.76,
                s=str_text, highlight_textprops=[{'color': '#FFD230', 'weight': 'bold'}],
                va='bottom', ha='left', fontname=text_font, weight='bold',
                fontsize=25, color='white', path_effects=[path_effects.Stroke(linewidth=0.8, foreground="#BD8B00"), path_effects.Normal()]
            )

            # Find current ball possession
            possession = self._team_possession_metrica(frame, poss_data)
            color_now = home_team_col if possession == "Home" else away_team_col if possession == "Away" else "white"
            str_text1 = f"Current Ball Possession: <{possession}> "
            fig_text(
                x=0.41, y=0.74,
                highlight_textprops=[{'color': color_now, 'weight': 'bold'}],
                s=str_text1,
                va='bottom', ha='left',
                fontname=text_font, weight='bold',
                fontsize=12, color='white'
            )

        ani = FuncAnimation(fig, sprint_plot, frames=frame_df['Frame'].unique(), repeat=False)
        if save:
            if video_writer =="gif":
                ani.save(f'videos/{target_player}_sprint_animation.gif', writer='pillow', fps=25, dpi=100)
            elif video_writer =="mp4":
                ani.save(f'videos/{target_player}_sprint_animation.mp4', writer='ffmpeg', fps=25, bitrate=2000, dpi=100)












