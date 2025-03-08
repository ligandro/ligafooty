
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
import matplotlib.patheffects as path_effects    

def plot_base_viz(
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


def plot_convexhull_viz(
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


def plot_voronoi_viz(
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
    tracking_home = tracking_full[tracking_full['team']=="home"]
    tracking_away = tracking_full[tracking_full['team']=="away"]

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



def plot_delaunay_viz(
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


















































































































































































































































































