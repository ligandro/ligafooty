# LigaFooty

![LigaFooty Logo](https://your-logo-url.com/logo.png)

*A Python package for football tracking visualization and analytics.*

## 📌 Overview
LigaFooty is a Python package designed for **football tracking visualization**, **player movement analysis**, and **possession-based statistics**. It provides easy-to-use functions for plotting football pitch events, animations, and team performance insights.

## 🔥 Features
- ✅ **Football pitch visualization** using `mplsoccer`
- ✅ **Player tracking & movement analysis**
- ✅ **Ball possession detection**
- ✅ **Animations of player positions**
- ✅ **Convex hull, Voronoi, and Delaunay triangulation visualizations**
- ✅ **Customizable pitch themes & team colors**

## 📦 Installation
Install LigaFooty using pip:

```sh
pip install git+https://github.com/ligandro/Liga.Footy.git
```

## 🚀 Getting Started
### 1️⃣ Import LigaFooty

```python
import LigaFooty
```

### 2️⃣ Example Usage
#### **Plot Player Tracking Data**

```python
from LigaFooty import liga_plot_poss

liga_plot_poss(tidy_data, event_data, target_team='home')
```

#### **Animate a Range of Frames**

```python
from LigaFooty import liga_animate

liga_animate(tidy_data, poss_data, frame_start=1000, frame_end=1050)
```

#### **Compute Team Stats**

```python
from LigaFooty import calculate_team_stats

width, defensive_line, def_height = calculate_team_stats(team_df, 'home')
```

## 📜 Available Functions
- `liga_plot_poss(tidy_data, ed, target_team, ...)` → Plots possession-based tracking.
- `liga_animate(tidy_data, poss_data, frame_start, frame_end, ...)` → Generates animations.
- `calculate_team_stats(team_df, team_name)` → Computes pitch width and defensive line stats.
- `team_possession(frame, possession_df)` → Determines which team has possession.
- `player_possession(frame_data)` → Finds which player has the ball.
- `plot_voronoi_viz(frame_df, pitch, ax, team_colors)` → Creates a Voronoi diagram.

## 🛠 Dependencies
LigaFooty requires the following packages:

```sh
pip install numpy pandas matplotlib mplsoccer highlight_text
```

## 🤝 Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add new feature"`).
4. Push to your branch (`git push origin feature-branch`).
5. Open a pull request.

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🌟 Support & Contact
For any questions or feature requests, feel free to reach out:

📧 Email: ligandro@example.com  
🐦 Twitter: [@Ligandro22](https://twitter.com/Ligandro22)

