MS_DT = 0.04  # Metrica Sports delta time
MS_LAG_SMOOTH = 20  # n° samples as smoothing windows 
PLAYER_MAX_SPEED = 12  # [m/s]
WALK_JOG_THRESHOLD = 2  # Speed threshold for walking
JOG_RUN_THRESHOLD = 4  # Speed threshold for jogging 
RUN_SPRINT_THRESHOLD = 7  # Speed threshold for sprint
SPRINTS_WINDOW = int(1/MS_DT)
SPRINTS_WINDOW = int(SPRINTS_WINDOW) 