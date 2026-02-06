#!/bin/bash
# Development environment with 4-pane split view

SESSION="findmeme-dev"

# Kill existing session if it exists
tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

# Create new session
tmux new-session -d -s $SESSION

# Split into 4 panes (2x2 grid)
tmux split-window -h    # Split horizontally (left|right)
tmux split-window -v    # Split right pane vertically
tmux select-pane -t 0   # Go back to left pane
tmux split-window -v    # Split left pane vertically

# Adjust layout to be evenly distributed
tmux select-layout tiled

# Pane 0 (top-left): Infrastructure + Temporal
tmux select-pane -t 0
tmux send-keys "docker compose up postgres minio -d && temporal server start-dev" C-m

# Pane 1 (bottom-left): API Server
tmux select-pane -t 1
tmux send-keys "sleep 3 && uv run alembic upgrade head && uv run uvicorn api.main:app --reload" C-m

# Pane 2 (top-right): CPU Worker
tmux select-pane -t 2
tmux send-keys "sleep 5 && uv run python -m workers.cpu_worker" C-m

# Pane 3 (bottom-right): GPU Worker
tmux select-pane -t 3
tmux send-keys "sleep 5 && uv run python -m workers.gpu_worker" C-m

# Focus on API pane
tmux select-pane -t 1

# Attach to session
tmux attach-session -t $SESSION
