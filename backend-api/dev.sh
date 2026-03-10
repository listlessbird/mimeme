#!/bin/bash
# Development environment with 4-pane split view

SESSION="mimeme-dev"
STAGING_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --staging)
      STAGING_MODE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--staging]"
      exit 1
      ;;
  esac
done

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

if [ "$STAGING_MODE" = true ]; then
  # Staging mode commands

  # Pane 0 (top-left): Temporal server only
  tmux select-pane -t 0
  tmux send-keys "temporal server start-dev" C-m

  # Pane 1 (bottom-left): API Server
  tmux select-pane -t 1
  tmux send-keys "sleep 3 && uv run uvicorn api.main:app --reload" C-m

  # Pane 2 (top-right): Worker
  tmux select-pane -t 2
  tmux send-keys "sleep 5 && uv run python -m workers.worker" C-m

  # Pane 3 (bottom-right): Modal app
  tmux select-pane -t 3
  # tmux send-keys "sleep 5 && cd src && uv run modal serve modal_app.app" C-m
else
  # Default development mode commands

  # Pane 0 (top-left): Infrastructure + Temporal
  tmux select-pane -t 0
  tmux send-keys "docker compose up postgres minio -d && temporal server start-dev" C-m

  # Pane 1 (bottom-left): API Server
  tmux select-pane -t 1
  tmux send-keys "sleep 3 && uv run alembic upgrade head && uv run uvicorn api.main:app --reload" C-m

  # Pane 2 (top-right): Worker
  tmux select-pane -t 2
  tmux send-keys "sleep 5 && uv run python -m workers.worker" C-m

  # Pane 3 (bottom-right): free
  tmux select-pane -t 3
fi

# Focus on API pane
tmux select-pane -t 1

# Attach to session
tmux attach-session -t $SESSION
