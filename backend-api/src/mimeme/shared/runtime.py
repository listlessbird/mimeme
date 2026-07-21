from mimeme.shared.config import Settings

# FROZEN process-scoped settings for unconverted activity/worker feature code.
# No new callers. Owners migrate to injected Settings: plans 003, 005, 006, 008.
settings = Settings()
