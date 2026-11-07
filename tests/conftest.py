"""Shared pytest set-up."""
import matplotlib

# Use a non-interactive backend so plotting works on machines without a display.
matplotlib.use("Agg")
