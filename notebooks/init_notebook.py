import sys, os

# Move up one directory
PROJECT_ROOT = os.path.abspath("..")
# print("PROJECT ROOT is:", PROJECT_ROOT)

# Append PROJECT_ROOT to sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
