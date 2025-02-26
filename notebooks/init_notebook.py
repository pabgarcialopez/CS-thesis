import sys, os

# Move up one directory from 'notebooks/' to 'TFG-info/'
PROJECT_ROOT = os.path.abspath("..")

# Append to sys.path so we can import `config.py` and code in `src/`
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
