import sys, os

# Move up one directory
PROJECT_ROOT = os.path.abspath("..")
print("PROJECT ROOT is:", PROJECT_ROOT)

# Append PROJECT_ROOT to sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Append src to sys.path explicitly
# SRC_PATH = os.path.join(PROJECT_ROOT, "src")
# if SRC_PATH not in sys.path:
#     print("HERE")
#     sys.path.append(SRC_PATH)
# print("SRC PATH is:", SRC_PATH)
