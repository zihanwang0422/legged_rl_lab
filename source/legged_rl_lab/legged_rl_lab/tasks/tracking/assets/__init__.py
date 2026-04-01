import os

# Asset directory - points to the assets directory within the tracking task
# Users should place their assets (URDF, USD, motion files, etc.) here
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
