"""
utilities/qr_utils/engines/__init__.py
----------------------------------------
Exposes the ordered list of QR detection engine functions.
Each engine's detect() takes a PIL Image and returns list[str] | None.
"""

from .zxing_engine import detect as zxing_detect

# from .pyzbar_engine import detect as pyzbar_detect
# from .qreader_engine import detect as qreader_detect
# from .opencv_engine import detect as opencv_detect

# Strict processing order as required
ENGINES = [
    ("zxing", zxing_detect),
    # ("pyzbar",  pyzbar_detect),
    # ("qreader", qreader_detect),
    # ("opencv",  opencv_detect),
]
