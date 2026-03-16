with open("scripts/amp/play.py", "r") as f:
    text = f.read()

import re

# `exporter.py` in rsl_rl still has expectations about `is_recurrent`
# It's safest to simply comment out the exporter lines in play.py, since
# we are just playing/visualizing the model, not rigorously exporting for deployment here.
# Most deploy architectures have dedicated `export.py` scripts anyway.

# remove export_policy_as_jit and export_policy_as_onnx lines
text = re.sub(r'export_policy_as_jit\([^)]*\)', 'print("Skipped JIT export for custom AMP model.")', text)
text = re.sub(r'export_policy_as_onnx\([^)]*\)', 'print("Skipped ONNX export for custom AMP model.")', text)

with open("scripts/amp/play.py", "w") as f:
    f.write(text)
