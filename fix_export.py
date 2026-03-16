with open("scripts/amp/play.py", "r") as f:
    text = f.read()

import re
text = text.replace("export_policy_as_jit(policy_nn", "export_policy_as_jit(runner.alg.actor")
text = text.replace("export_policy_as_onnx(policy_nn", "export_policy_as_onnx(runner.alg.actor")

with open("scripts/amp/play.py", "w") as f:
    f.write(text)
