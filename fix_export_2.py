with open("scripts/amp/play.py", "r") as f:
    text = f.read()

import re
old_text = """    # export policy to jit / onnx for deployment
    normalizer = None"""

# Notice how _TorchPolicyExporter strictly expects an object with `.actor` or `.student` or `_actor`
# Because we changed `runner.alg.actor_critic` (which has .actor) to `runner.alg.actor` in our fix, 
# _TorchPolicyExporter(runner.alg.actor, ...) looks for runner.alg.actor.actor which doesn't exist.
# Let's wrap our local actor in a dummy module just for the exporter, or better yet, since rsl_rl's exporter 
# might crash without standard structures, we can just comment out the export during play if we aren't explicitly deploying,
# OR we pass runner.alg which has `.actor`. 

new_text = """    # export policy to jit / onnx for deployment
    normalizer = None"""

text = text.replace('export_policy_as_jit(runner.alg.actor', 'export_policy_as_jit(runner.alg')
text = text.replace('export_policy_as_onnx(runner.alg.actor', 'export_policy_as_onnx(runner.alg')


with open("scripts/amp/play.py", "w") as f:
    f.write(text)
