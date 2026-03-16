with open("scripts/amp/train.py", "r") as f:
    text = f.read()

import re
match = re.search(r'agent_dict \= agent_cfg\.to_dict\(\)(.*?)runner \= OnPolicyRunner', text, re.DOTALL)
if match:
    print(match.group(1))
