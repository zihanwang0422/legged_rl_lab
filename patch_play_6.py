with open('scripts/amp/play.py', 'r') as f:
    content = f.read()

import re
content = re.sub(r'"units":', '"hidden_dims":', content)

with open('scripts/amp/play.py', 'w') as f:
    f.write(content)
