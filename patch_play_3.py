with open('scripts/amp/play.py', 'r') as f:
    content = f.read()

old_text = 'parser.add_argument("--task", type=str, default=None, help="Name of the task.")'
new_text = 'parser.add_argument("--task", type=str, default=None, help="Name of the task.")\nparser.add_argument("--motion_file", type=str, default=None, help="Path to motion file or directory.")'

content = content.replace(old_text, new_text)

with open('scripts/amp/play.py', 'w') as f:
    f.write(content)
