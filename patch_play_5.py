with open('scripts/amp/play.py', 'r') as f:
    lines = f.readlines()

new_lines = []
motion_file_count = 0
for line in lines:
    if 'parser.add_argument("--motion_file"' in line:
        motion_file_count += 1
        if motion_file_count > 1:
            continue  # skip duplicates
    new_lines.append(line)

with open('scripts/amp/play.py', 'w') as f:
    f.writelines(new_lines)
