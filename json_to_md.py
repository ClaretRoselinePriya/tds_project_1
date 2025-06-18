import json
from pathlib import Path

def json_to_markdown(json_path, md_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    md_lines = []
    for i, item in enumerate(data, 1):
        md_lines.append(f"## Post {i}\n")
        for key, value in item.items():
            md_lines.append(f"### {key}\n{value}\n")
        md_lines.append("---\n")  # separator

    Path(md_path).write_text("\n".join(md_lines), encoding="utf-8")
    print(f"✅ Saved: {md_path}")

# Example usage
json_to_markdown("discourse_posts.json", "discourse_posts.md")
