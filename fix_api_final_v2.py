import sys

with open("frontend/api.py", "r") as f:
    lines = f.readlines()

content = []
in_app_block = False

# Remove the problematic static mounting block I added earlier
skip = False
for line in lines:
    if line.strip().startswith("# Serve static files from the frontend directory"):
        skip = True
        continue
    if skip:
        if line.strip() == "" or line.strip().startswith("@app.get") or "FileResponse" in line or "RedirectResponse" in line or "app.mount" in line or "return FileResponse" in line:
            continue
        else:
            skip = False
    
    # Also clean up the double imports if they exist
    if "from fastapi.responses import FileResponse, RedirectResponse" in line:
        continue
        
    content.append(line)

# Final cleanup of any orphaned serve_html or read_index
final_content = []
i = 0
while i < len(content):
    line = content[i]
    if "@app.get(\"/\")" in line or "@app.get(\"/{path_name}\")" in line:
        # Skip the next 4-5 lines of the function
        i += 4
        while i < len(content) and content[i].startswith("    "):
            i += 1
        continue
    final_content.append(line)
    i += 1

# Add imports at top if missing
import_static = "from fastapi.staticfiles import StaticFiles\n"
import_responses = "from fastapi.responses import FileResponse, RedirectResponse\n"
if import_static not in final_content:
    final_content.insert(29, import_static)
if import_responses not in final_content:
    final_content.insert(30, import_responses)

# Add static mount at the very end
final_content.append("\n# Static files mounting (Root)\n")
final_content.append("@app.get(\"/\")\nasync def redirect_to_index():\n    return RedirectResponse(url=\"/index.html\")\n\n")
final_content.append("app.mount(\"/\", StaticFiles(directory=\"frontend\", html=True), name=\"frontend\")\n")

with open("frontend/api.py", "w") as f:
    f.writelines(final_content)
