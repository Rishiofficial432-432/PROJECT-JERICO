import os

with open("frontend/api.py", "r") as f:
    lines = f.readlines()

content = []
for line in lines:
    if "app.mount" in line or "redirect_to_index" in line:
        continue
    content.append(line)

# Remove any existing StaticFiles or RedirectResponse routes at the end
while content and (content[-1].strip() == "" or content[-1].startswith("@app.get") or "StaticFiles" in content[-1] or "RedirectResponse" in content[-1]):
    content.pop()

# Add robust static mounting at the end
frontend_dir = os.path.abspath("frontend")
content.append("\n# Final Static File Configuration\n")
content.append("from fastapi.staticfiles import StaticFiles\n")
content.append("from fastapi.responses import RedirectResponse\n\n")
content.append("@app.get(\"/\")\nasync def root_redirect():\n    return RedirectResponse(url=\"/index.html\")\n\n")
content.append(f"app.mount(\"/\", StaticFiles(directory=\"{frontend_dir}\", html=True), name=\"frontend\")\n")

with open("frontend/api.py", "w") as f:
    f.writelines(content)
