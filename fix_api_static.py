import sys

with open("frontend/api.py", "r") as f:
    lines = f.readlines()

new_content = []
added = False

for line in lines:
    if "from fastapi import" in line and "StaticFiles" not in line:
        new_content.append(line.replace("from fastapi import", "from fastapi import StaticFiles,"))
        continue
    
    if "app = FastAPI" in line:
        new_content.append(line)
        new_content.append("\n# Serve static files from the frontend directory\n")
        new_content.append("from fastapi.responses import FileResponse, RedirectResponse\n")
        new_content.append("app.mount(\"/static\", StaticFiles(directory=\"frontend\"), name=\"static\")\n")
        new_content.append("\n@app.get(\"/\")\ndef read_index():\n    return RedirectResponse(url=\"/index.html\")\n")
        new_content.append("\n@app.get(\"/{path_name}\")\ndef serve_html(path_name: str):\n    if path_name.endswith(\".html\"):\n        return FileResponse(f\"frontend/{path_name}\")\n    return JSONResponse(status_code=404, content={\"error\": \"Not Found\"})\n")
        added = True
        continue
    
    new_content.append(line)

with open("frontend/api.py", "w") as f:
    f.writelines(new_content)
