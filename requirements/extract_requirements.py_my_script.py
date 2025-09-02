import ast
import sys
import pkg_resources
import importlib.util

def extract_imports(file_path):
    with open(file_path, "r") as file:
        node = ast.parse(file.read(), filename=file_path)

    imports = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                imports.add(n.module.split('.')[0])
    return sorted(imports)

def get_installed_packages(imports):
    packages = {}
    for name in imports:
        try:
            spec = importlib.util.find_spec(name)
            if spec and spec.origin and 'site-packages' in spec.origin:
                dist = pkg_resources.get_distribution(name)
                packages[dist.project_name] = dist.version
        except Exception:
            pass  # Skip standard libraries or not-found packages
    return packages

def write_requirements(packages, output="requirements.txt"):
    with open(output, "w") as f:
        for name, version in sorted(packages.items()):
            f.write(f"{name}=={version}\n")
    print(f"✅ requirements.txt created with {len(packages)} packages.")

if __name__ == "__main__":
    script_path = './your_script.py'
    imports = extract_imports(script_path)
    packages = get_installed_packages(imports)
    write_requirements(packages)
