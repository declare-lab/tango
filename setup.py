import os
requirement_path = "requirements.txt"
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()
setup(name="mypackage", install_requires=install_requires, [...])