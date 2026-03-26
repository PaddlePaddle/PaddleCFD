import argparse


def get_tagname():
    parser = argparse.ArgumentParser(description='Replace version in setup.cfg with the created git tag')
    parser.add_argument('--tagname', help='tag name', required=True)
    args = parser.parse_args()
    return args.tagname.replace("v", "")


with open("setup.cfg") as f:
    lines = f.read().splitlines()

metadata_line = -1
for i in range(len(lines)):
    if "[metadata]" in lines[i]:
        metadata_line = i

print(f"inserting version into setup.cfg after line {metadata_line}")
lines.insert(metadata_line + 1, f"version = {get_tagname()}")

# for unified access to the version number across python versions
print("inserting version into kappamodules/__init__.py")
with open("kappamodules/__init__.py", "r") as f:
    content = f.read()
with open("kappamodules/__init__.py", "w") as f:
    f.write(f"__version__ = \"{get_tagname()}\"\n\n" + content)

with open("setup.cfg", "w") as f:
    f.write("\n".join(lines))
