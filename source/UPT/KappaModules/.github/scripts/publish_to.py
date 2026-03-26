import argparse


def get_tagname():
    parser = argparse.ArgumentParser(description='Replace version in setup.cfg with the created git tag')
    parser.add_argument('--tagname', help='tag name', required=True)
    args = parser.parse_args()
    return args.tagname


tagname = get_tagname()
assert tagname[0] == "v", "tagname must start with 'v'"
version_numbers = tagname[1:].split(".")
version_number_count = len(version_numbers)
if version_number_count == 3:
    print(f"::set-output name=publish_to_prod::true")
    print(f"::set-output name=publish_to_dev::false")
elif version_number_count == 4:
    print(f"::set-output name=publish_to_prod::false")
    print(f"::set-output name=publish_to_dev::true")
else:
    raise ValueError("version must contain 3 (for prod publish) or 4 (for test publish) numbers")
