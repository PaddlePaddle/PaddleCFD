from setuptools import find_packages, setup


def get_requirements() -> list:
    """get requirements from PaddleCFD/requirements.txt"""
    req_list = []
    with open("requirements.txt", "r") as f:
        req_list = f.read().splitlines()
    return req_list


if __name__ == "__main__":
    setup(
        name="ppcfd",
        version="0.1.0.2",
        packages=find_packages(
            exclude=(
                "config",
                "doc",
                "examples",
            )
        ),
        include_package_data=True,
        description="PaddleCFD is a deep learning toolkit for surrogate modeling, equation discovery, shape optimization and flow-control strategy discovery in the field of fluid mechanics.",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        author="AI4S@PaddlePaddle",
        author_email="paddlescience@baidu.com",
        url="https://github.com/PaddlePaddle/PaddleCFD",
        install_requires=get_requirements(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
        license_file="LICENSE",
        python_requires=">=3.10",
    )
    print("PaddleCFD has been installed successfully.")
    print("Please check https://github.com/PaddlePaddle/PaddleCFD for more information.")
