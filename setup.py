from setuptools import setup, find_packages


def get_requirements() -> list:
    """get requirements from PaddleCFD/requirements.txt"""
    req_list = []
    with open("requirements.txt", "r") as f:
        req_list = f.read().splitlines()
    return req_list


if __name__ == "__main__":
    setup(
        name='ppcfd',
        version='0.1.0',
        packages=find_packages(
            exclude=(
                "config",
                "doc",
                "examples",
            )
        ),
        description='PaddleCFD is a deep learning toolkit for surrogate modeling, equation discovery, shape optimization and flow-control strategy discovery in the field of fluid mechanics.',
        long_description=open('README.md', encoding="utf-8").read(),
        long_description_content_type='text/markdown',
        author='PaddlePaddle AI4S',
        author_email='paddlescience@baidu.com',
        url='https://github.com/PaddlePaddle/PaddleCFD',
        install_requires=get_requirements(),
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: Apache-2.0 license',
            'Operating System :: OS Independent',
        ],
        python_requires='>=3.10',
    )