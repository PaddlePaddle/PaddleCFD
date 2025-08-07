import sys
import os


def run():
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    import paddle

    sys.stdout = original_stdout
    sys.stderr = original_stderr

    paths = paddle.utils.cpp_extension.extension_utils.find_paddle_includes()
    result = ""
    for path in paths:
        result += f";{path}"

    return result


if __name__ == "__main__":
    print(run())
