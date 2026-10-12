import logging
import sys

import packaging.version
import paddle

expected_paddle = "2.0.0"
expected_paddlevision = "0.15.0"
expected_kappabenchmark = "0.0.10"
expected_kappaconfig = "1.0.29"
expected_kappadata = "1.3.78"
expected_kappamodules = "0.1.24"
expected_kappaprofiler = "1.0.11"
expected_kappaschedules = "0.0.18"
expected_timm = "0.9.2"
expected_paddlemetrics_version = "0.11.0"
expected_python_major = 3
expected_python_minor = 9


def check_versions(verbose):
    log_fn = logging.info if verbose else lambda _: None
    log_fn("------------------")
    log_fn("VERSION CHECK")
    executable_log_fn = logging.info if verbose else print
    executable_log_fn(f"executable: {sys.executable}")
    py_version = sys.version_info
    msg = f"upgrade python ({py_version.major}.{py_version.minor} < {expected_python_major}.{expected_python_minor})"
    assert (
        py_version.major >= expected_python_major
        and py_version.minor >= expected_python_minor
    ), msg
    log_fn(f"python version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    log_fn(f"paddle version: {paddle.__version__}")
    assert packaging.version.parse(paddle.__version__) >= packaging.version.parse(
        expected_paddle
    )

    # def _check_pip_dependency(actual_version, expected_version, pip_dependency_name):
    #     assert packaging.version.parse(actual_version) >= packaging.version.parse(
    #         expected_version
    #     ), f"upgrade {pip_dependency_name} with 'pip install {pip_dependency_name} --upgrade' ({actual_version} < {expected_version})"
    #     log_fn(f"{pip_dependency_name} version: {actual_version}")

    # import kappabenchmark

    # _check_pip_dependency(
    #     kappabenchmark.__version__, expected_kappabenchmark, "kappabenchmark"
    # )
    # import kappaconfig

    # _check_pip_dependency(kappaconfig.__version__, expected_kappaconfig, "kappaconfig")
    # import kappadata

    # _check_pip_dependency(kappadata.__version__, expected_kappadata, "kappadata")
    # import KappaModules.kappamodules

    # _check_pip_dependency(
    #     KappaModules.kappamodules.__version__, expected_kappamodules, "kappamodules"
    # )
    # import kappaprofiler

    # _check_pip_dependency(
    #     KappaModules.kappaprofiler.__version__, expected_kappaprofiler, "kappaprofiler"
    # )
    # import kappaschedules

    # _check_pip_dependency(
    #     KappaModules.kappaschedules.__version__, expected_kappaschedules, "kappaschedules"
    # )
    # import paddlemetrics

    # _check_pip_dependency(
    #     paddlemetrics.__version__, expected_paddlemetrics_version, "paddlemetrics"
    # )
