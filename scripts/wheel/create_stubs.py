import functools
import os
import shutil
import subprocess
import sys

from argparse import ArgumentParser

from build_constants import *

def install_packages():
    packages = [
        "pybind11-stubgen",
        "typing-extensions",
    ]

    subprocess.check_call(
        ["pip", "install", "--upgrade", "pip"]
    )
    subprocess.check_call(
        ["pip", "install", "--upgrade", *packages]
    )


def setup_workspace( modules, clear_folder = True ):
    if clear_folder and WHEEL_ROOT_DIR.exists():
        shutil.rmtree(WHEEL_ROOT_DIR)

    if not WHEEL_ROOT_DIR.exists():
        WHEEL_SRC_DIR.mkdir(parents=True)

    init_file = LIB_DIR_MESHLIB / "__init__.py"
    if init_file.exists():
        shutil.copy(init_file, WHEEL_SRC_DIR / "__init__.py")
    else:
        shutil.copy(WHEEL_SCRIPT_DIR / "init.py", WHEEL_SRC_DIR / "__init__.py")

    print(f"Copying {SYSTEM} files...")
    for module in modules:
        lib = LIB_DIR_MESHLIB / f"{module}{LIB_EXTENSION}"
        print(lib)
        shutil.copy(lib, WHEEL_SRC_DIR)
    for pybind_shim in LIB_DIR_MESHLIB.glob("*pybind11nonlimitedapi_meshlib_*"):
        print(pybind_shim)
        shutil.copy(pybind_shim, WHEEL_SRC_DIR)

def generate_stubs(modules):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(WHEEL_ROOT_DIR)

    if SYSTEM == "Windows":
        env['PYBIND11_STUBGEN_PATH'] = str(LIB_DIR)
        pybind11_stubgen_command = [sys.executable, "..\\pybind11-stubgen.py"]
    else:
        pybind11_stubgen_command = ["pybind11-stubgen"]

    os.chdir(WHEEL_ROOT_DIR)
    for module in modules:
        subprocess.check_call(
            [*pybind11_stubgen_command, "--exit-code", "--output-dir", ".", f"meshlib.{module}"],
            env=env,
        )

if __name__ == "__main__":
    csv = functools.partial(str.split, sep=",")

    parser = ArgumentParser()
    parser.add_argument("--modules", type=csv, default=MODULES)
    args = parser.parse_args()

    try:
        install_packages()
        setup_workspace( modules=args.modules)
        generate_stubs(modules=args.modules)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
