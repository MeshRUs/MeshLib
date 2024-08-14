import functools
import itertools
import os
import platform
import shutil
import subprocess
import sys

from argparse import ArgumentParser
from pathlib import Path
from string import Template


MODULES = [
    "mrmeshpy",
    "mrmeshnumpy",
    "mrviewerpy",
]

WHEEL_SCRIPT_DIR = Path(__file__).parent.resolve()
WHEEL_ROOT_DIR = WHEEL_SCRIPT_DIR / "meshlib"
WHEEL_SRC_DIR = WHEEL_ROOT_DIR / "meshlib"
SOURCE_DIR = (WHEEL_SCRIPT_DIR / ".." / "..").resolve()

SYSTEM = platform.system()
LIB_EXTENSION = {
    'Darwin': ".so",
    'Linux': ".so",
    'Windows': ".pyd",
}[SYSTEM]
LIB_DIR = {
    'Darwin': SOURCE_DIR / "build" / "Release" / "bin" / "meshlib",
    'Linux': SOURCE_DIR / "build" / "Release" / "bin" / "meshlib",
    'Windows': SOURCE_DIR / "source" / "x64" / "Release",
}[SYSTEM]


def install_packages():
    packages = [
        "build",
        "pybind11-stubgen",
        "setuptools",
        "typing-extensions",
        "wheel",
    ]

    platform_specific_packages = {
        'Darwin': [
            "delocate==0.10.7",
        ],
        'Linux': [
            "auditwheel",
        ],
        'Windows': [
            "delvewheel",
        ],
    }
    packages += platform_specific_packages[SYSTEM]

    subprocess.check_call(
        ["pip", "install", "--upgrade", "pip"]
    )
    subprocess.check_call(
        ["pip", "install", "--upgrade", *packages]
    )


def setup_workspace(version, modules):
    if WHEEL_ROOT_DIR.exists():
        shutil.rmtree(WHEEL_ROOT_DIR)

    WHEEL_SRC_DIR.mkdir(parents=True)

    print("Copying LICENSE and readme.md")
    shutil.copy(SOURCE_DIR / "LICENSE", WHEEL_SRC_DIR)
    shutil.copy(SOURCE_DIR / "readme.md", WHEEL_SRC_DIR)

    # create empty file
    with open(WHEEL_SRC_DIR / "__init__.py", 'w'):
        pass

    print(f"Copying {SYSTEM} files...")
    for module in modules:
        lib = LIB_DIR / f"{module}{LIB_EXTENSION}"
        print(lib)
        shutil.copy(lib, WHEEL_SRC_DIR)

    shutil.copy(WHEEL_SCRIPT_DIR / "pyproject.toml", WHEEL_ROOT_DIR)
    #shutil.copy(WHEEL_SCRIPT_DIR / "setup.py", WHEEL_ROOT_DIR)

    package_files = itertools.chain.from_iterable(
        [f"{module}{LIB_EXTENSION}", f"{module}.pyi"]
        for module in modules
    )
    with open(WHEEL_SCRIPT_DIR / "setup.cfg.in", 'r') as config_template_file:
        config = Template(config_template_file.read()).substitute(
            VERSION=version,
            PACKAGE_DATA=", ".join(package_files),
        )
    with open(WHEEL_ROOT_DIR / "setup.cfg", 'w') as config_file:
        config_file.write(config)


def generate_stubs(modules):
    os.chdir(WHEEL_ROOT_DIR)
    for module in modules:
        subprocess.check_call(
            ["pybind11-stubgen", "--exit-code", "--output-dir", ".", f"meshlib.{module}"],
            env={
                'PYTHONPATH': ".",
            },
        )


def build_wheel():
    os.chdir(WHEEL_ROOT_DIR)
    subprocess.check_call(
        [sys.executable, "-m", "build", "--wheel"]
    )

    wheel_file = list(WHEEL_ROOT_DIR.glob("dist/*.whl"))[0]

    if SYSTEM == "Linux":
        # see also: https://github.com/mayeut/pep600_compliance
        manylinux_version = "2_31"

        os.chdir(WHEEL_ROOT_DIR)
        subprocess.check_call(
            [
                sys.executable, "-m", "auditwheel",
                "repair",
                "--plat", f"manylinux_{manylinux_version}_{platform.machine()}",
                wheel_file
            ]
        )

    elif SYSTEM == "Windows":
        os.chdir(SOURCE_DIR)
        subprocess.check_call(
            [
                sys.executable, "-m", "delvewheel",
                "repair",
                # We use --no-dll "msvcp140.dll;vcruntime140_1.dll;vcruntime140.dll" here to avoid strange conflict
                # that happens if we pack these dlls into whl.
                # Another option is to use --no-mangle "msvcp140.dll;vcruntime140_1.dll;vcruntime140.dll"
                # to pack these dlls with original names and let system solve conflicts on import
                # https://stackoverflow.com/questions/78817088/vsruntime-dlls-conflict-after-delvewheel-repair
                "--no-dll", "msvcp140.dll;vcruntime140_1.dll;vcruntime140.dll",
                "--add-path", LIB_DIR,
                wheel_file
            ]
        )

    elif SYSTEM == "Darwin":
        os.chdir(WHEEL_ROOT_DIR)
        subprocess.check_call(
            ["delocate-path", "meshlib"]
        )
        os.chdir(SOURCE_DIR)
        subprocess.check_call(
            ["delocate-wheel", "-w", ".", "-v", wheel_file]
        )


if __name__ == "__main__":
    csv = functools.partial(str.split, sep=",")

    parser = ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--modules", type=csv, default=MODULES)
    args = parser.parse_args()

    try:
        install_packages()
        setup_workspace(version=args.version, modules=args.modules)
        generate_stubs(modules=args.modules)
        build_wheel()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
