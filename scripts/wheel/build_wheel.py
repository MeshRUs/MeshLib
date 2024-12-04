import functools
import os
import platform
import shutil
import subprocess
import sys

from argparse import ArgumentParser
from pathlib import Path
from string import Template

import build_constans
import create_stubs

CONSTANTS = build_constans.get_build_consts()
MODULES = CONSTANTS['MODULES']
WHEEL_SCRIPT_DIR = CONSTANTS['WHEEL_SCRIPT_DIR']
WHEEL_ROOT_DIR = CONSTANTS['WHEEL_ROOT_DIR']
WHEEL_SRC_DIR = CONSTANTS['WHEEL_SRC_DIR']
SOURCE_DIR = CONSTANTS['SOURCE_DIR']
SYSTEM = CONSTANTS['SYSTEM']
LIB_EXTENSION = CONSTANTS['LIB_EXTENSION']
LIB_DIR = CONSTANTS['LIB_DIR']

def install_packages():
    create_stubs.install_packages()

    packages = [
        "build",
        "setuptools",
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


def setup_workspace(version, modules, plat_name):
    if WHEEL_ROOT_DIR.exists():
        shutil.rmtree(WHEEL_ROOT_DIR)

    WHEEL_SRC_DIR.mkdir(parents=True)

    create_stubs.setup_workspace(modules, False)

    print("Copying LICENSE and readme.md")
    shutil.copy(SOURCE_DIR / "LICENSE", WHEEL_ROOT_DIR)
    shutil.copy(SOURCE_DIR / "readme.md", WHEEL_ROOT_DIR)

    print("Copying resource files...")
    shutil.copy(SOURCE_DIR / "source" / "MRViewer" / "MRDarkTheme.json", WHEEL_SRC_DIR)
    shutil.copy(SOURCE_DIR / "source" / "MRViewer" / "MRLightTheme.json", WHEEL_SRC_DIR)
    shutil.copy(SOURCE_DIR / "thirdparty" / "fontawesome-free" / "fa-solid-900.ttf", WHEEL_SRC_DIR)
    shutil.copy(SOURCE_DIR / "thirdparty" / "Noto_Sans" / "NotoSansSC-Regular.otf", WHEEL_SRC_DIR)

    shutil.copy(WHEEL_SCRIPT_DIR / "pyproject.toml", WHEEL_ROOT_DIR)

    # generate setup.cfg
    package_files = [
        "MRDarkTheme.json",
        "MRLightTheme.json",
        "fa-solid-900.ttf",
        "NotoSansSC-Regular.otf"
    ]
    for module in modules:
        package_files += [
            f"{module}{LIB_EXTENSION}",
            f"{module}.pyi",
        ]
    with open(WHEEL_SCRIPT_DIR / "setup.cfg.in", 'r') as config_template_file:
        config = Template(config_template_file.read()).substitute(
            VERSION=version,
            PACKAGE_DATA=", ".join(package_files),
            PYTHON_TAG=f"py{sys.version_info.major}{sys.version_info.minor}",
            PLAT_NAME=plat_name,
        )
    with open(WHEEL_ROOT_DIR / "setup.cfg", 'w') as config_file:
        config_file.write(config)


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

        print("Wheel files are ready:")
        for repaired_wheel_file in WHEEL_ROOT_DIR.glob("wheelhouse/meshlib-*.whl"):
            print(repaired_wheel_file)

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
    parser.add_argument("--plat-name", default="any")
    args = parser.parse_args()

    try:
        install_packages()
        setup_workspace(version=args.version, modules=args.modules, plat_name=args.plat_name)
        create_stubs.generate_stubs(modules=args.modules)
        build_wheel()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
