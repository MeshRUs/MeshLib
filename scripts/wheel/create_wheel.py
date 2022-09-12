import argparse
import os
import platform
import sys
import glob
import shutil

WHEEL_SRC_DIR = os.path.join(os.getcwd(), "scripts/wheel/meshlib/meshlib/")
WHEEL_ROOT_DIR = os.path.join(os.getcwd(), "scripts/wheel/meshlib/")
WHEEL_SCRIPT_DIR = os.path.join(os.getcwd(), "scripts/wheel/")


def prepare_workspace():
    if not os.path.isdir(os.path.join(os.getcwd(),"scripts")):
        print("Please run this script from MeshLib root")
        sys.exit(1)

    if os.path.exists(WHEEL_ROOT_DIR):
        shutil.rmtree(WHEEL_ROOT_DIR)

    os.makedirs(WHEEL_SRC_DIR, exist_ok=True)
    print("Copying LICENSE and readme.md")
    shutil.copy("LICENSE", WHEEL_ROOT_DIR)
    shutil.copy("readme.md", WHEEL_ROOT_DIR)
    # create empty file
    open(os.path.join(WHEEL_SRC_DIR, "__init__.py"), "w").close()


def copy_linux_src():
    print("Copying files...")
    for file in glob.glob(r'./build/Release/bin/meshlib/mr*.so'):
        print(file)
        shutil.copy(file, WHEEL_SRC_DIR)


def copy_windows_src():
    print("Copying files...")
    for file in glob.glob(r'./source/x64/Release/*.pyd'):
        print(file)
        shutil.copy(file, WHEEL_SRC_DIR)


def setup_wheel_info(args):
    platform_system = platform.system()
    print(platform_system)
    with open(os.path.join(WHEEL_SCRIPT_DIR, "setup.py"), 'r') as input:
        data = input.readlines()

    with open(os.path.join(WHEEL_ROOT_DIR, "setup.py"), 'w') as output:
        for line in data:
            if "version=" in line:
                line = line.replace("$", args.version)
            elif "package_data=" in line:
                if platform_system == "Windpows":
                    line = line.replace("$", "pyd")
                else:
                    line = line.replace("$", "so")
            elif "Programming Language ::" in line or "python_requires=" in line:
                line = line.replace("$", str(sys.version_info[0]) + "." + str(sys.version_info[1]))
            elif "Operating System ::" in line:
                if platform_system == "Windpows":
                    line = line.replace("$", "Microsoft :: Windows :: Windows 10")
                elif platform_system == "Linux":
                    line = line.replace("$", "POSIX :: Linux")
                elif platform_system == "Darwin":
                    line = line.replace("$", "MacOS :: MacOS 10")

            output.write(line)


def parse_args():
    parser = argparse.ArgumentParser(description="Just an example",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version", help="wheel version")
    args = parser.parse_args()
    config = vars(args)
    print(config)
    return args


args = parse_args()
prepare_workspace()
copy_linux_src()
setup_wheel_info(args)
