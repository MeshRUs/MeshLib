import os
import sys
import platform
import argparse
import shutil

parser = argparse.ArgumentParser(description="Python Test Script")

parser.add_argument("-cmd", dest="cmd", type=str, help='Overwrite python run cmd')
parser.add_argument('-multi-cmd', dest='multi_cmd', action='store_true', help='Repeat tests several times, with python versions taken from `python_versions.txt`. Replaces `-cmd`.')
parser.add_argument('-create-venv', dest='create_venv', action='store_true', help='Create a venv and install the dependencies in it. Can combine with `-multi-cmd`.')
parser.add_argument("-d", dest="dir", type=str, help='Path to tests')
parser.add_argument("-s", dest="smoke", type=str, help='Run reduced smoke set')
parser.add_argument("-bv", dest="bindings_vers", type=str,
                    help='Version of bindings to run tests, "2" or "3"', default='3')
parser.add_argument("-a", dest="pytest_args", type=str,
                    help='Args string to be added to pytest command', default='')

args = parser.parse_args()
print(args)

python_cmds = ["py -3.11"]
platformSystem = platform.system()

if platformSystem == 'Linux':
    python_cmds = ["python3"]

    os_name = ""
    os_version = ""
    if os.path.exists('/etc/os-release'):
        lines = open('/etc/os-release').read().split('\n')
        for line in lines:
            if line.startswith('NAME='):
                os_name = line.split('=')[-1].replace('"', '')
            if line.startswith('VERSION_ID='):
                os_version = line.split('=')[-1].replace('"', '')

    if "ubuntu" in os_name.lower():
        if os_version.startswith("20"):
            python_cmds = ["python3.8"]
        elif os_version.startswith("22"):
            python_cmds = ["python3.10"]
    elif "fedora" in os_name.lower():
        if os_version.startswith("35"):
            python_cmds = ["python3.9"]
        elif os_version.startswith("37"):
            python_cmds = ["python3.11"]
        elif os_version.startswith("39"):
            python_cmds = ["python3.12"]

elif platformSystem == 'Darwin':
    python_cmds = ["python3.10"]

if args.cmd:
    python_cmds = [str(args.cmd).strip()]
elif args.multi_cmd:
    with open(os.path.dirname(os.path.realpath(__file__)) + "/mrbind-pybind11/python_versions.txt") as file:
        if platform.system() == "Windows":
            python_cmds = ["py -" + line.rstrip() for line in file]
        else:
            python_cmds = ["python" + line.rstrip() for line in file]

directory = os.getcwd()
try:
    directory = os.path.dirname(os.path.abspath(__file__))
except NameError:  # embedded python exception
    print("trying to resolve path manually...")
    directory = os.path.join(directory, "../../../MeshLib/")
    directory = os.path.join(directory, "test_python")
    print(directory)

if args.dir:
    directory = os.path.join(directory, args.dir)
else:
    directory = os.path.join(directory, "..")
    directory = os.path.join(directory, "test_python")

os.environ["MeshLibPyModulesPath"] = os.getcwd()
os.chdir(directory)

#command line to start test
pytest_cmd = "-m pytest -s -v --basetemp=../pytest_temp --durations 30"
if args.bindings_vers == '2':
    pytest_cmd += ' -m "not bindingsV3'
elif args.bindings_vers == '3':
    pytest_cmd += ' -m "not bindingsV2'
else:
    print("Error: Unknown version of bindings")
    exit(5)
if args.smoke == "true":
    pytest_cmd += f' and smoke"'
else:
    pytest_cmd += f'"'

if args.pytest_args:
    pytest_cmd += f' {args.pytest_args}'

failed = False
venv_failed = False
for py_cmd in python_cmds:
    if platform.system() == "Darwin" and shutil.which(py_cmd) is None:
        continue; # Skip if no such command. Some python versions are not supported on some macs.

    # remove meshlib package if installed to not shadow dynamically attached
    os.system(py_cmd + " -m pip uninstall -y meshlib")

    if args.create_venv:
        print("CREATING VENV --- [  " + py_cmd + " -m venv venv_" + py_cmd)
        if os.system(py_cmd + " -m venv venv_" + py_cmd) != 0:
            venv_failed = True
        if os.system(". venv_" + py_cmd + "/bin/activate && pip install pytest numpy"):
            venv_failed = True
        py_cmd_fixed = ". venv_" + py_cmd + "/bin/activate && " + py_cmd
    else:
        py_cmd_fixed = py_cmd

    print(py_cmd_fixed + " " + pytest_cmd)
    if os.system(py_cmd_fixed + " " + pytest_cmd) != 0:
        failed = True

    if args.create_venv:
        shutil.rmtree("venv_" + py_cmd);
        print("] --- DELETING VENV")

if venv_failed:
    print("ERROR: Couldn't create some of the venvs!")

if failed:
    print("ERROR: Some tests failed!")

if failed or venv_failed:
    sys.exit(1)
