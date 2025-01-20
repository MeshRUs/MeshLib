import os
import sys
import platform
import argparse

parser = argparse.ArgumentParser(description="Python Test Script")

parser.add_argument("-cmd", dest="cmd", type=str, help='Overwrite python run cmd')
parser.add_argument('-multi-cmd', dest='multi_cmd', action='store_true', help='Repeat tests several times, with python versions taken from `python_versions.txt`. Replaces `-cmd`.')
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

for py_cmd in python_cmds:
    # remove meshlib package if installed to not shadow dynamically attached
    os.system(py_cmd + " -m pip uninstall -y meshlib")

    print(py_cmd + " " + pytest_cmd)
    res = os.system(py_cmd + " " + pytest_cmd)

if res != 0:
    sys.exit(1)
