import platform
from pathlib import Path

SYSTEM = platform.system()

MODULES = [
    "mrmeshpy",
    "mrmeshnumpy",
    "mrviewerpy",
]
if SYSTEM != "Darwin":
    MODULES.append("mrcudapy")


WHEEL_SCRIPT_DIR = Path(__file__).parent.resolve()
WHEEL_ROOT_DIR = WHEEL_SCRIPT_DIR / "meshlib"
WHEEL_SRC_DIR = WHEEL_ROOT_DIR / "meshlib"
SOURCE_DIR = (WHEEL_SCRIPT_DIR / ".." / "..").resolve()

LIB_EXTENSION = {
    'Darwin': ".so",
    'Linux': ".so",
    'Windows': ".pyd",
}[SYSTEM]
LIB_DIR = {
    'Darwin': SOURCE_DIR / "build" / "Release" / "bin",
    'Linux': SOURCE_DIR / "build" / "Release" / "bin",
    'Windows': SOURCE_DIR / "source" / "x64" / "Release",
}[SYSTEM]
LIB_DIR_MESHLIB = LIB_DIR / "meshlib"
