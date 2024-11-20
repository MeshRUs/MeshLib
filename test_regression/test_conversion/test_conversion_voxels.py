from module_helper import *
from pathlib import Path
import pytest
from constants import test_files_path
from helpers.meshlib_helpers import compare_voxels


@pytest.mark.parametrize("ext", ["gav", "dcm", "vdb"])
def test_conversion_from_raw(ext, tmp_path):
    filename_in = "W48_H58_S78_V1.5e+03_1.5e+03_1.5e+03_G0_F voxels.raw"
    filename_out = f"voxels.{ext}"
    filename_ref = "voxels_ref.gav"

    input_folder = Path(test_files_path) / "conversion" / "voxels_from_raw"

    vxl_in = mrmeshpy.loadVoxelsRaw(input_folder / filename_in)

    mrmeshpy.VoxelsSave.toAnySupportedFormat(vxl_in, str(tmp_path / filename_out))

    # Verify
    vxl_ref =  mrmeshpy.VoxelsLoad.fromAnySupportedFormat(input_folder / filename_ref)[0]
    vxl_out =  mrmeshpy.VoxelsLoad.fromAnySupportedFormat(tmp_path / filename_out)[0]

    compare_voxels(vxl_out, vxl_ref)


@pytest.mark.parametrize("ext", ["gav"])
def test_conversion_to_raw(ext, tmp_path):
    filename_in = "voxels.gav"
    filename_out = "voxels.raw"
    filename_out_full = "W48_H58_S78_V1.5e+03_1.5e+03_1.5e+03_G0_F voxels.raw"


    input_folder = Path(test_files_path) / "conversion" / "voxels_to_raw"

    vxl_in = mrmeshpy.loadVoxelsGav(input_folder / filename_in)

    mrmeshpy.saveVoxelsRaw(vxl_in, str(tmp_path / filename_out))

    # Verify
    vxl_out = mrmeshpy.loadVoxelsRaw(tmp_path / filename_out_full)
    vxl_ref = mrmeshpy.loadVoxelsGav(input_folder / filename_in)
    compare_voxels(vxl_out, vxl_ref)
