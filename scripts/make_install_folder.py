import glob
import os
import shutil
import sys

import install_tools as it

it.base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
it.path_to_install_folder = os.path.join(it.base_path,'install')
it.path_to_includes = os.path.join(it.path_to_install_folder,'include')
it.path_to_libs = os.path.join(it.path_to_install_folder,'lib')
it.path_to_app = os.path.join(it.path_to_install_folder,'app')
it.path_to_sources = os.path.join(it.base_path,'source')
it.path_to_copyright_header = os.path.join(os.path.dirname(os.path.abspath(__file__)),'copyright_header.txt')

path_to_imgui = os.path.join(os.path.join(it.base_path,'thirdparty'),'imgui')
path_to_phmap = os.path.join(os.path.join(it.base_path,'thirdparty'),'parallel-hashmap')
path_to_pybind11 = os.path.join(os.path.join(os.path.join(it.base_path,'thirdparty'),'pybind11'),'include')

not_app_extentions = ['.lib','.obj','.pdb','.obj','.exp','.iobj','.ipdb']

def vcpkg_dir():
	vcpkg_exe_dir = ""
	if len(sys.argv) > 2:
		vcpkg_exe_dir = sys.argv[2]
	else:
		vcpkg_exe_dir = os.popen("where vcpkg").read().strip()
		if "vcpkg.exe" not in vcpkg_exe_dir:
			vcpkg_exe_dir = "C:\\vcpkg"
		else:
			vcpkg_exe_dir = os.path.dirname( vcpkg_exe_dir )
	return os.path.join(os.path.join(vcpkg_exe_dir, "installed"),"x64-windows-meshlib")


vcpkg_directory = vcpkg_dir()
print (vcpkg_directory)

def prepare_includes_list():
	it.includes_src_dst.clear()
	it.includes_src_dst_thirdparty.clear()
	it.append_includes_list(os.path.join(vcpkg_directory,"include"), True)
	it.append_includes_list(it.path_to_sources)
	it.append_includes_list(path_to_phmap, True,'parallel_hashmap')
	it.append_includes_list(path_to_pybind11, True)
	it.append_includes_list(path_to_imgui, True)

def copy_includes():
	prepare_includes_list()
	for src,dst in it.includes_src_dst:
		dst_folder = os.path.dirname(dst)
		os.makedirs(dst_folder,exist_ok=True)
		shutil.copyfile(src, dst)
	it.inject_copyright()
	for src,dst in it.includes_src_dst_thirdparty:
		dst_folder = os.path.dirname(dst)
		os.makedirs(dst_folder,exist_ok=True)
		shutil.copyfile(src, dst)

def copy_app():
	shutil.copytree(os.path.join(it.path_to_sources,'x64'),it.path_to_app,dirs_exist_ok=True)
	folder = os.walk(it.path_to_app)
	for address, dirs, files in folder:
		for file in files:
			if (any(map(file.endswith, not_app_extentions))):
				os.remove(os.path.join(address,file))

def copy_lib():
	shutil.copytree(os.path.join(it.path_to_sources,'x64'),it.path_to_libs,dirs_exist_ok=True)
	shutil.copytree(os.path.join(os.path.join(vcpkg_directory,'debug'),'lib'),os.path.join(it.path_to_libs,"Debug"),dirs_exist_ok=True)
	shutil.copytree(os.path.join(vcpkg_directory,'lib'),os.path.join(it.path_to_libs,"Release"),dirs_exist_ok=True)
	folder = os.walk(it.path_to_libs)
	for address, dirs, files in folder:
		for file in files:
			if ( not any(map(file.endswith, it.lib_extentions))):
				os.remove(os.path.join(address,file))

	# Prune .pyd Python modules, but only in the Debug/Release directories, not in their subdirectories.
	# This is only needed on Windows. On Windows they are initially present both there and in `__/meshlib`, because it's hard
	# to make VS build them directly in the subdirectory. And the `.pyd` extension is only used on Windows.
	for f in glob.glob(os.path.join(it.path_to_app, "*/*.pyd")):
		os.remove(f)
	for f in glob.glob(os.path.join(it.path_to_libs, "__init__.py")):
		os.remove(f)
	for f in glob.glob(os.path.join(it.path_to_app, "*/*pybind11nonlimitedapi_meshlib_*")):
		os.remove(f)

it.prepare_includes_list = prepare_includes_list
it.copy_includes = copy_includes
it.copy_app = copy_app
it.copy_lib = copy_lib

it.main()
