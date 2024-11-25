import os
import shutil
import sys

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
path_to_objects = os.path.join( base_path, "source/x64/Release/" )
print( path_to_objects )
path_to_spec = os.path.join(base_path, 'Package.nuspec')
path_to_targets = os.path.join(base_path, 'MeshLib.targets')
print( path_to_spec )
print('\n')

path_to_license = os.path.join(base_path, 'LICENSE')
shutil.copy(path_to_license, path_to_license + '.txt')

excluded_modules = ['MRCommonPlugins', 'MRCuda', 'MRViewer', 'MRMeshViewer', 'MRTest', 'MRTestC']
path_to_copyright_header = os.path.join(os.path.dirname(os.path.abspath(__file__)),'copyright_header.txt')
copyright_header = open(path_to_copyright_header,'r').read()[3:]

fTargets = open(path_to_targets, 'w')
fTargets.write('<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">\n')
fTargets.write('\t<ItemGroup>\n')


f = open(path_to_spec, 'w')
f.write('<?xml version="1.0" encoding="utf-8"?>\n')
f.write('<package>\n')
f.write('\t<metadata>\n')

f.write('\t\t<id>MeshLib</id>\n')

f.write('\t\t<version>')
f.write(sys.argv[1][1:])
f.write('\t\t</version>\n')

f.write('\t\t<authors>AMV Consulting</authors>\n')
f.write('\t\t<owners>AMV Consulting</owners>\n')
f.write('\t\t<projectUrl>https://meshlib.io</projectUrl>\n')
f.write('\t\t<description>Mesh processing library</description>\n')

f.write('\t\t<releaseNotes>https://github.com/MeshInspector/MeshLib/releases</releaseNotes>\n')

f.write('\t\t<copyright>')
f.write(copyright_header)
f.write('</copyright>\n')

f.write('\t\t<dependencies>\n')
f.write('\t\t\t<group targetFramework="netstandard2.0"/>\n')
f.write('\t\t</dependencies>\n')

f.write('\t\t<icon>images/MeshInspector_icon.png</icon>\n')
f.write('\t\t<license type="file">LICENSE.txt</license>\n')
f.write('\t\t<readme>docs/readme_dotnet.md</readme>\n')
f.write('\t</metadata>\n')

f.write('\t<files>\n')
f.write('\t\t<file src="./macos/MeshInspector_icon.png" target="images/"></file>\n')
f.write('\t\t<file src="./LICENSE.txt" target=""></file>\n')
f.write('\t\t<file src="./readme_dotnet.md" target="docs/"></file>\n')
folder = os.walk(path_to_objects)
anyDllIsFound = False
for address, dirs, files in folder:
	for file in files:
		if file.startswith('nunit'):
			continue
            
		if (file.endswith('.dll') and not any(map(file.startswith, excluded_modules)) and not file.startswith('System') and not file.startswith('MRDotNet')):
			anyDllIsFound = True
			src = os.path.join(address,file)
			print(src)
			f.write('\t\t<file src="./source/x64/Release/')
			f.write(file)
			f.write('" target="content/"></file>\n')
			fTargets.write('\t\t<None Include="$(MSBuildThisFileDirectory)\\..\\content\\')
			fTargets.write(file)
			fTargets.write('">\n')
			fTargets.write('\t\t\t<Link>')
			fTargets.write(file)
			fTargets.write('</Link>\n')
			fTargets.write('\t\t\t<Visible>false</Visible>\n')
			fTargets.write('\t\t\t<CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>\n')
			fTargets.write('\t\t</None>\n')
		elif file == 'MRDotNet.dll':
			src = os.path.join(address,file)
			print(src)
			f.write('\t\t<file src="./source/x64/Release/')
			f.write(file)
			f.write('" target="lib/netstandard2.0/"></file>\n')

if not anyDllIsFound:
    raise Exception("No DLLs found")
    
fTargets.write('\t</ItemGroup>\n')
fTargets.write('</Project>\n')
fTargets.close()

f.write('\t\t<file src="./MeshLib.targets" target="build/"></file>\n')

            
f.write('\t</files>\n')
f.write('</package>\n')
f.close()
