set VCPKG_DEFAULT_TRIPLET=x64-windows-meshlib

for /f "delims=" %%i in ('where vcpkg') do set vcpkg_path=%%~dpi
if not exist "%vcpkg_path%downloads" mkdir "%vcpkg_path%downloads"
copy "%~dp0vcpkg\downloads\*" "%vcpkg_path%downloads"

REM Set the binary sources to use the S3 bucket
set VCPKG_BINARY_SOURCES=clear;x-aws,s3://vcpkg-export/2023.04.15/test3/,readwrite;

vcpkg install vcpkg-cmake:x64-windows-meshlib vcpkg-cmake-config:x64-windows-meshlib --debug --x-abi-tools-use-exact-versions
vcpkg install cpr --debug --x-abi-tools-use-exact-versions