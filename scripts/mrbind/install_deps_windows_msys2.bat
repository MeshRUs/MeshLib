@echo off

rem This script installs (or updates) MSYS2, and some Clang tools in it.
rem Which is everything you need to generate MRBind bindings on Windows.

rem We create a separate copy of MSYS2 at C:\msys64_meshlib_mrbind for simplicity.
rem If you don't like that, you can manually install all dependencies into
rem some other MSYS2 copy.

setlocal

if "%MSYS2_DIR%" == "" set MSYS2_DIR=C:\msys64_meshlib_mrbind
if "%CLANG_VER%" == "" set /p CLANG_VER=<%~dp0\clang_version_msys2.txt

rem ------ Ensure MSYS2 is installed

if exist %MSYS2_DIR% (
    echo MSYS2 is already installed to `%MSYS2_DIR%`.
) else (
    echo MSYS2 was NOT found at `%MSYS2_DIR%`.

    rem --- Download installer
    if exist %TMP%\msys2-base-x86_64-latest.sfx.exe (
        echo Will use existing installer at `%TMP%\msys2-base-x86_64-latest.sfx.exe`.
        echo     If it misbehaves, delete it and restart this script to download a new one. Or install manually to `%MSYS2_DIR%` and restart this script to download the required packages into it.
    ) else (
        echo Downloading installer to `%TMP%\msys2-base-x86_64-latest.sfx.exe`...
        rem This seems to automatically overwrite the file!
        powershell -Command "Invoke-WebRequest https://github.com/msys2/msys2-installer/releases/latest/download/msys2-base-x86_64-latest.sfx.exe -OutFile %TMP%\msys2-base-x86_64-latest.sfx.exe.part"
        move /Y %TMP%\msys2-base-x86_64-latest.sfx.exe.part %TMP%\msys2-base-x86_64-latest.sfx.exe
    )

    rem --- Run the installer
    echo Installing MSYS2...
    %TMP%\msys2-base-x86_64-latest.sfx.exe -y -o%MSYS2_DIR%
    rem Move files out of the `msys64` subdirectory.
    for %%a in ("%MSYS2_DIR%\msys64\*") do move /y "%%~fa" %MSYS2_DIR%
    for /d %%a in ("%MSYS2_DIR%\msys64\*") do move /y "%%~fa" %MSYS2_DIR%
    rmdir msys64
)

rem ------ Update MSYS2
rem Note that we're running the update twice, because MSYS2 can close itself during the initial update,
rem   and requires re-running the same command to finish the update.
rem It's not entirely optimal to run the command twice even if the first one finishes successfully, but I'm not sure how to check that it failed.
call %MSYS2_DIR%\msys2_shell.cmd -no-start -defterm -c "pacman -Syu --noconfirm"
call %MSYS2_DIR%\msys2_shell.cmd -no-start -defterm -c "pacman -Syu --noconfirm"

rem ------ Install some MSYS (non-mingw-w64-...) packages needed for the `msys2_install_clang_ver.sh` script called below.
call %MSYS2_DIR%\msys2_shell.cmd -no-start -defterm -clang64 -c "pacman -S --noconfirm --needed gawk"

rem ------ Install a specific version of Clang
call %MSYS2_DIR%\msys2_shell.cmd -no-start -defterm -clang64 -c "'%~dp0'/msys2_install_clang_ver.sh %CLANG_VER%"

rem ------ Install needed packages
rem `procps-ns` is for the `free` command to measure RAM.
call %MSYS2_DIR%\msys2_shell.cmd -no-start -defterm -clang64 -c "pacman -S --noconfirm --needed make procps-ng $MINGW_PACKAGE_PREFIX-cmake"

endlocal
