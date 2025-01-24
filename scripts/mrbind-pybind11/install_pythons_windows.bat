@echo off

rem Allow `!var!` variables, which is the only way you can set variables in loops.
setlocal enabledelayedexpansion
rem Allow `%RANDOM%`.
setlocal enableextensions

rem Download the page with the list of available versions. We need this to get the latest minor version number.
echo Downloading the list of known versions...
set "tempfile=%tmp%\meshlib_pythons_downloader_%RANDOM%.tmp"
del %tempfile% 2>nul
powershell -Command "Invoke-WebRequest https://www.python.org/ftp/python -OutFile %tempfile%"
echo Finished downloading to: %tempfile%

rem Extract the version numbers from the page and version-sort them. Write the result to `%tempfile%2`.
powershell (type %tempfile%) -match 'href=""""3\.' -replace '.*^>(.*)/^<.*','$1' "|" Sort-Object { $_ -as [version] } -Descending >%tempfile%2

for /f %%x in (%~dp0\python_versions.txt) do (
    echo.
    echo -----------------------
    echo.

    rem Here %%x is the version number, such as 3.13.
    rem Set `!ver_terse!` to the version number without the dot, such as `313`.
    set ver_terse=%%x
    set ver_terse=!ver_terse:.=!
    if exist %localappdata%\Programs\Python\Python!ver_terse!\python.exe (
        echo Python %%x - already installed
    ) else (
        echo Python %%x - installing...

        rem Guess the minor version number.
        set done=0
        call :install_ver %%x !ver_terse!
        if !done! == 0 (
            echo Couldn't find a suitable version to install.
            exit /b 1
        )
    )
    echo Installing Pip and dependencies for Python %%x
    py -%%x -m pip install --upgrade pip
    py -%%x -m pip install --upgrade -r %~dp0\..\..\requirements\python.txt
    py -%%x -m pip install pytest
)

echo Done

endlocal
endlocal

goto :eof


rem --- Now some functions:

:install_ver
for /f %%y in ('findstr %1 %tempfile%2') do (
    if !done! == 0 (
        echo Trying version: %%y
        set installer=%tmp%\python-%%y-amd64.exe
        del /S /Q "!installer!" >nul 2>nul
        powershell -Command "Invoke-WebRequest https://www.python.org/ftp/python/%%y/python-%%y-amd64.exe -OutFile !installer!" >nul
        if exist "!installer!" (
            echo Download successful.

            echo Deleting existing version at: %localappdata%\Programs\Python\Python%2
            rmdir /S /Q %localappdata%\Programs\Python\Python%2

            echo Installing... ^(click "yes" if prompted^)
            "!installer!" /quiet
            set done=1
        ) else (
            echo No Windows installer for version %%y
        )
    )
)

goto :eof