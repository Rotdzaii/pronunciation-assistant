@echo off
rem Typo-safe wrapper — delegates to the correctly named run_server.bat.
rem Users who type "run_sever" (missing 'r') are redirected here.
call "%~dp0run_server.bat" %*
