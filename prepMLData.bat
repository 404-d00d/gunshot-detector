@echo off
echo [1/4] Running augmentAudio.py...
py D:\augmentAudio.py

echo [2/4] Running generateNewCSV.py...
py D:\generateNewCSV.py

echo [3/4] Running cleanData.py...
py D:\cleanData.py

echo [4/4] Running bodycamData.py...
py D:\bodycamData.py

echo All scripts completed successfully.
pause
