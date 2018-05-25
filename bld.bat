"%PYTHON%" setup.py install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1
echo trying to install opencv-python
pip install opencv-python
echo ok tried
if errorlevel 1 exit 1
