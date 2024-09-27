import multiplemasks
import svgs_to_gcode
import os
import subprocess

multiplemasks.run_all()
svgs_to_gcode.run_all(pendownzheight=9.8,offset=4)

command = "python3 PyGcodeSender/pyGcodeSender.py output51.gcode"

subprocess.run(command, shell=True)