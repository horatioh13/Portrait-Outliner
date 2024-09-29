import multiplemasks
import svgs_to_gcode
import os
import subprocess

def run_all(image_source):
    multiplemasks.run_all(image_source)
    svgs_to_gcode.run_all(pendownzheight=17,offset=4)


if __name__ == '__main__':
    multiplemasks.run_all('thispersondoesnotexist')
    svgs_to_gcode.run_all(pendownzheight=17,offset=4)

    command = "python3 pyGcodeSender.py output51.gcode"

    subprocess.run(command, shell=True)


