import multiplemasks
import svgs_to_gcode
import os
import subprocess
import argparse

def run_all(image_source, plot_bitmaps, pendownzheight, offset, scalingfactor, nudgexy, gcode_file):
    multiplemasks.run_all(image_source=image_source, plot_bitmaps=plot_bitmaps)
    svgs_to_gcode.run_all(pendownzheight=pendownzheight, offset=offset, scalingfactor=scalingfactor, nudgexy=nudgexy)

    if gcode_file is not None:
        command = f"python3 pyGcodeSender.py {gcode_file}"
        subprocess.run(command, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the drawing robot script with specified parameters.')
    parser.add_argument('--image_source', type=str, default='thispersondoesnotexist', help='Source of the image')
    parser.add_argument('--plot_bitmaps', type=bool, default=False, help='Whether to plot bitmaps')
    parser.add_argument('--pendownzheight', type=float, default=14.5, help='Pen down Z height')
    parser.add_argument('--offset', type=float, default=4, help='Offset value')
    parser.add_argument('--scalingfactor', type=float, default=0.586, help='Scaling factor')
    parser.add_argument('--nudgexy', type=float, default=42.5, help='Nudge XY value')
    parser.add_argument('--gcode_file', type=str, default=None, help='G-code file to send')
    args = parser.parse_args()

    run_all(args.image_source, args.plot_bitmaps, args.pendownzheight, args.offset, args.scalingfactor, args.nudgexy, args.gcode_file)


