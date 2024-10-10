import os
import xml.etree.ElementTree as ET
import svgutils.transform as sg
from svgutils.compose import Unit

def parse_svg(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    paths = []
    
    for polyline in root.findall('.//{http://www.w3.org/2000/svg}polyline'):
        points = polyline.get('points').strip().split()
        points = [tuple(map(float, point.split(','))) for point in points]
        paths.append(points)
    
    return paths

def generate_gcode(paths,pendownzheight,offset,speed):
    speed_gcode = f"F{speed}"
    penupzheight = pendownzheight + offset
    gcode = []
    gcode.append("G21 ; Set units to millimeters")
    gcode.append("G90 ; Use absolute positioning")
    gcode.append("G28 ; Home all axes")
    gcode.append(f"G0 Z{penupzheight} {speed_gcode} ; Lift pen")

    for path in paths:
        if not path:
            continue

        # Move to the start of the path
        start_x, start_y = path[0]

        gcode.append(f"G0 X{start_x} Y{start_y} {speed_gcode} ; Move to start of path")
        gcode.append(f"G0 Z{pendownzheight} {speed_gcode} ; Lower pen to start drawing")

        # Draw the path
        for x, y in path[1:]:
            gcode.append(f"G0 X{x} Y{y} {speed_gcode}")
        
        # Lift the pen after drawing the path
        gcode.append('G91')
        gcode.append(f'G0 Z{offset} {speed_gcode} ; Lift pen')
        gcode.append('G90')
        
    # Ensure the pen is lifted before moving to the home position
    gcode.append(f"G0 Z{penupzheight} {speed_gcode} ; Lift pen")
    gcode.append("G0 X0 Y0 {speed_gcode} ; Move to home position")
    gcode.append("M84 ; Disable motors")

    return "\n".join(gcode)

def scale_paths(paths):
    scaled_paths = []
    for path in paths:
        scaled_path = [(x *scaling_factor + nudge_xy, y * scaling_factor + nudge_xy) for x, y in path]
        scaled_paths.append(scaled_path)
    
    return scaled_paths

def svg_to_gcode(svg_path, output_path,pendownzheight,offset,speed):
    paths = parse_svg(svg_path)
    scaled_paths = scale_paths(paths)
    gcode = generate_gcode(scaled_paths,pendownzheight,offset,speed)
    
    with open(output_path, 'w') as f:
        f.write(gcode)

def combine_svgs(input_dir, output_svg_path):
    combined_svg = ET.Element('svg', xmlns="http://www.w3.org/2000/svg", version="1.1")
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.svg'):
            file_path = os.path.join(input_dir, file_name)
            tree = ET.parse(file_path)
            root = tree.getroot()
            ns = {'svg': 'http://www.w3.org/2000/svg'}
            
            for polyline in root.findall('.//{http://www.w3.org/2000/svg}polyline'):
                combined_svg.append(polyline)
    
    tree = ET.ElementTree(combined_svg)
    tree.write(output_svg_path)

def run_all(pendownzheight,offset,scalingfactor,nudgexy,speed):
    global scaling_factor
    global nudge_xy
    scaling_factor = scalingfactor
    nudge_xy = nudgexy

    combine_svgs('outlines_svg', 'combined_output.svg')
    svg_to_gcode('combined_output.svg','output.gcode',pendownzheight,offset,speed)

if __name__ == '__main__':
    run_all(pendownzheight = 14.5 , offset = 4, scalingfactor = .586, nudgexy = 42.5,speed = 3000)
    