import os
import xml.etree.ElementTree as ET
import svgutils.transform as sg
from svgutils.compose import Unit

pendownzheight = 12.7
offset = 4


penupzheight = pendownzheight + offset


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

def generate_gcode(paths):
    gcode = []
    gcode.append("G21 ; Set units to millimeters")
    gcode.append("G90 ; Use absolute positioning")
    gcode.append("G28 ; Home all axes")
    gcode.append(f"G0 Z{penupzheight} F3000 ; Lift pen")

    for path in paths:
        if not path:
            continue

        # Move to the start of the path
        start_x, start_y = path[0]

        gcode.append(f"G0 X{start_x} Y{start_y} F3000 ; Move to start of path")
        gcode.append(f"G0 Z{pendownzheight} F3000 ; Lower pen to start drawing")

        # Draw the path
        for x, y in path[1:]:
            gcode.append(f"G0 X{x} Y{y} F3000")
        
        # Lift the pen after drawing the path
        gcode.append('G91')
        gcode.append(f'G0 Z{offset} F3000 ; Lift pen')
        gcode.append('G90')
        
    # Ensure the pen is lifted before moving to the home position
    gcode.append(f"G0 Z{penupzheight} F3000 ; Lift pen")
    gcode.append("G0 X0 Y0 F3000 ; Move to home position")
    gcode.append("M84 ; Disable motors")

    return "\n".join(gcode)

def scale_paths(paths):
    scaled_paths = []
    scaling_factor = 150/256
    nudgexy = 42.5
    for path in paths:
        scaled_path = [(x *scaling_factor + nudgexy, y * scaling_factor + nudgexy) for x, y in path]
        scaled_paths.append(scaled_path)
    
    return scaled_paths

def svg_to_gcode(svg_path, output_path):
    paths = parse_svg(svg_path)
    scaled_paths = scale_paths(paths)
    gcode = generate_gcode(scaled_paths)
    
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

# Example usage
if __name__ == '__main__':
    combine_svgs('outlines_svg', 'combined_output.svg')
    svg_to_gcode('combined_output.svg','output51.gcode')
