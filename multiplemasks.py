import os
import shutil
import cv2
import requests
import io
import numpy as np
from rembg import new_session, remove
from glasses_detector import GlassesClassifier,GlassesSegmenter
from face_crop_plus import Cropper
from svgwrite import Drawing
from skimage.morphology import skeletonize
from svgpathtools import svg2paths, wsvg, Path, Line
import xml.etree.ElementTree as ET
from math import sqrt
from rdp import rdp
import math

from svgs_to_gcode import parse_svg

############################################
# Configuration

remove_model = 'rembg'
#remove_model = 'removebg'

apikey = 'dyqhCX5Zx5vRi9r1Hw3uVrky'

imagesource = 'laptopwebcam'
#imagesource = 'usbwebcam'
#imagesource = 'thispersondoesnotexist'
#imagesource = 'file'

#edgemode = 'canny'
edgemode = 'multiplemasks'

############################################

#old/obsolete functions

def remove_background(input_path,output_path):
    if remove_model == 'rembg':
        #model_name = 'birefnet-portrait'
        model_name = 'u2net'
        #model_name = 'u2net_cloth_seg'
        session = new_session(model_name)
        input = cv2.imread(input_path)
        output = remove(input, session=session, bgcolor=(255, 255, 255, 255),post_process_mask=True)
        cv2.imwrite(output_path, output)

    elif remove_model == 'removebg':
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': open(input_path, 'rb')},
            data={'size': 'auto', 'type': 'person','bg_color': 'white'},
            headers={'X-Api-Key': 'dyqhCX5Zx5vRi9r1Hw3uVrky'},
        )
        if response.status_code == requests.codes.ok:
            with open(output_path, 'wb') as out:
                out.write(response.content)
        else:
            print("Error:", response.status_code, response.text)
    
def get_edges_from_colored_image(input_path,output_path):
    # Read the input image
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarize the image: set every non-white pixel to black
    _, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty image to draw contours
    contour_image = np.zeros_like(image)
    
    # Draw contours
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)
    
    # Save the resulting image
    cv2.imwrite(output_path, contour_image)

def get_edges_from_mask(input_path,output_path):
    # Read the input image
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)    
    
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty image to draw contours
    contour_image = np.zeros_like(image)
    
    # Draw contours
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)
    
    # Save the resulting image
    cv2.imwrite(output_path, contour_image)

def detect_and_segment_glasses(inputpath,outputpath):
    clf = GlassesClassifier()
    ans = clf([inputpath], format="bool")[0]
    if ans == True:
        print('Glasses detected')
        seg = GlassesSegmenter(kind='lenses',size='m')
        seg.process_file(inputpath,outputpath, format='mask')

    elif ans == False:
        print('No glasses detected')

def create_outlines():
    dir_path = 'masks'
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if file_path.endswith('_mask'):
            image_path = os.path.join(file_path, 'image1.png')
            name = file_path[6:-5]
            print(name)
            new_path = os.path.join('outlines_png', name + '_outline.png')
            get_edges_from_mask(image_path, new_path)


#### functions for working in RAM ####

def NUMPY_get_edges_from_mask(image):    
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty image to draw contours
    contour_image = np.zeros_like(image)
    
    # Draw contours
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)
    
    # Return the resulting image
    return contour_image     

def mask_dir_to_svgs():
    dir_path = 'masks'
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if file_path.endswith('_mask'):
            image_path = os.path.join(file_path, 'image1.png')
            name = file_path[6:-5]
            mask = read_PNG_mask(image_path)
            #mask = NUMPY_process_bitmaps(mask)
            svg_path = os.path.join('outlines_svg', name + '_outline.svg')
            NUMPY_convert_to_SVG(mask, svg_path)

def combine_and_plot_masks(masks_dict):
    combined_mask = None
    masks = masks_dict.values()
    for mask in masks:
        if mask is not None:        
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = cv2.bitwise_or(combined_mask, mask)

    if combined_mask is not None:
        # Display the combined mask using OpenCV
        cv2.imshow('Combined Mask', combined_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No masks provided.")

def dilate_and_subtract(target, source):
    if (source is not None) and (target is not None):
        # Create a copy of the target to modify
        result = target.copy()

        # Dilate the source image by 1 pixel
        kernel = np.ones((3, 3), np.uint8)
        dilated_source = cv2.dilate(source, kernel, iterations=1)

        # Subtract the dilated source from the target
        result = cv2.subtract(result, dilated_source)

        return result
    else:
        return target

#secondary functions

def read_PNG_mask(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def NUMPY_get_edges_from_mask2(image):
    # Create a copy of the image to modify
    result_image = image.copy()

    # Define the gray value
    gray_value = 127

    # Get the dimensions of the image
    rows, cols = image.shape

    # Iterate through each pixel in the image
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if image[i, j] == 255:  # Check if the pixel is white
                # Get the 8-connected neighbors
                neighbors = image[i-1:i+2, j-1:j+2].flatten()
                # Check if all neighbors are white or gray
                if all(neighbor in [255, gray_value] for neighbor in neighbors):
                    result_image[i, j] = gray_value

    result_image[result_image == 127] = 0 

    return result_image


###### MAIN FUNCTIONS ######
def cleardata():
    # Directories to clear
    directories = ['original_image', 'cropped_image', 'masks', 'outlines_svg']

    if imagesource == 'file':
        directories.remove('original_image')

    for dir_path in directories:
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                if file == '.gitignore':
                    continue
                
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

    # Remove .gcode and .svg files from the base directory
    base_dir = os.getcwd()
    for file in os.listdir(base_dir):
        if file.endswith('.gcode') or file.endswith('.svg'):
            file_path = os.path.join(base_dir, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
    


def get_image_from_source():
    if imagesource == 'thispersondoesnotexist':
        image_url = 'https://thispersondoesnotexist.com'
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            image_bytes = io.BytesIO(response.content)
            image1 = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join('original_image', 'image1.png'), image1)
        except requests.RequestException as e:
            print(f'Failed to download the image: {e}')

    elif imagesource == 'laptopwebcam':
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            cv2.imwrite('original_image/image1.png', frame)
        else:
            print('Failed to capture image from webcam.')

    elif imagesource == 'usbwebcam':
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            cv2.imwrite('original_image/image1.png', frame)
        else:
            print('Failed to capture image from webcam.')
    
    elif imagesource == 'file':
        image1 = 'original_image/image1.png'

def align_and_crop():
    cropper = Cropper(strategy="largest",face_factor=.7)
    cropper.process_dir(input_dir='original_image',output_dir='cropped_image')

def segment_face():
    cropper = Cropper(
    mask_groups={"skin":[1],"eybrows":[2,3],"eyes":[4,5],"glasses":[6],"ears":[7,8],"earings":[9],"nose":[10],"mouth":[11],"upper lip":[12],"lower lip":[13],"neck":[14],"necklace":[15],"clothes":[16],"hair":[17],"hat":[18]},
    )
    cropper.process_dir(input_dir="cropped_image",output_dir="masks")

def mask_dir_to_NUMPY_arrays():
    dir_path = 'masks'

    mask_groups = {
        "skin": [], "eybrows": [], "eyes": [], "glasses": [], "ears": [], "earings": [],
        "nose": [], "mouth": [], "upper lip": [], "lower lip": [], "neck": [], "necklace": [],
        "clothes": [], "hair": [], "hat": []
    }
    
    keys = mask_groups.keys()
    for key in keys:
        png_path_mask = os.path.join(dir_path + '/' + key + '_mask/image1.png')
        if os.path.isfile(png_path_mask):
            current_mask = read_PNG_mask(png_path_mask)
            current_outline = NUMPY_get_edges_from_mask2(current_mask)
            mask_groups[key] = current_outline
        else:
            mask_groups[key] = None
    
    return mask_groups

def process_NUMPY_outlines(groups):
    # Process the outlines
    ### subtracting everything from skin aside from neck, and ears
    groups['skin'] = dilate_and_subtract(groups['skin'], groups['eybrows'])
    groups['skin'] = dilate_and_subtract(groups['skin'], groups['eyes'])
    groups['skin'] = dilate_and_subtract(groups['skin'], groups['glasses'])
    groups['skin'] = dilate_and_subtract(groups['skin'], groups['earings'])
    groups['skin'] = dilate_and_subtract(groups['skin'], groups['nose'])
    groups['skin'] = dilate_and_subtract(groups['skin'], groups['mouth'])
    groups['skin'] = dilate_and_subtract(groups['skin'], groups['upper lip'])
    groups['skin'] = dilate_and_subtract(groups['skin'], groups['lower lip'])
    groups['skin'] = dilate_and_subtract(groups['skin'], groups['clothes'])
    groups['skin'] = dilate_and_subtract(groups['skin'], groups['hair'])
    groups['skin'] = dilate_and_subtract(groups['skin'], groups['hat'])
    groups['skin'] = dilate_and_subtract(groups['skin'], groups['necklace'])

    #subtracting necklace, clothes, hair, skin from neck
    groups['neck'] = dilate_and_subtract(groups['neck'], groups['necklace'])
    groups['neck'] = dilate_and_subtract(groups['neck'], groups['clothes'])
    groups['neck'] = dilate_and_subtract(groups['neck'], groups['hair'])
    groups['neck'] = dilate_and_subtract(groups['neck'], groups['skin'])

    #subtracting hat and glasses from hair
    groups['hair'] = dilate_and_subtract(groups['hair'], groups['hat'])
    groups['hair'] = dilate_and_subtract(groups['hair'], groups['glasses'])

    #subtracting hair and skin from ears
    groups['ears'] = dilate_and_subtract(groups['ears'], groups['hair'])
    groups['ears'] = dilate_and_subtract(groups['ears'], groups['skin'])

    #subtracting glasses and hat from eybrows
    groups['eybrows'] = dilate_and_subtract(groups['eybrows'], groups['glasses'])
    groups['eybrows'] = dilate_and_subtract(groups['eybrows'], groups['hat'])
    
    #subtracting glasses from nose
    groups['nose'] = dilate_and_subtract(groups['nose'], groups['glasses'])

    #subtracting upper and lower lip from mouth
    groups['mouth'] = dilate_and_subtract(groups['mouth'], groups['upper lip'])
    groups['mouth'] = dilate_and_subtract(groups['mouth'], groups['lower lip'])

    #subtracting ears and hair and earings from clothes
    groups['clothes'] = dilate_and_subtract(groups['clothes'], groups['ears'])
    groups['clothes'] = dilate_and_subtract(groups['clothes'], groups['hair'])
    groups['clothes'] = dilate_and_subtract(groups['clothes'], groups['earings'])

    #subtracting earings from ears
    groups['ears'] = dilate_and_subtract(groups['ears'], groups['earings'])





    return groups

def distance(p1, p2):
    x1, y1 = map(float, p1.split(','))
    x2, y2 = map(float, p2.split(','))
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def NUMPY_convert_to_SVG(image, output_path):
    # Skeletonize the binary image to get the centerline
    skeleton = skeletonize(image // 255)  # Convert to binary (0, 1) for skeletonize
    
    # Find contours on the skeletonized image
    contours, _ = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an SVG drawing
    height, width = image.shape
    dwg = Drawing(output_path, profile='tiny', size=(width, height))
    
    # Draw contours as polylines
    for contour in contours:
        points = []
        for point in contour:
            x = int(point[0][0])
            y = int(point[0][1])
            points.append((x, y))
        dwg.add(dwg.polyline(points, fill='none', stroke='black', stroke_width=1))
    
    # Convert the SVG drawing to an XML string
    svg_string = dwg.tostring()
    
    # Parse the SVG string
    root = ET.fromstring(svg_string)    

    def scale_point(point, scale_factor):
        x, y = map(float, point.split(','))
        x = int(x * scale_factor)
        y = int(y * scale_factor)
        return f'{x},{y}'
    
    def find_longest_line_segment(points):
        longest_segment_length = 0
        longest_segment_index = 0
        for i in range(len(points) - 1):
            segment_length = distance(points[i], points[i + 1])
            if segment_length > longest_segment_length:
                longest_segment_length = segment_length
                longest_segment_index = i
        return longest_segment_index, longest_segment_length
    
    def reorder_paths(points, longest_segment_index, longest_segment_length):
        if longest_segment_index != 0:
            points_before = points[:longest_segment_index + 1]
            points_after = points[longest_segment_index + 1:]
    
            reversed_before = points_before[::-1] + points_after
            reversed_after = points_before + points_after[::-1]
    
            new_segment_length_before = distance(reversed_before[longest_segment_index], reversed_before[longest_segment_index + 1])
            new_segment_length_after = distance(reversed_after[longest_segment_index], reversed_after[longest_segment_index + 1])
    
            if new_segment_length_before < new_segment_length_after and new_segment_length_before < longest_segment_length:
                return reversed_before
            if new_segment_length_after < new_segment_length_before and new_segment_length_after < longest_segment_length:
                return reversed_after
        return points
    
    # Iterate through each polyline in the SVG
    for polyline in root.findall('.//{http://www.w3.org/2000/svg}polyline'):
        points = polyline.get('points').strip().split()
        unique_points = []
        visited_points = set()
        
        for point in points:
            point = scale_point(point, 150 / 256)
            if point not in visited_points:
                visited_points.add(point)
                unique_points.append(point)
        
        longest_segment_index, longest_segment_length = find_longest_line_segment(unique_points)
        unique_points = reorder_paths(unique_points, longest_segment_index, longest_segment_length)
    
        if unique_points:
            polyline.set('points', ' '.join(unique_points))
    
    # Convert the cleaned XML tree back to a string
    cleaned_svg_string = ET.tostring(root, encoding='unicode')

    # Save the cleaned SVG string to the output file
    with open(output_path, 'w') as f:
        f.write(cleaned_svg_string)

def closed_contor_calculation(contour):
    contor_status = False
    area = cv2.contourArea(contour)
    if area > 0:
        contor_status = True

    return contor_status

def NUMPY_convert_to_SVG2(image, output_path):
    # Skeletonize the binary image to get the centerline
    skeleton = skeletonize(image // 255)  # Convert to binary (0, 1) for skeletonize
    
    # Convert skeleton to 8-bit single-channel image for display
    skeleton_display = (skeleton * 255).astype(np.uint8)

    # Find contours on the skeletonized image
    contours, _ = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an SVG drawing
    height, width = image.shape
    dwg = Drawing(output_path, profile='tiny', size=(width, height))
    
    # Draw contours as polylines
    for contour in contours:
        points = []
        visited_points = set()
        for point in contour:
            x = int(point[0][0])
            y = int(point[0][1])
            current_point = (x, y)
            
            #if current_point not in visited_points:
            points.append(current_point)

            visited_points.add(current_point)
        
        if closed_contor_calculation(contour) == True:
            points.append(points[0])
        print(points)  
        dwg.add(dwg.polyline(points, fill='none', stroke='black', stroke_width=1))
    
    # Convert the SVG drawing to an XML string
    svg_string = dwg.tostring()
    
    # Parse the SVG string
    root = ET.fromstring(svg_string)    

    # Convert the cleaned XML tree back to a string
    cleaned_svg_string = ET.tostring(root, encoding='unicode')

    # Save the cleaned SVG string to the output file
    with open(output_path, 'w') as f:
        f.write(cleaned_svg_string)

def unest_list(points):
    visited_points = set()
    result = []
    current_list = []

    for point in points:
        if point in visited_points:
            if current_list:
                result.append(current_list)
            current_list = [point]
            result.append(current_list)
            current_list = []
        else:
            visited_points.add(point)
            current_list.append(point)

    if current_list:
        result.append(current_list)

    return result

def loop_calculation(contour):
    points = [(int(point[0][0]), int(point[0][1])) for point in contour]
    return unest_list(points)

def add_features_from_svg():
    if os.path.isfile(os.path.join('outlines_svg', 'eyes_outline.svg')):
    
        eyes = []
        paths = parse_svg(svg_path=os.path.join('outlines_svg', 'eyes_outline.svg'))
        #print(paths)

        for path in paths:
            # Convert the list of coordinates to a numpy array
            contour = np.array(path, dtype=np.int32)

            # Calculate the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate the center of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2

            # Append the center, height, and width to the list
            eyes.append([(center_x, center_y), h, w])

        # Create a new SVG drawing
        output_dir = os.path.join('outlines_svg', 'eyeballs.svg')
        dwg = Drawing(output_dir, profile='tiny')

        def approximate_circle(center, radius, num_points=36):
            cx, cy = center
            points = [
                (cx + radius * math.cos(2 * math.pi * i / num_points),
                 cy + radius * math.sin(2 * math.pi * i / num_points))
                for i in range(num_points)
            ]
            points.append(points[0])  # Close the circle
            return points

        average_iris_radius = sum(eye[1] for eye in eyes) // len(eyes) // 2
        pupil_radius = average_iris_radius // 3
        # Add the pupils and iris to the new SVG
        for eye in eyes:
            center = eye[0]

            # Draw the iris as a polyline
            iris_points = approximate_circle(center, average_iris_radius)
            dwg.add(dwg.polyline(points=iris_points, fill='none', stroke='black', stroke_width=1))

            # Draw the pupil as a polyline
            pupil_points = approximate_circle(center, pupil_radius)
            dwg.add(dwg.polyline(points=pupil_points, fill='none', stroke='black', stroke_width=1))

        # Save the new SVG file
        dwg.save()
    
def init_file_structure():
    directories = ['original_image', 'cropped_image', 'masks', 'outlines_svg']
    for dir_path in directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def set_list_value(lst, index, value):
    if index >= len(lst):
        lst.extend([None] * (index + 1 - len(lst)))
    lst[index] = value

def process_points1(points):
    final_list = []
    points_buffer_list = []
    points_target = set(points)
    mode = 'new'
    buffer_index = 0
    traversed_points = set() 
    if len(points) == len(points_target):
        return [points]
    else:
        for point in points:
            if point not in traversed_points:
                traversed_points.add(point)
                if mode == 'new':
                    points_buffer_list.append(point)
                elif mode == 'dupe':
                    #set_list_value(final_list, buffer_index, points_buffer_list)
                    points_buffer_list = []
                    #buffer_index += 1
                    mode = 'new'
            elif point in traversed_points:
                if mode == 'dupe':
                    pass
                elif mode == 'new':
                    set_list_value(final_list, buffer_index, points_buffer_list)
                    points_buffer_list = []
                    buffer_index += 1
                    mode = 'dupe'
        if points_buffer_list:
            set_list_value(final_list, buffer_index, points_buffer_list)
    return final_list

def process_points2(points):
    final_list = []
    points_buffer_list = []
    points_target = set(points)
    mode = 'new'
    buffer_index = 0
    traversed_points = set() 
    if len(points) == len(points_target):
        return [points]
    else:
        for point in points:
            if mode == 'new':
                if point not in traversed_points:
                    points_buffer_list.append(point)
                    traversed_points.add(point)
                elif point in traversed_points:
                    points_buffer_list.append(point)
                    set_list_value(final_list, buffer_index, points_buffer_list)
                    points_buffer_list = []
                    buffer_index += 1
                    mode = 'dupe'
 
            elif mode == 'dupe':
                if point not in traversed_points:
                    points_buffer_list = []
                    points_buffer_list.append(points[points.index(point) - 1])
                    points_buffer_list.append(point)
                    buffer_index += 1
                    mode = 'new'
                elif point in traversed_points:
                    pass

        if points_buffer_list:
            set_list_value(final_list, buffer_index, points_buffer_list)
    return final_list
            
def NUMPY_convert_to_SVG3(image, output_path):
    # Skeletonize the binary image to get the centerline
    skeleton = skeletonize(image // 255)  # Convert to binary (0, 1) for skeletonize
    
    # Convert skeleton to 8-bit single-channel image for display
    skeleton_display = (skeleton * 255).astype(np.uint8)

    # Find contours on the skeletonized image
    contours, _ = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an SVG drawing
    height, width = image.shape
    dwg = Drawing(output_path, profile='tiny', size=(width, height))
    
    # Draw contours as polylines
    for contour in contours:
        points = []
        visited_points = set()
        for point in contour:
            x = int(point[0][0])
            y = int(point[0][1])
            current_point = (x, y)
            
            #if current_point not in visited_points:
            points.append(current_point)

            visited_points.add(current_point)
        
        if closed_contor_calculation(contour) == True:
            points.append(points[0])

        list_of_list_of_points = process_points2(points)
        for list_of_points in list_of_list_of_points:
            if list_of_points:
                dwg.add(dwg.polyline(list_of_points, fill='none', stroke='black', stroke_width=1))
    
    # Convert the SVG drawing to an XML string
    svg_string = dwg.tostring()
    
    # Parse the SVG string
    root = ET.fromstring(svg_string)    

    # Convert the cleaned XML tree back to a string
    cleaned_svg_string = ET.tostring(root, encoding='unicode')

    # Save the cleaned SVG string to the output file
    with open(output_path, 'w') as f:
        f.write(cleaned_svg_string)

def run_all():
    init_file_structure()

    cleardata()
    get_image_from_source()
    align_and_crop()

    segment_face()
    
    groups = mask_dir_to_NUMPY_arrays() 
    groups2 = process_NUMPY_outlines(groups)
    combine_and_plot_masks(groups2)
    items = groups2.items()
    keys = groups2.keys()

    

    for item, key in zip(items, keys):
        if item[1] is not None:
            output_path = os.path.join('outlines_svg', f'{key}_outline.svg')
            NUMPY_convert_to_SVG3(item[1], output_path)
    
    add_features_from_svg()

if __name__ == '__main__':
    run_all()




    


