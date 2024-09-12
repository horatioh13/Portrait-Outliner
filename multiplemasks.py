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
from svgwrite.container import Group



############################################
# Configuration

remove_model = 'rembg'
#remove_model = 'removebg'

apikey = 'dyqhCX5Zx5vRi9r1Hw3uVrky'

#imagesource = 'laptopwebcam'
imagesource = 'usbwebcam'
#imagesource = 'thispersondoesnotexist'
#imagesource = 'file'

#edgemode = 'canny'
edgemode = 'multiplemasks'

############################################

##### OLD FUNCTIONS #####

    

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
            image_path = file_path + '/image1.png'
            name = file_path[6:-5]
            print(name)
            new_path = 'outlines_png/' + name + '_outline.png'
            get_edges_from_mask(image_path, new_path)



#remove_background('original_image/image1.png','original_image/image1.png')
#get_edges()
#detect_and_segment_glasses('cropped_image/image1.png','glasses.png')
#segment_face()
#create_outlines()

#############################################################################
                              ##### NEW FUNCTIONS #####
#############################################################################



#### helper functions ####
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

def NUMPY_convert_to_SVG(image, output_path):
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an SVG drawing
    height, width = image.shape
    dwg = Drawing(output_path, profile='tiny', size=(width, height))
    
    # Draw contours
    for contour in contours:
        points = [(int(point[0][0]), int(point[0][1])) for point in contour]
        dwg.add(dwg.polygon(points, fill='none', stroke='black', stroke_width=1))
    
    # Save the SVG file
    dwg.save()    

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



##### MAIN FUNCTIONS #####
#############################################################################

def cleardata():
    dir_path = 'original_image'
    for file in os.listdir(dir_path):
        if imagesource == 'file':
            continue
        
        file_path = os.path.join(dir_path, file)
        os.remove(file_path)

    dir_path = 'cropped_image'
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        os.remove(file_path)
    
    dir_path = 'masks'
    # Check if the directory exists
    if os.path.exists(dir_path):
        # Remove everything in the directory
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    dir_path = 'outlines_svg'
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        os.remove(file_path)

    dir_path = 'outlines_png'
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        os.remove(file_path)

def get_image_from_source():
    if imagesource == 'thispersondoesnotexist':
        image_url = 'https://thispersondoesnotexist.com'
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            image_bytes = io.BytesIO(response.content)
            image1 = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite('original_image/image1.png', image1)
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
        cap = cv2.VideoCapture(2)
        ret, frame = cap.read()
        cap.release()
        if ret:
            cv2.imwrite('original_image/image1.png', frame)
        else:
            print('Failed to capture image from webcam.')
    
    elif imagesource == 'file':
        image1 = 'original_image/image1.png'

def align_and_crop():
    cropper = Cropper(strategy="largest")
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

    #subtracting glasses from eybrows
    groups['eybrows'] = dilate_and_subtract(groups['eybrows'], groups['glasses'])
    
    #subtracting glasses from nose
    groups['nose'] = dilate_and_subtract(groups['nose'], groups['glasses'])

    #subtracting upper and lower lip from mouth
    groups['mouth'] = dilate_and_subtract(groups['mouth'], groups['upper lip'])
    groups['mouth'] = dilate_and_subtract(groups['mouth'], groups['lower lip'])

    return groups

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




if __name__ == '__main__':
    cleardata()
    get_image_from_source()
    align_and_crop()
    
    groups = mask_dir_to_NUMPY_arrays() 
    groups2 = process_NUMPY_outlines(groups)
    combine_and_plot_masks(groups2)

    #NUMPY_convert_to_SVG(groups2['eybrows'], 'neck.svg')

    


