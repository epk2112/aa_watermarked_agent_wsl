###  GRAPH PREP


import os
import json

# Define the home directory and BW Agent directory
home_dir = '/home/jupyter_user/documents'
mockups_agent_dir = os.path.join(home_dir, 'doc_content_gen/mockups_agent_watermarked')

# Define the final Exports directory and create it if it doesn't exist
final_exports_dir = os.path.join(home_dir, 'final_exports')
os.makedirs(final_exports_dir, exist_ok=True)

os.chdir(mockups_agent_dir)

# Create new Directory for outputs if it doesnt exit
os.makedirs('outputs', exist_ok=True)

# Load variables from JSON file
with open('nested_variables.json', 'r') as json_file:
    nested_variables = json.load(json_file)



from typing import TypedDict, Annotated
import operator

# Define the State of the Graph
class ParentState(TypedDict):
    nested_variables: Annotated[dict, 'Nested Variables']
    office_set_v1_image_name: Annotated[str, operator.add]
    office_set_v2_image_name: Annotated[str, operator.add]
    office_set_v3_image_name: Annotated[str, operator.add]
    office_set_v4_image_name: Annotated[str, operator.add]
    office_set_v5_image_name: Annotated[str, operator.add]
    office_set_v6_image_name: Annotated[str, operator.add]
    office_set_v7_image_name: Annotated[str, operator.add]
    office_set_v8_image_name: Annotated[str, operator.add]
    office_set_v9_image_name: Annotated[str, operator.add]
    office_set_v10_image_name: Annotated[str, operator.add]
    office_set_v11_image_name: Annotated[str, operator.add]
    office_set_v12_image_name: Annotated[str, operator.add]
    office_set_v13_image_name: Annotated[str, operator.add]
    office_set_v14_image_name: Annotated[str, operator.add]
    office_set_v15_image_name: Annotated[str, operator.add]
    office_set_v16_image_name: Annotated[str, operator.add]
    office_set_v17_image_name: Annotated[str, operator.add]
    office_set_v18_image_name: Annotated[str, operator.add]



# Define Function to Prepare State
def prepare_state(state):
    print('-> Calling Prepare State Function ->')


# Define function For Bilinear Wrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def bilinear_warp(x1, y1, x2, y2, x3, y3, x4, y4, path_to_image):
    # Read the image
    img = cv2.imread(path_to_image, cv2.IMREAD_UNCHANGED)

    # Get image dimensions
    rows, cols = img.shape[:2]

    # Define source points (corners of the input image)
    src_points = np.float32([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]])

    # Define destination points (where corners should be in output image)
    dst_points = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    # Calculate the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Calculate the dimensions of the output image
    min_x = min(x1, x2, x3, x4)
    max_x = max(x1, x2, x3, x4)
    min_y = min(y1, y2, y3, y4)
    max_y = max(y1, y2, y3, y4)

    output_width = int(max_x - min_x)
    output_height = int(max_y - min_y)

    # Apply the perspective transformation
    result = cv2.warpPerspective(img, matrix, (output_width, output_height),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))

    return result   


from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import math
import json
import io

# Define the Function to Manipulate Selected template image
def manipulate_selected_image(single_image_obj):
    # Define Single image Variables
    variables = single_image_obj

    # Load the background image
    background = Image.open(variables['background_image_path']).convert('RGBA')
    width, height = background.size

    # Change hue and saturation of the background
    def change_hue_saturation(image, hue_shift, saturation_shift):
        # Convert PIL Image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)

        # Convert string hue_shift to integer if necessary
        hue_shift_map = {
            'blue': -61,
            'green': -120,
            'purple': -28,
            'cyan': 90,
            'yellow': 30,
            'orange': 15,
            'pink': -15,
            'red': 0
        }

        if isinstance(hue_shift, str):
            hue_shift = hue_shift_map.get(hue_shift.lower(), 0)

        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Adjust hue
        hsv_image[..., 0] = (hsv_image[..., 0] + hue_shift) % 180  # Hue values range from 0 to 179 in OpenCV
        hsv_image[..., 0] = np.clip(hsv_image[..., 0], 0, 179)

        # Adjust saturation
        hsv_image[..., 1] = hsv_image[..., 1] + (hsv_image[..., 1] * (saturation_shift / 100.0))
        hsv_image[..., 1] = np.clip(hsv_image[..., 1], 0, 255)

        # Convert back to BGR color space
        result_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Convert back to PIL Image
        return Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGBA))

    # Apply hue and saturation change to the background
    hue_shift = variables.get('hue_saturation', {}).get('hue_shift', 0)
    saturation_shift = variables.get('hue_saturation', {}).get('saturation_shift', 0)
    background = change_hue_saturation(background, hue_shift, saturation_shift)

    # Create a drawing object
    draw = ImageDraw.Draw(background)

    # Function to calculate new coordinates based on reference point
    def calculate_coordinates(x, y, width, height, reference_point):
        if reference_point == 'left':
            return (x, y - height // 2)
        elif reference_point == 'right':
            return (x - width, y - height // 2)
        elif reference_point == 'top':
            return (x - width // 2, y)
        elif reference_point == 'bottom':
            return (x - width // 2, y - height)
        elif reference_point == 'top-left':
            return (x, y)
        elif reference_point == 'top-right':
            return (x - width, y)
        elif reference_point == 'bottom-left':
            return (x, y - height)
        elif reference_point == 'bottom-right':
            return (x - width, y - height)
        else:  # Default to center
            return (x - width // 2, y - height // 2)

    # Add text
    for item in variables['text_items']:
        try:
            font = ImageFont.truetype(item['font_path'], item['font_size'])
        except IOError:
            font = ImageFont.load_default()

        # Get the bounding box of the text
        left, top, right, bottom = draw.textbbox((0, 0), item['text'], font=font)
        text_width = right - left
        text_height = bottom - top

        # Calculate the new coordinates based on reference point
        x, y = item['coordinates']
        new_x, new_y = calculate_coordinates(x, y, text_width, text_height, item['reference_point'])

        # Draw the text at the new coordinates
        draw.text((new_x, new_y), item['text'], font=font, fill=item['color'])

        # Apply bilinear warp if bilinear_points are provided
        if item['bilinear_points']:
            points = [float(coord) for coord in item['bilinear_points'].split()]
            if len(points) == 8:
                x1, y1, x2, y2, x3, y3, x4, y4 = points
                warped_text = bilinear_warp(x1, y1, x2, y2, x3, y3, x4, y4, item['font_path'])
                # Convert warped_text to PIL Image and paste it onto the background
                warped_text_pil = Image.fromarray(cv2.cvtColor(warped_text, cv2.COLOR_BGRA2RGBA))
                background.paste(warped_text_pil, (int(min(x1, x2, x3, x4)), int(min(y1, y2, y3, y4))), warped_text_pil)

    # Function to apply blending mode
    def blend_images(background, overlay, blend_mode):
        if blend_mode == 'normal':
            return Image.alpha_composite(background, overlay)
        elif blend_mode == 'multiply':
            return Image.blend(background, Image.composite(overlay, background, overlay), 0.5)
        elif blend_mode == 'screen':
            return Image.blend(background, Image.composite(overlay, background, overlay), 0.5)
        else:
            return Image.alpha_composite(background, overlay)

    # Function to apply color fill to non-transparent pixels
    def apply_color_fill(image, color):
        if color:
            color_image = Image.new('RGBA', image.size, color)
            return Image.composite(color_image, image, image.split()[3])
        return image

    # Function to trim a PNG image and remove unused pixels
    def trim_png(image):
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        alpha = image.getchannel('A')
        bbox = alpha.getbbox()

        if bbox:
            return image.crop(bbox)
        else:
            return image

    for asset in variables['png_assets']:
        png_asset = Image.open(asset['path']).convert('RGBA')

        # Apply bilinear warp if bilinear_points are provided
        if asset['bilinear_points']:
            points = [float(coord) for coord in asset['bilinear_points'].split()]
            if len(points) == 8:
                x1, y1, x2, y2, x3, y3, x4, y4 = points
                warped_asset = bilinear_warp(x1, y1, x2, y2, x3, y3, x4, y4, asset['path'])
                png_asset = Image.fromarray(cv2.cvtColor(warped_asset, cv2.COLOR_BGRA2RGBA))

        # Now trim the PNG after applying bilinear warp (if any)
        png_asset = trim_png(png_asset)
        png_asset_width, png_asset_height = png_asset.size

        resize_ratio = asset['max_width'] / png_asset_width
        new_png_asset_width = asset['max_width']
        new_png_asset_height = int(png_asset_height * resize_ratio)
        png_asset = png_asset.resize((new_png_asset_width, new_png_asset_height), Image.Resampling.LANCZOS)

        png_asset_width, png_asset_height = png_asset.size  # Get new dimensions after potential resize

        # Apply color fill
        png_asset = apply_color_fill(png_asset, asset['color_fill'])

        # Apply fill intensity
        png_asset_array = list(png_asset.split())
        png_asset_array[3] = png_asset_array[3].point(lambda x: int(x * asset['fill_intensity']))
        png_asset = Image.merge('RGBA', png_asset_array)

        # Calculate the coordinates based on the reference point
        x_center, y_center = asset['coordinates']
        x, y = calculate_coordinates(x_center, y_center, png_asset_width, png_asset_height, asset['reference_point'])

        # Create a new image for blending
        asset_layer = Image.new('RGBA', background.size, (0, 0, 0, 0))
        asset_layer.paste(png_asset, (x, y), png_asset)

        # Apply blending mode
        background = blend_images(background, asset_layer, asset['blend_mode'])

    # Save the final image
    background.save(variables['final_image_name'], 'PNG')
    print(f"Image processing complete. Result saved as '{variables['final_image_name']}'")

    return variables['final_image_name']



import os
import shutil
import zipfile
import requests
import uuid

# Define Function to upload_directory_zip to server
def upload_directory_zip(source_directory):
    # Create a zip file
    parent_directory = os.path.dirname(source_directory)
    zip_filename = os.path.join(parent_directory, f"{os.path.basename(source_directory)}_BRANDING_{str(uuid.uuid4().hex)[:4]}.zip")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, filenames in os.walk(source_directory):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                arcname = os.path.relpath(file_path, source_directory)
                zipf.write(file_path, arcname)

    shutil.copy(zip_filename, final_exports_dir)            

    # # Upload the zip file to the server
    # with open(zip_filename, 'rb') as f:
    #     response = requests.post(
    #         'https://bongographics.com/mockups_projects/upload.php',
    #         files={'zip_file': f}
    #     )

    # # Cleanup: Remove the local zip file after uploading
    # os.remove(zip_filename)

    # website_link = f"https://bongographics.com/mockups_projects/{zip_filename}"
    # if response.status_code == 200:
    #     print(response.json())
    #     print(f"Website Link: {website_link}")
    #     return website_link
    # else:
    #     print(f"Failed to upload files. Status code: {response.status_code}, Response: {response.text}")
    # 

# Define Function to generate_office_set_v1_image
def generate_office_set_v1_image(state):
     print('-> Calling generate_office_set_v1_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v1']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v1_image_name': generated_image_name }


# Define Function to generate_office_set_v2_image
def generate_office_set_v2_image(state):
     print('-> Calling generate_office_set_v2_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v2']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v2_image_name': generated_image_name }

# Define Function to generate_office_set_v3_image
def generate_office_set_v3_image(state):
     print('-> Calling generate_office_set_v3_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v3']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v3_image_name': generated_image_name }

# Define Function to generate_office_set_v4_image
def generate_office_set_v4_image(state):
     print('-> Calling generate_office_set_v4_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v4']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v4_image_name': generated_image_name }

# Define Function to generate_office_set_v5_image
def generate_office_set_v5_image(state):
     print('-> Calling generate_office_set_v5_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v5']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v5_image_name': generated_image_name }

# Define Function to generate_office_set_v6_image
def generate_office_set_v6_image(state):
     print('-> Calling generate_office_set_v6_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v6']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v6_image_name': generated_image_name }

# Define Function to generate_office_set_v7_image
def generate_office_set_v7_image(state):
     print('-> Calling generate_office_set_v7_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v7']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v7_image_name': generated_image_name }

# Define Function to generate_office_set_v8_image
def generate_office_set_v8_image(state):
     print('-> Calling generate_office_set_v8_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v8']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v8_image_name': generated_image_name }

# Define Function to generate_office_set_v9_image
def generate_office_set_v9_image(state):
     print('-> Calling generate_office_set_v9_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v9']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v9_image_name': generated_image_name }

# Define Function to generate_office_set_v10_image
def generate_office_set_v10_image(state):
     print('-> Calling generate_office_set_v10_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v10']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v10_image_name': generated_image_name }

# Define Function to generate_office_set_v11_image
def generate_office_set_v11_image(state):
     print('-> Calling generate_office_set_v11_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v11']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v11_image_name': generated_image_name }

# Define Function to generate_office_set_v12_image
def generate_office_set_v12_image(state):
     print('-> Calling generate_office_set_v12_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v12']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v12_image_name': generated_image_name }

# Define Function to generate_office_set_v13_image
def generate_office_set_v13_image(state):
     print('-> Calling generate_office_set_v13_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v13']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v13_image_name': generated_image_name }

# Define Function to generate_office_set_v14_image
def generate_office_set_v14_image(state):
     print('-> Calling generate_office_set_v14_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v14']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v14_image_name': generated_image_name }

# Define Function to generate_office_set_v15_image
def generate_office_set_v15_image(state):
     print('-> Calling generate_office_set_v15_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v15']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v15_image_name': generated_image_name }

# Define Function to generate_office_set_v16_image
def generate_office_set_v16_image(state):
     print('-> Calling generate_office_set_v16_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v16']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v16_image_name': generated_image_name }

# Define Function to generate_office_set_v17_image
def generate_office_set_v17_image(state):
     print('-> Calling generate_office_set_v17_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v17']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v17_image_name': generated_image_name }

# Define Function to generate_office_set_v18_image
def generate_office_set_v18_image(state):
     print('-> Calling generate_office_set_v18_image Function ->')
     nested_variables_obj = state['nested_variables']
     variables = nested_variables_obj['st_variables']['office_set_v18']
     generated_image_name = manipulate_selected_image(variables)
     return { 'office_set_v18_image_name': generated_image_name }



from langgraph.graph import StateGraph, START, END

# Define MainGrap
main_workflow = StateGraph(ParentState)

# Add vital nodes
main_workflow.add_node("Prepare the State", prepare_state)
main_workflow.add_node("Generate Office Set V1 Image", generate_office_set_v1_image)
main_workflow.add_node("Generate Office Set V2 Image", generate_office_set_v2_image)
main_workflow.add_node("Generate Office Set V3 Image", generate_office_set_v3_image)
main_workflow.add_node("Generate Office Set V4 Image", generate_office_set_v4_image)
main_workflow.add_node("Generate Office Set V5 Image", generate_office_set_v5_image)
main_workflow.add_node("Generate Office Set V6 Image", generate_office_set_v6_image)
main_workflow.add_node("Generate Office Set V7 Image", generate_office_set_v7_image)
main_workflow.add_node("Generate Office Set V8 Image", generate_office_set_v8_image)
main_workflow.add_node("Generate Office Set V9 Image", generate_office_set_v9_image)
main_workflow.add_node("Generate Office Set V10 Image", generate_office_set_v10_image)
main_workflow.add_node("Generate Office Set V11 Image", generate_office_set_v11_image)
main_workflow.add_node("Generate Office Set V12 Image", generate_office_set_v12_image)
main_workflow.add_node("Generate Office Set V13 Image", generate_office_set_v13_image)
main_workflow.add_node("Generate Office Set V14 Image", generate_office_set_v14_image)
main_workflow.add_node("Generate Office Set V15 Image", generate_office_set_v15_image)
main_workflow.add_node("Generate Office Set V16 Image", generate_office_set_v16_image)
main_workflow.add_node("Generate Office Set V17 Image", generate_office_set_v17_image)
main_workflow.add_node("Generate Office Set V18 Image", generate_office_set_v18_image)

# Add vital edges
main_workflow.add_edge(START, "Prepare the State")
main_workflow.add_edge("Prepare the State", "Generate Office Set V1 Image")
main_workflow.add_edge("Prepare the State", "Generate Office Set V2 Image")
main_workflow.add_edge("Prepare the State", "Generate Office Set V3 Image")
main_workflow.add_edge("Generate Office Set V1 Image", "Generate Office Set V4 Image")
main_workflow.add_edge("Generate Office Set V2 Image", "Generate Office Set V5 Image")
main_workflow.add_edge("Generate Office Set V3 Image", "Generate Office Set V6 Image")
main_workflow.add_edge("Generate Office Set V4 Image", "Generate Office Set V7 Image")
main_workflow.add_edge("Generate Office Set V5 Image", "Generate Office Set V8 Image")
main_workflow.add_edge("Generate Office Set V6 Image", "Generate Office Set V9 Image")
main_workflow.add_edge("Generate Office Set V7 Image", "Generate Office Set V10 Image")
main_workflow.add_edge("Generate Office Set V8 Image", "Generate Office Set V11 Image")
main_workflow.add_edge("Generate Office Set V9 Image", "Generate Office Set V12 Image")
main_workflow.add_edge("Generate Office Set V10 Image", "Generate Office Set V13 Image")
main_workflow.add_edge("Generate Office Set V11 Image", "Generate Office Set V14 Image")
main_workflow.add_edge("Generate Office Set V12 Image", "Generate Office Set V15 Image")
main_workflow.add_edge("Generate Office Set V13 Image", "Generate Office Set V16 Image")
main_workflow.add_edge("Generate Office Set V14 Image", "Generate Office Set V17 Image")
main_workflow.add_edge("Generate Office Set V15 Image", "Generate Office Set V18 Image")
main_workflow.add_edge("Generate Office Set V16 Image", END)
main_workflow.add_edge("Generate Office Set V17 Image", END)
main_workflow.add_edge("Generate Office Set V18 Image", END)


# Compile the Main Graph
graph = main_workflow.compile()



###  GRAPH START BULK


from PIL import Image
import os
import json
import copy

resized_bulk_pngLogos_dir = 'resized_bulk_pngLogos_dir'
bulk_pngLogos_dir = 'bulk_pngLogos_dir'

def trim_png(image):
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    alpha = image.getchannel('A')
    bbox = alpha.getbbox()
    if bbox:
        return image.crop(bbox)
    else:
        return image

def process_image(input_path, output_path):
    # Open the image
    with Image.open(input_path) as img:
        # Trim the image
        trimmed_img = trim_png(img)

        # Create a new 1000x1000 transparent image
        new_img = Image.new('RGBA', (1000, 1000), (0, 0, 0, 0))

        # Calculate the scaling factor to fit within 1000x1000 while maintaining aspect ratio
        scale = min(1000 / trimmed_img.width, 1000 / trimmed_img.height)
        new_size = (int(trimmed_img.width * scale), int(trimmed_img.height * scale))

        # Resize the trimmed image
        resized_img = trimmed_img.resize(new_size, Image.LANCZOS)

        # Calculate position to paste in the center
        position = ((1000 - new_size[0]) // 2, (1000 - new_size[1]) // 2)

        # Paste the resized image onto the new image
        new_img.paste(resized_img, position, resized_img)

        # Save the result with 300 DPI
        new_img.save(output_path, 'PNG', dpi=(300, 300))

# Delete the directory if it exists
if os.path.exists(resized_bulk_pngLogos_dir):
    shutil.rmtree(resized_bulk_pngLogos_dir)

# Create a new directory
os.makedirs(resized_bulk_pngLogos_dir, exist_ok=True)

# Process all PNG images in bulk_pngLogos_dir
for filename in os.listdir(bulk_pngLogos_dir):
    if filename.endswith('.png'):
        input_path = os.path.join(bulk_pngLogos_dir, filename)
        output_path = os.path.join(resized_bulk_pngLogos_dir, filename)
        process_image(input_path, output_path)
        print(f"Processed image {filename} saved to {resized_bulk_pngLogos_dir}")



png_files = [f for f in os.listdir(resized_bulk_pngLogos_dir) if f.endswith('.png')]

def update_nested_variables(obj, old_value, new_value, key=None, partial_replace=False):
    """
    Recursively update the nested dictionary.
    If partial_replace is True, only replace the partial string within the value.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                update_nested_variables(v, old_value, new_value, key=k, partial_replace=partial_replace)
            elif isinstance(v, str):
                if v == old_value:
                    obj[k] = new_value
                elif partial_replace and k == 'final_image_name':
                    obj[k] = v.replace(old_value, new_value)
    elif isinstance(obj, list):
        for item in obj:
            update_nested_variables(item, old_value, new_value, key=key, partial_replace=partial_replace)

for png_file in png_files:
    # Get the name of the file without its extension
    file_name = os.path.splitext(png_file)[0]
    # Get the name of the file with its extension
    file_name_with_extension = png_file
    print(f"Processing file: {file_name_with_extension}")
    
    # Create output Directories inside resized_bulk_pngLogos_dir
    output_dir = os.path.join(resized_bulk_pngLogos_dir, f"{file_name}_outputs")
    if os.path.exists(output_dir):
        os.system(f"rm -r {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a copy of the nested_variables object to work with
    updated_nested_variables = copy.deepcopy(nested_variables)
    new_png_path = os.path.join(resized_bulk_pngLogos_dir, file_name_with_extension)
    print(f"New PNG Path: {new_png_path}")
    
    # Check if file_name ends with any of the specified colors
    colors = ['blue', 'green', 'purple', 'cyan', 'yellow', 'orange', 'pink', 'red']
    new_value = 'blue'  # Default value
    
    for color in colors:
        if file_name.endswith(color):
            new_value = color
            break
    
    # Perform the replacements
    update_nested_variables(updated_nested_variables, 'my_logo1.png', new_png_path)
    update_nested_variables(updated_nested_variables, 'outputs', output_dir, partial_replace=True)
    update_nested_variables(updated_nested_variables, key='hue_shift', old_value='blue', new_value=new_value)
    
    # Invoke the Graph
    inputs = {"nested_variables": updated_nested_variables}
    outputs = graph.invoke(inputs)
    print(outputs)
    upload_directory_zip(output_dir)

              