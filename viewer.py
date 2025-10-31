import sys
import os
import shutil
import logging
from typing import Optional, Tuple
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QFileDialog, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QMessageBox, QProgressDialog)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np
from PIL import Image, ImageOps
from stl import mesh
import tempfile
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_WIDTH = 100
DEFAULT_HEIGHT = 100
PREVIEW_SIZE = 300
MAX_DIMENSION = 1000
MIN_DIMENSION = 1
DEFAULT_BASE_HEIGHT = 2.0
DEFAULT_MAX_HEIGHT = 4.0
DEFAULT_MIN_THICKNESS = 0.5
PREVIEW_DEBOUNCE_MS = 500

def rgb2gray(image_path: str, save_path: str, invert: bool = False) -> str:
    """
    Converts image from RGB to Grayscale.

    Args:
        image_path: Input image file path
        save_path: Directory to save the processed image
        invert: Whether to invert the image

    Returns:
        Path of the converted image

    Raises:
        FileNotFoundError: If input image does not exist
        IOError: If image cannot be processed or saved
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    try:
        # Open image
        with Image.open(image_path) as img:
            # Convert RGB to grayscale
            gray_img = img.convert('L')

            # Invert if requested
            if invert:
                gray_img = ImageOps.invert(gray_img)

            # Generate new filename
            new_filename = "grayscale.png"
            new_path = os.path.join(save_path, new_filename)

            # Save the processed image
            gray_img.save(new_path)

        logger.info(f"Converted image to grayscale: {new_path}")
        return new_path
    except Exception as e:
        logger.error(f"Failed to convert image to grayscale: {e}")
        raise IOError(f"Failed to convert image to grayscale: {e}")

def resize_image(image_path: str, save_path: str, ratio: Tuple[int, int], resize_type: int = 1) -> str:
    """
    Resizes image into target ratio with selected resizing method.

    Args:
        image_path: Input image file path
        save_path: Directory to save the processed image
        ratio: Target (width, height)
        resize_type: Resize method (1=Crop, 2=Pad, 3=Stretch)

    Returns:
        Path of the resized image

    Raises:
        FileNotFoundError: If input image does not exist
        ValueError: If invalid resize_type or dimensions
        IOError: If image cannot be processed or saved
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    target_width, target_height = ratio

    # Validate dimensions
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Invalid dimensions: {ratio}. Width and height must be positive.")

    if resize_type not in (1, 2, 3):
        raise ValueError(f"Invalid resize type: {resize_type}. Must be 1 (Crop), 2 (Pad), or 3 (Stretch).")

    try:
        # Open image
        with Image.open(image_path) as img:
            # Get original image size
            orig_width, orig_height = img.size

            if orig_width == 0 or orig_height == 0:
                raise ValueError(f"Invalid image dimensions: {orig_width}x{orig_height}")

            if resize_type == 1:  # Cropping (Centered)
                resized_img = _crop_image(img, target_width, target_height, orig_width, orig_height)

            elif resize_type == 2:  # Padding (Centered)
                resized_img = _pad_image(img, target_width, target_height, orig_width, orig_height)

            elif resize_type == 3:  # Stretching
                resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

            # Generate new filename
            _, filename = os.path.split(image_path)
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_resized{ext}"
            new_path = os.path.join(save_path, new_filename)

            # Check if filename already exists and modify if necessary
            counter = 1
            while os.path.exists(new_path):
                new_filename = f"{name}_resized({counter}){ext}"
                new_path = os.path.join(save_path, new_filename)
                counter += 1

            # Save the resized image
            resized_img.save(new_path)

        logger.info(f"Resized image saved: {new_path}")
        return new_path
    except Exception as e:
        logger.error(f"Failed to resize image: {e}")
        raise IOError(f"Failed to resize image: {e}")


def _crop_image(img: Image.Image, target_width: int, target_height: int,
                orig_width: int, orig_height: int) -> Image.Image:
    """Crop image to target aspect ratio (centered)."""
    target_ratio = target_width / target_height
    current_ratio = orig_width / orig_height

    if current_ratio > target_ratio:
        # Wider than target aspect ratio
        new_width = int(orig_height * target_ratio)
        left = (orig_width - new_width) // 2
        right = left + new_width
        top = 0
        bottom = orig_height
    else:
        # Taller than target aspect ratio
        new_height = int(orig_width / target_ratio)
        left = 0
        right = orig_width
        top = (orig_height - new_height) // 2
        bottom = top + new_height

    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _pad_image(img: Image.Image, target_width: int, target_height: int,
               orig_width: int, orig_height: int) -> Image.Image:
    """Pad image to target aspect ratio (centered)."""
    target_ratio = target_width / target_height
    current_ratio = orig_width / orig_height

    if current_ratio > target_ratio:
        # Wider than target aspect ratio
        new_width = orig_width
        new_height = int(orig_width / target_ratio)
    else:
        # Taller than target aspect ratio
        new_width = int(orig_height * target_ratio)
        new_height = orig_height

    pad_left = (new_width - orig_width) // 2
    pad_right = new_width - orig_width - pad_left
    pad_top = (new_height - orig_height) // 2
    pad_bottom = new_height - orig_height - pad_top

    padded_img = ImageOps.expand(img, (pad_left, pad_top, pad_right, pad_bottom))
    return padded_img.resize((target_width, target_height), Image.Resampling.LANCZOS)

def grayscale_to_lithophane_flat(input_filepath: str, output_filepath: Optional[str] = None,
                           base_height: float = DEFAULT_BASE_HEIGHT,
                           max_height: float = DEFAULT_MAX_HEIGHT,
                           min_thickness: float = DEFAULT_MIN_THICKNESS) -> str:
    """
    Convert a grayscale image to a lithophane STL file.

    Parameters:
        input_filepath: Path to input grayscale image file
        output_filepath: Path for output STL file. If None, uses input path with .stl extension
        base_height: Base height of the lithophane in mm
        max_height: Maximum additional height from base in mm
        min_thickness: Minimum thickness of thinnest parts in mm

    Returns:
        Path to the generated STL file

    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If invalid parameters provided
        IOError: If mesh cannot be created or saved
    """
    # Verify input file exists
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"Input file not found: {input_filepath}")

    # Validate parameters
    if base_height <= 0 or max_height <= 0 or min_thickness <= 0:
        raise ValueError("All height parameters must be positive")

    # Set output filepath if not provided
    if output_filepath is None:
        output_filepath = os.path.splitext(input_filepath)[0] + '.stl'

    try:
        # Read and verify grayscale image
        with Image.open(input_filepath) as img:
            if img.mode != 'L':
                img = img.convert('L')

            # Convert image to numpy array and normalize to [0, 1]
            img_array = np.array(img).astype(float) / 255.0
    
        # Get image dimensions
        height, width = img_array.shape

        # --- Vectorized Vertex Creation ---

        # Create a grid of x, y coordinates
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)

        # Calculate z coordinates for the top surface
        z_top = base_height + (1 - img_array) * max_height + min_thickness

        # Create top and bottom vertices
        top_vertices = np.stack([xx.ravel(), yy.ravel(), z_top.ravel()], axis=1)
        bottom_vertices = np.stack([xx.ravel(), yy.ravel(), np.zeros_like(z_top).ravel()], axis=1)

        all_vertices = np.vstack([top_vertices, bottom_vertices])

        # --- Vectorized Face Creation ---

        # Create indices for the grid
        i = np.arange(height * width).reshape(height, width)

        # Top faces
        v1_top = i[:-1, :-1].ravel()
        v2_top = i[:-1, 1:].ravel()
        v3_top = i[1:, :-1].ravel()
        v4_top = i[1:, 1:].ravel()

        top_faces = np.vstack([
            np.stack([v1_top, v2_top, v3_top], axis=1),
            np.stack([v2_top, v4_top, v3_top], axis=1)
        ])

        # Bottom faces (inverted)
        bottom_faces = np.vstack([
            np.stack([v1_top, v3_top, v2_top], axis=1),
            np.stack([v2_top, v3_top, v4_top], axis=1)
        ]) + (height * width)

        # Side faces
        side_faces = []

        # Front and back walls
        for x in range(width - 1):
            # Front
            v1 = x
            v2 = x + 1
            v3 = v1 + height * width
            v4 = v2 + height * width
            side_faces.extend([[v1, v3, v2], [v2, v3, v4]])

            # Back
            v1 = (height - 1) * width + x
            v2 = v1 + 1
            v3 = v1 + height * width
            v4 = v2 + height * width
            side_faces.extend([[v1, v2, v3], [v2, v4, v3]])

        # Left and right walls
        for y in range(height - 1):
            # Left
            v1 = y * width
            v2 = (y + 1) * width
            v3 = v1 + height * width
            v4 = v2 + height * width
            side_faces.extend([[v1, v3, v2], [v2, v3, v4]])

            # Right
            v1 = y * width + width - 1
            v2 = (y + 1) * width + width - 1
            v3 = v1 + height * width
            v4 = v2 + height * width
            side_faces.extend([[v1, v2, v3], [v2, v4, v3]])

        all_faces = np.vstack([top_faces, bottom_faces, np.array(side_faces)])

        # --- Mesh Creation ---

        # Create the mesh
        lithophane_mesh = mesh.Mesh(np.zeros(all_faces.shape[0], dtype=mesh.Mesh.dtype))

        # Assign vertices to the mesh
        lithophane_mesh.vectors = all_vertices[all_faces]

        # Save the mesh
        lithophane_mesh.save(output_filepath)

        logger.info(f"Lithophane STL saved: {output_filepath}")
        return output_filepath
    except Exception as e:
        logger.error(f"Failed to generate lithophane: {e}")
        raise IOError(f"Failed to generate lithophane: {e}")

def grayscale_to_lithophane_box(input_filepath: str, output_filepath: Optional[str] = None,
                                base_height: float = DEFAULT_BASE_HEIGHT,
                                max_height: float = DEFAULT_MAX_HEIGHT,
                                min_thickness: float = DEFAULT_MIN_THICKNESS,
                                box_width: float = 50.0,
                                box_depth: float = 50.0) -> str:
    """
    Convert a grayscale image to a box-shaped (rectangular prism) lithophane STL file.
    The flat lithophane becomes the side walls of the box.

    Parameters:
        input_filepath: Path to input grayscale image file
        output_filepath: Path for output STL file. If None, uses input path with .stl extension
        base_height: Base height of the lithophane in mm
        max_height: Maximum additional height from base in mm
        min_thickness: Minimum thickness of thinnest parts in mm
        box_width: Width of the box (X dimension) in mm
        box_depth: Depth of the box (Y dimension) in mm

    Returns:
        Path to the generated STL file

    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If invalid parameters provided
        IOError: If mesh cannot be created or saved
    """
    # Verify input file exists
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"Input file not found: {input_filepath}")

    # Validate parameters
    if base_height <= 0 or max_height <= 0 or min_thickness <= 0:
        raise ValueError("All height parameters must be positive")
    if box_width <= 0 or box_depth <= 0:
        raise ValueError("Box dimensions must be positive")

    # Set output filepath if not provided
    if output_filepath is None:
        output_filepath = os.path.splitext(input_filepath)[0] + '_box.stl'

    try:
        # Read and verify grayscale image
        with Image.open(input_filepath) as img:
            if img.mode != 'L':
                img = img.convert('L')

            # Convert image to numpy array and normalize to [0, 1]
            img_array = np.array(img).astype(float) / 255.0

        # Get image dimensions
        img_height, img_width = img_array.shape

        # Box dimensions (in mm)
        box_height = img_height  # Height of the box equals image height

        # Calculate thickness variation for each pixel
        thickness_values = base_height + (1 - img_array) * max_height + min_thickness

        # We'll create 4 walls, each using the lithophane pattern
        all_vertices = []
        all_faces = []
        vertex_offset = 0

        # Wall 1: Front (along X axis, at Y=0)
        for y in range(img_height):
            for x in range(img_width):
                # Map image coordinates to box coordinates
                box_x = (x / img_width) * box_width
                box_z = (y / img_height) * box_height
                thickness = thickness_values[y, x]

                # Outer surface (front face)
                all_vertices.append([box_x, 0, box_z])
                # Inner surface
                all_vertices.append([box_x, thickness, box_z])

        # Create faces for front wall
        for y in range(img_height - 1):
            for x in range(img_width - 1):
                idx = y * img_width + x
                # Outer face indices
                v1_out = vertex_offset + idx * 2
                v2_out = vertex_offset + (idx + 1) * 2
                v3_out = vertex_offset + (idx + img_width) * 2
                v4_out = vertex_offset + (idx + img_width + 1) * 2

                # Inner face indices
                v1_in = v1_out + 1
                v2_in = v2_out + 1
                v3_in = v3_out + 1
                v4_in = v4_out + 1

                # Outer faces (facing out)
                all_faces.append([v1_out, v2_out, v3_out])
                all_faces.append([v2_out, v4_out, v3_out])

                # Inner faces (facing in)
                all_faces.append([v1_in, v3_in, v2_in])
                all_faces.append([v2_in, v3_in, v4_in])

        vertex_offset += img_height * img_width * 2

        # Wall 2: Right side (along Y axis, at X=box_width)
        for y in range(img_height):
            for x in range(img_width):
                box_y = (x / img_width) * box_depth
                box_z = (y / img_height) * box_height
                thickness = thickness_values[y, x]

                # Outer surface (right face)
                all_vertices.append([box_width, box_y, box_z])
                # Inner surface
                all_vertices.append([box_width - thickness, box_y, box_z])

        # Create faces for right wall
        for y in range(img_height - 1):
            for x in range(img_width - 1):
                idx = y * img_width + x
                v1_out = vertex_offset + idx * 2
                v2_out = vertex_offset + (idx + 1) * 2
                v3_out = vertex_offset + (idx + img_width) * 2
                v4_out = vertex_offset + (idx + img_width + 1) * 2

                v1_in = v1_out + 1
                v2_in = v2_out + 1
                v3_in = v3_out + 1
                v4_in = v4_out + 1

                # Outer faces
                all_faces.append([v1_out, v2_out, v3_out])
                all_faces.append([v2_out, v4_out, v3_out])

                # Inner faces
                all_faces.append([v1_in, v3_in, v2_in])
                all_faces.append([v2_in, v3_in, v4_in])

        vertex_offset += img_height * img_width * 2

        # Wall 3: Back (along X axis, at Y=box_depth)
        for y in range(img_height):
            for x in range(img_width):
                box_x = box_width - (x / img_width) * box_width  # Reverse X for back wall
                box_z = (y / img_height) * box_height
                thickness = thickness_values[y, x]

                # Outer surface
                all_vertices.append([box_x, box_depth, box_z])
                # Inner surface
                all_vertices.append([box_x, box_depth - thickness, box_z])

        # Create faces for back wall
        for y in range(img_height - 1):
            for x in range(img_width - 1):
                idx = y * img_width + x
                v1_out = vertex_offset + idx * 2
                v2_out = vertex_offset + (idx + 1) * 2
                v3_out = vertex_offset + (idx + img_width) * 2
                v4_out = vertex_offset + (idx + img_width + 1) * 2

                v1_in = v1_out + 1
                v2_in = v2_out + 1
                v3_in = v3_out + 1
                v4_in = v4_out + 1

                # Outer faces
                all_faces.append([v1_out, v2_out, v3_out])
                all_faces.append([v2_out, v4_out, v3_out])

                # Inner faces
                all_faces.append([v1_in, v3_in, v2_in])
                all_faces.append([v2_in, v3_in, v4_in])

        vertex_offset += img_height * img_width * 2

        # Wall 4: Left side (along Y axis, at X=0)
        for y in range(img_height):
            for x in range(img_width):
                box_y = box_depth - (x / img_width) * box_depth  # Reverse Y for left wall
                box_z = (y / img_height) * box_height
                thickness = thickness_values[y, x]

                # Outer surface
                all_vertices.append([0, box_y, box_z])
                # Inner surface
                all_vertices.append([thickness, box_y, box_z])

        # Create faces for left wall
        for y in range(img_height - 1):
            for x in range(img_width - 1):
                idx = y * img_width + x
                v1_out = vertex_offset + idx * 2
                v2_out = vertex_offset + (idx + 1) * 2
                v3_out = vertex_offset + (idx + img_width) * 2
                v4_out = vertex_offset + (idx + img_width + 1) * 2

                v1_in = v1_out + 1
                v2_in = v2_out + 1
                v3_in = v3_out + 1
                v4_in = v4_out + 1

                # Outer faces
                all_faces.append([v1_out, v2_out, v3_out])
                all_faces.append([v2_out, v4_out, v3_out])

                # Inner faces
                all_faces.append([v1_in, v3_in, v2_in])
                all_faces.append([v2_in, v3_in, v4_in])

        vertex_offset += img_height * img_width * 2

        # Add top and bottom caps
        # Top cap (at Z=box_height)
        cap_vertices_top = [
            [0, 0, box_height],
            [box_width, 0, box_height],
            [box_width, box_depth, box_height],
            [0, box_depth, box_height]
        ]

        top_idx = vertex_offset
        all_vertices.extend(cap_vertices_top)
        all_faces.append([top_idx, top_idx + 1, top_idx + 2])
        all_faces.append([top_idx, top_idx + 2, top_idx + 3])

        # Bottom cap (at Z=0)
        cap_vertices_bottom = [
            [0, 0, 0],
            [box_width, 0, 0],
            [box_width, box_depth, 0],
            [0, box_depth, 0]
        ]

        bottom_idx = vertex_offset + 4
        all_vertices.extend(cap_vertices_bottom)
        all_faces.append([bottom_idx, bottom_idx + 2, bottom_idx + 1])
        all_faces.append([bottom_idx, bottom_idx + 3, bottom_idx + 2])

        # Convert to numpy arrays
        vertices_array = np.array(all_vertices)
        faces_array = np.array(all_faces)

        # Create the mesh
        lithophane_mesh = mesh.Mesh(np.zeros(faces_array.shape[0], dtype=mesh.Mesh.dtype))

        # Assign vertices to the mesh
        lithophane_mesh.vectors = vertices_array[faces_array]

        # Save the mesh
        lithophane_mesh.save(output_filepath)

        logger.info(f"Box lithophane STL saved: {output_filepath}")
        return output_filepath
    except Exception as e:
        logger.error(f"Failed to generate box lithophane: {e}")
        raise IOError(f"Failed to generate box lithophane: {e}")

def grayscale_to_lithophane_cylinder(input_filepath: str, output_filepath: Optional[str] = None,
                                     base_height: float = DEFAULT_BASE_HEIGHT,
                                     max_height: float = DEFAULT_MAX_HEIGHT,
                                     min_thickness: float = DEFAULT_MIN_THICKNESS,
                                     radius: float = 30.0,
                                     arc_angle: float = 360.0,
                                     add_caps: bool = True) -> str:
    """
    Convert a grayscale image to a cylindrical lithophane STL file.
    The flat lithophane is curved around a cylinder.

    Parameters:
        input_filepath: Path to input grayscale image file
        output_filepath: Path for output STL file. If None, uses input path with .stl extension
        base_height: Base height of the lithophane in mm
        max_height: Maximum additional height from base in mm
        min_thickness: Minimum thickness of thinnest parts in mm
        radius: Inner radius of the cylinder in mm
        arc_angle: Arc angle in degrees (180, 270, 360, etc.)
        add_caps: Whether to add top and bottom caps

    Returns:
        Path to the generated STL file

    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If invalid parameters provided
        IOError: If mesh cannot be created or saved
    """
    # Verify input file exists
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"Input file not found: {input_filepath}")

    # Validate parameters
    if base_height <= 0 or max_height <= 0 or min_thickness <= 0:
        raise ValueError("All height parameters must be positive")
    if radius <= 0:
        raise ValueError("Radius must be positive")
    if arc_angle <= 0 or arc_angle > 360:
        raise ValueError("Arc angle must be between 0 and 360 degrees")

    # Set output filepath if not provided
    if output_filepath is None:
        output_filepath = os.path.splitext(input_filepath)[0] + '_cylinder.stl'

    try:
        # Read and verify grayscale image
        with Image.open(input_filepath) as img:
            if img.mode != 'L':
                img = img.convert('L')

            # Convert image to numpy array and normalize to [0, 1]
            img_array = np.array(img).astype(float) / 255.0

        # Get image dimensions
        img_height, img_width = img_array.shape

        # Convert arc angle to radians
        arc_radians = np.radians(arc_angle)

        # Calculate thickness for each pixel
        thickness_values = base_height + (1 - img_array) * max_height + min_thickness

        # Create vertices
        all_vertices = []

        # Create cylindrical mesh
        for y in range(img_height):
            for x in range(img_width):
                # Map x coordinate to angle (theta)
                theta = (x / (img_width - 1)) * arc_radians

                # Map y coordinate to height (z)
                z = (y / (img_height - 1)) * img_height

                # Get thickness at this pixel
                thickness = thickness_values[y, x]

                # Outer surface (inner cylinder radius)
                outer_x = radius * np.cos(theta)
                outer_y = radius * np.sin(theta)
                all_vertices.append([outer_x, outer_y, z])

                # Inner surface (outer cylinder radius)
                inner_radius = radius + thickness
                inner_x = inner_radius * np.cos(theta)
                inner_y = inner_radius * np.sin(theta)
                all_vertices.append([inner_x, inner_y, z])

        vertices_array = np.array(all_vertices)

        # Create faces
        all_faces = []

        # Cylindrical surface faces
        for y in range(img_height - 1):
            for x in range(img_width - 1):
                idx = y * img_width + x

                # Outer surface indices
                v1_out = idx * 2
                v2_out = (idx + 1) * 2
                v3_out = (idx + img_width) * 2
                v4_out = (idx + img_width + 1) * 2

                # Inner surface indices
                v1_in = v1_out + 1
                v2_in = v2_out + 1
                v3_in = v3_out + 1
                v4_in = v4_out + 1

                # Outer faces (facing inward - reversed winding)
                all_faces.append([v1_out, v3_out, v2_out])
                all_faces.append([v2_out, v3_out, v4_out])

                # Inner faces (facing outward)
                all_faces.append([v1_in, v2_in, v3_in])
                all_faces.append([v2_in, v4_in, v3_in])

        # Add edge faces if not a complete 360 cylinder
        if arc_angle < 360:
            # Left edge (at theta=0)
            for y in range(img_height - 1):
                idx = y * img_width
                v1_out = idx * 2
                v1_in = v1_out + 1
                v2_out = (idx + img_width) * 2
                v2_in = v2_out + 1

                all_faces.append([v1_out, v1_in, v2_out])
                all_faces.append([v1_in, v2_in, v2_out])

            # Right edge (at theta=arc_angle)
            for y in range(img_height - 1):
                idx = y * img_width + (img_width - 1)
                v1_out = idx * 2
                v1_in = v1_out + 1
                v2_out = (idx + img_width) * 2
                v2_in = v2_out + 1

                all_faces.append([v1_out, v2_out, v1_in])
                all_faces.append([v1_in, v2_out, v2_in])

        # Add top and bottom edge loops
        # Bottom edge (at z=0)
        for x in range(img_width - 1):
            v1_out = x * 2
            v1_in = v1_out + 1
            v2_out = (x + 1) * 2
            v2_in = v2_out + 1

            all_faces.append([v1_out, v2_out, v1_in])
            all_faces.append([v1_in, v2_out, v2_in])

        # Top edge (at z=img_height)
        top_row_start = (img_height - 1) * img_width
        for x in range(img_width - 1):
            idx = top_row_start + x
            v1_out = idx * 2
            v1_in = v1_out + 1
            v2_out = (idx + 1) * 2
            v2_in = v2_out + 1

            all_faces.append([v1_out, v1_in, v2_out])
            all_faces.append([v1_in, v2_in, v2_out])

        # Add caps if requested
        if add_caps:
            cap_resolution = max(20, img_width // 2)  # Resolution for circular caps
            vertex_offset = len(vertices_array)

            # Bottom cap (at z=0)
            cap_vertices_bottom = []
            cap_center_bottom = [0, 0, 0]
            cap_vertices_bottom.append(cap_center_bottom)

            for i in range(cap_resolution + 1):
                theta = (i / cap_resolution) * arc_radians
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                cap_vertices_bottom.append([x, y, 0])

            bottom_center_idx = vertex_offset
            all_vertices.extend(cap_vertices_bottom)

            for i in range(cap_resolution):
                v1 = bottom_center_idx
                v2 = bottom_center_idx + i + 1
                v3 = bottom_center_idx + i + 2
                all_faces.append([v1, v3, v2])

            # Top cap (at z=img_height)
            vertex_offset = len(all_vertices)
            cap_vertices_top = []
            cap_center_top = [0, 0, img_height]
            cap_vertices_top.append(cap_center_top)

            for i in range(cap_resolution + 1):
                theta = (i / cap_resolution) * arc_radians
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                cap_vertices_top.append([x, y, img_height])

            top_center_idx = vertex_offset
            all_vertices.extend(cap_vertices_top)

            for i in range(cap_resolution):
                v1 = top_center_idx
                v2 = top_center_idx + i + 1
                v3 = top_center_idx + i + 2
                all_faces.append([v1, v2, v3])

        # Convert to numpy arrays
        vertices_array = np.array(all_vertices)
        faces_array = np.array(all_faces)

        # Create the mesh
        lithophane_mesh = mesh.Mesh(np.zeros(faces_array.shape[0], dtype=mesh.Mesh.dtype))

        # Assign vertices to the mesh
        lithophane_mesh.vectors = vertices_array[faces_array]

        # Save the mesh
        lithophane_mesh.save(output_filepath)

        logger.info(f"Cylindrical lithophane STL saved: {output_filepath}")
        return output_filepath
    except Exception as e:
        logger.error(f"Failed to generate cylindrical lithophane: {e}")
        raise IOError(f"Failed to generate cylindrical lithophane: {e}")

def grayscale_to_lithophane(input_filepath: str, output_filepath: Optional[str] = None,
                           base_height: float = DEFAULT_BASE_HEIGHT,
                           max_height: float = DEFAULT_MAX_HEIGHT,
                           min_thickness: float = DEFAULT_MIN_THICKNESS,
                           shape_type: str = 'flat',
                           **shape_params) -> str:
    """
    Convert a grayscale image to a lithophane STL file with various shapes.

    This is a wrapper function that calls the appropriate shape-specific function.

    Parameters:
        input_filepath: Path to input grayscale image file
        output_filepath: Path for output STL file
        base_height: Base height of the lithophane in mm
        max_height: Maximum additional height from base in mm
        min_thickness: Minimum thickness of thinnest parts in mm
        shape_type: Type of lithophane shape ('flat', 'box', 'cylinder')
        **shape_params: Additional parameters specific to each shape type
            For 'box': box_width, box_depth
            For 'cylinder': radius, arc_angle, add_caps

    Returns:
        Path to the generated STL file
    """
    if shape_type == 'flat':
        return grayscale_to_lithophane_flat(
            input_filepath, output_filepath,
            base_height, max_height, min_thickness
        )
    elif shape_type == 'box':
        return grayscale_to_lithophane_box(
            input_filepath, output_filepath,
            base_height, max_height, min_thickness,
            box_width=shape_params.get('box_width', 50.0),
            box_depth=shape_params.get('box_depth', 50.0)
        )
    elif shape_type == 'cylinder':
        return grayscale_to_lithophane_cylinder(
            input_filepath, output_filepath,
            base_height, max_height, min_thickness,
            radius=shape_params.get('radius', 30.0),
            arc_angle=shape_params.get('arc_angle', 360.0),
            add_caps=shape_params.get('add_caps', True)
        )
    else:
        raise ValueError(f"Unknown shape type: {shape_type}. Must be 'flat', 'box', or 'cylinder'.")

class LithophaneGenerator(QMainWindow):
    """Main application window for lithophane generation."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Lithophane Generator")
        self.resize(1200, 800)

        # Initialize variables
        self.current_image_path: Optional[str] = None
        self.processed_image_path: Optional[str] = None
        self.stl_path: Optional[str] = None
        self.preview_timer: Optional[QTimer] = None
        self.current_shape_type: str = 'flat'

        # Create temporary directory in a safe location
        try:
            # Get system temp directory
            system_temp = tempfile.gettempdir()
            # Create a unique subfolder name
            unique_folder = f"lithophane_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            # Combine paths safely
            self.temp_dir = os.path.join(system_temp, unique_folder)
            # Create directory
            os.makedirs(self.temp_dir, exist_ok=True)
            logger.info(f"Created temporary directory: {self.temp_dir}")
        except Exception as e:
            error_msg = f"Failed to create temporary directory: {str(e)}\n\nThe application will now exit."
            logger.critical(error_msg)
            QMessageBox.critical(self, "Critical Error", error_msg)
            sys.exit(1)

        # Create the main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Create left panel for controls
        self.create_left_panel()

        # Create right panel for visualization
        self.create_right_panel()

        # Set up VTK visualization
        self.setup_vtk()

        # Initially disable convert and export buttons
        self.convert_button.setEnabled(False)
        self.export_button.setEnabled(False)

    def create_left_panel(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Image loading section
        load_group = QGroupBox("Image Loading")
        load_layout = QVBoxLayout()
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        self.image_label = QLabel("No image loaded")
        load_layout.addWidget(self.load_button)
        load_layout.addWidget(self.image_label)
        load_group.setLayout(load_layout)
        left_layout.addWidget(load_group)

        # Image resize options
        resize_group = QGroupBox("Resize Options")
        resize_layout = QVBoxLayout()
        
        # Width and height inputs
        size_layout = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(MIN_DIMENSION, MAX_DIMENSION)
        self.width_spin.setValue(DEFAULT_WIDTH)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(MIN_DIMENSION, MAX_DIMENSION)
        self.height_spin.setValue(DEFAULT_HEIGHT)
        size_layout.addWidget(QLabel("Width:"))
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("Height:"))
        size_layout.addWidget(self.height_spin)
        resize_layout.addLayout(size_layout)

        # Resize method selection
        self.resize_method = QComboBox()
        self.resize_method.addItems(["Crop (Centered)", "Pad (Centered)", "Stretch"])
        resize_layout.addWidget(QLabel("Resize Method:"))
        resize_layout.addWidget(self.resize_method)
        resize_group.setLayout(resize_layout)
        left_layout.addWidget(resize_group)

        # Shape selection
        shape_group = QGroupBox("Lithophane Shape")
        shape_layout = QVBoxLayout()

        self.shape_type = QComboBox()
        self.shape_type.addItems(["Flat (Standard)", "Box (4 Walls)", "Cylinder (Curved)"])
        self.shape_type.currentIndexChanged.connect(self.on_shape_changed)
        shape_layout.addWidget(QLabel("Shape Type:"))
        shape_layout.addWidget(self.shape_type)

        shape_group.setLayout(shape_layout)
        left_layout.addWidget(shape_group)

        # Shape-specific parameters
        self.shape_params_group = QGroupBox("Shape Parameters")
        self.shape_params_layout = QVBoxLayout()

        # Box parameters
        self.box_params_widget = QWidget()
        box_params_layout = QVBoxLayout()

        box_width_layout = QHBoxLayout()
        self.box_width = QDoubleSpinBox()
        self.box_width.setRange(10.0, 200.0)
        self.box_width.setValue(50.0)
        self.box_width.setSingleStep(5.0)
        box_width_layout.addWidget(QLabel("Box Width (mm):"))
        box_width_layout.addWidget(self.box_width)
        box_params_layout.addLayout(box_width_layout)

        box_depth_layout = QHBoxLayout()
        self.box_depth = QDoubleSpinBox()
        self.box_depth.setRange(10.0, 200.0)
        self.box_depth.setValue(50.0)
        self.box_depth.setSingleStep(5.0)
        box_depth_layout.addWidget(QLabel("Box Depth (mm):"))
        box_depth_layout.addWidget(self.box_depth)
        box_params_layout.addLayout(box_depth_layout)

        self.box_params_widget.setLayout(box_params_layout)

        # Cylinder parameters
        self.cylinder_params_widget = QWidget()
        cylinder_params_layout = QVBoxLayout()

        radius_layout = QHBoxLayout()
        self.cylinder_radius = QDoubleSpinBox()
        self.cylinder_radius.setRange(10.0, 100.0)
        self.cylinder_radius.setValue(30.0)
        self.cylinder_radius.setSingleStep(5.0)
        radius_layout.addWidget(QLabel("Radius (mm):"))
        radius_layout.addWidget(self.cylinder_radius)
        cylinder_params_layout.addLayout(radius_layout)

        arc_layout = QHBoxLayout()
        self.arc_angle = QDoubleSpinBox()
        self.arc_angle.setRange(90.0, 360.0)
        self.arc_angle.setValue(360.0)
        self.arc_angle.setSingleStep(45.0)
        arc_layout.addWidget(QLabel("Arc Angle (deg):"))
        arc_layout.addWidget(self.arc_angle)
        cylinder_params_layout.addLayout(arc_layout)

        self.cylinder_params_widget.setLayout(cylinder_params_layout)

        # Add both parameter widgets to the layout
        self.shape_params_layout.addWidget(self.box_params_widget)
        self.shape_params_layout.addWidget(self.cylinder_params_widget)

        self.shape_params_group.setLayout(self.shape_params_layout)
        left_layout.addWidget(self.shape_params_group)

        # Initially hide all shape-specific parameters
        self.box_params_widget.hide()
        self.cylinder_params_widget.hide()

        # Connect signals for live preview with debouncing
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._do_update_preview)

        self.width_spin.valueChanged.connect(self._schedule_preview_update)
        self.height_spin.valueChanged.connect(self._schedule_preview_update)
        self.resize_method.currentIndexChanged.connect(self._schedule_preview_update)

        # Lithophane parameters
        param_group = QGroupBox("Lithophane Parameters")
        param_layout = QVBoxLayout()
        
        # Base height
        base_layout = QHBoxLayout()
        self.base_height = QDoubleSpinBox()
        self.base_height.setRange(0.1, 10.0)
        self.base_height.setValue(DEFAULT_BASE_HEIGHT)
        self.base_height.setSingleStep(0.1)
        base_layout.addWidget(QLabel("Base Height (mm):"))
        base_layout.addWidget(self.base_height)
        param_layout.addLayout(base_layout)

        # Max height
        max_layout = QHBoxLayout()
        self.max_height = QDoubleSpinBox()
        self.max_height.setRange(0.1, 20.0)
        self.max_height.setValue(DEFAULT_MAX_HEIGHT)
        self.max_height.setSingleStep(0.1)
        max_layout.addWidget(QLabel("Max Height (mm):"))
        max_layout.addWidget(self.max_height)
        param_layout.addLayout(max_layout)

        # Min thickness
        min_layout = QHBoxLayout()
        self.min_thickness = QDoubleSpinBox()
        self.min_thickness.setRange(0.1, 5.0)
        self.min_thickness.setValue(DEFAULT_MIN_THICKNESS)
        self.min_thickness.setSingleStep(0.1)
        min_layout.addWidget(QLabel("Min Thickness (mm):"))
        min_layout.addWidget(self.min_thickness)
        param_layout.addLayout(min_layout)

        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)

        # Convert and Export buttons
        self.convert_button = QPushButton("Convert to Lithophane")
        self.convert_button.clicked.connect(self.convert_to_lithophane)
        self.export_button = QPushButton("Export STL")
        self.export_button.clicked.connect(self.export_stl)
        
        left_layout.addWidget(self.convert_button)
        left_layout.addWidget(self.export_button)
        left_layout.addStretch()

        self.main_layout.addWidget(left_panel, stretch=1)

    def create_right_panel(self):
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Image preview
        preview_group = QGroupBox("Image Preview")
        preview_layout = QVBoxLayout()
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(PREVIEW_SIZE, PREVIEW_SIZE)
        self.preview_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)
        right_layout.addWidget(preview_group)

        # 3D viewport
        viewport_group = QGroupBox("3D Preview")
        viewport_layout = QVBoxLayout()
        self.vtk_widget = QVTKRenderWindowInteractor()
        viewport_layout.addWidget(self.vtk_widget)
        viewport_group.setLayout(viewport_layout)
        right_layout.addWidget(viewport_group)

        self.main_layout.addWidget(right_panel, stretch=2)

    def setup_vtk(self):
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        # Set the interactor style
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        # Set up the camera
        self.renderer.ResetCamera()
        self.renderer.SetBackground(0.2, 0.2, 0.2)

        # Initialize the interactor and start the event loop
        self.interactor.Initialize()
        self.interactor.Start()

        self.stl_actor = None

    def load_image(self) -> None:
        """Load an image file selected by the user."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)")

        if file_path:
            self.current_image_path = file_path
            self.image_label.setText(os.path.basename(file_path))
            logger.info(f"Loaded image: {file_path}")

            # Show processed preview
            self._schedule_preview_update()

            # Enable convert button
            self.convert_button.setEnabled(True)
            # Disable export button until new conversion
            self.export_button.setEnabled(False)

    def update_image_preview(self, image_path: str) -> None:
        """Update the preview label with the given image."""
        try:
            with Image.open(image_path) as img:
                # Create a copy for thumbnail to avoid modifying original
                img_copy = img.copy()
                img_copy.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.LANCZOS)

                # Convert PIL image to QPixmap
                if img_copy.mode == "RGB":
                    qim = QImage(img_copy.tobytes("raw", "RGB"), img_copy.size[0], img_copy.size[1], QImage.Format_RGB888)
                else:  # Grayscale
                    qim = QImage(img_copy.tobytes("raw", "L"), img_copy.size[0], img_copy.size[1], img_copy.size[0], QImage.Format_Grayscale8)

                pixmap = QPixmap.fromImage(qim)
                self.preview_label.setPixmap(pixmap)
        except Exception as e:
            logger.error(f"Failed to update image preview: {e}")
            self.preview_label.setText("Preview unavailable")

    def _schedule_preview_update(self) -> None:
        """Schedule a preview update with debouncing."""
        if self.preview_timer:
            self.preview_timer.stop()
            self.preview_timer.start(PREVIEW_DEBOUNCE_MS)

    def on_shape_changed(self) -> None:
        """Handle shape type selection change."""
        shape_index = self.shape_type.currentIndex()

        # Hide all shape-specific parameter widgets
        self.box_params_widget.hide()
        self.cylinder_params_widget.hide()

        # Show the appropriate parameter widget and update shape type
        if shape_index == 0:  # Flat
            self.current_shape_type = 'flat'
        elif shape_index == 1:  # Box
            self.current_shape_type = 'box'
            self.box_params_widget.show()
        elif shape_index == 2:  # Cylinder
            self.current_shape_type = 'cylinder'
            self.cylinder_params_widget.show()

        logger.info(f"Shape type changed to: {self.current_shape_type}")

    def _do_update_preview(self) -> None:
        """Actually perform the preview update (called after debounce timer)."""
        if not self.current_image_path:
            return

        try:
            # Convert to grayscale (use the returned path!)
            gray_path = rgb2gray(self.current_image_path, self.temp_dir, invert=False)

            # Resize image (use the returned path!)
            resize_type = self.resize_method.currentIndex() + 1
            ratio = (self.width_spin.value(), self.height_spin.value())
            resized_preview_path = resize_image(gray_path, self.temp_dir, ratio, resize_type)

            self.update_image_preview(resized_preview_path)

        except Exception as e:
            logger.error(f"Error updating processed preview: {e}")
            # Revert to original image preview
            if self.current_image_path:
                self.update_image_preview(self.current_image_path)
            else:
                self.preview_label.setText("No image loaded")

    def convert_to_lithophane(self) -> None:
        """Convert the current image to a lithophane STL file."""
        if not self.current_image_path:
            return

        # Create progress dialog
        progress = QProgressDialog("Converting to lithophane...", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        try:
            # Convert to grayscale (use the returned path!)
            gray_path = rgb2gray(self.current_image_path, self.temp_dir, invert=False)

            # Resize image (use the returned path!)
            resize_type = self.resize_method.currentIndex() + 1
            ratio = (self.width_spin.value(), self.height_spin.value())
            resized_path = resize_image(gray_path, self.temp_dir, ratio, resize_type)
            self.processed_image_path = resized_path

            # Update preview with processed image
            self.update_image_preview(resized_path)

            # Prepare shape-specific parameters
            shape_params = {}
            if self.current_shape_type == 'box':
                shape_params['box_width'] = self.box_width.value()
                shape_params['box_depth'] = self.box_depth.value()
            elif self.current_shape_type == 'cylinder':
                shape_params['radius'] = self.cylinder_radius.value()
                shape_params['arc_angle'] = self.arc_angle.value()
                shape_params['add_caps'] = True

            # Convert to lithophane
            self.stl_path = os.path.join(self.temp_dir, "lithophane.stl")
            grayscale_to_lithophane(
                resized_path,
                self.stl_path,
                base_height=self.base_height.value(),
                max_height=self.max_height.value(),
                min_thickness=self.min_thickness.value(),
                shape_type=self.current_shape_type,
                **shape_params
            )

            # Update 3D preview
            self.load_stl_to_viewer(self.stl_path)

            # Enable export button
            self.export_button.setEnabled(True)

            progress.close()
            logger.info("Lithophane conversion completed successfully")

        except Exception as e:
            progress.close()
            error_msg = f"Failed to convert to lithophane:\n{str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(self, "Conversion Error", error_msg)

    def load_stl_to_viewer(self, stl_path: str) -> None:
        """Load and display an STL file in the 3D viewer."""
        try:
            # Remove existing STL actor if it exists
            if self.stl_actor:
                self.renderer.RemoveActor(self.stl_actor)

            # Create STL reader
            reader = vtk.vtkSTLReader()
            reader.SetFileName(stl_path)

            # Create mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            # Create actor
            self.stl_actor = vtk.vtkActor()
            self.stl_actor.SetMapper(mapper)

            # Add actor to renderer
            self.renderer.AddActor(self.stl_actor)

            # Reset camera to fit the new model
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

            logger.info(f"Loaded STL to viewer: {stl_path}")
        except Exception as e:
            logger.error(f"Failed to load STL to viewer: {e}")
            raise

    def export_stl(self) -> None:
        """Export the generated STL file to a user-selected location."""
        if not self.stl_path:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save STL File", "", "STL Files (*.stl)")

        if save_path:
            if not save_path.lower().endswith('.stl'):
                save_path += '.stl'

            try:
                # Use shutil for efficient file copying
                shutil.copyfile(self.stl_path, save_path)
                logger.info(f"STL file exported to: {save_path}")
                QMessageBox.information(self, "Export Successful",
                                      f"STL file saved successfully to:\n{save_path}")
            except Exception as e:
                error_msg = f"Failed to save STL file:\n{str(e)}"
                logger.error(error_msg)
                QMessageBox.critical(self, "Export Error", error_msg)

    def closeEvent(self, event) -> None:
        """Clean up resources when closing the application."""
        # Cleanup temporary files
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {e}")

        # Properly close the VTK widget
        try:
            self.vtk_widget.Finalize()
        except Exception as e:
            logger.warning(f"Error finalizing VTK widget: {e}")

        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LithophaneGenerator()
    window.show()
    sys.exit(app.exec_())