import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QPushButton, QFileDialog, QLabel, QSpinBox, QDoubleSpinBox, 
    QComboBox, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np
from PIL import Image, ImageOps
from stl import mesh
import tempfile
from datetime import datetime

def rgb2gray(image_path, save_path, invert=False):
    """
    Converts image from RGB to Grayscale.
    
    Args:
        image_path (str): Input image file path
        save_path (str): Directory to save the processed image
        invert (bool): Whether to invert the image
    
    Returns:
        str: Path of the converted image
    """
    # Open image
    with Image.open(image_path) as img:
        # Convert RGB to grayscale
        gray_img = img.convert('L')
        
        # Invert if requested
        if invert:
            gray_img = ImageOps.invert(gray_img)
        
        # Generate new filename
        new_filename = f"grayscale.png"
        new_path = os.path.join(save_path, new_filename)

        # Save the processed image
        gray_img.save(new_path)
    
    return new_path

def resize_image(image_path, save_path, ratio, resize_type=1):
    """
    Resizes image into target ratio with selected resizing method.
    
    Args:
        image_path (str): Input image file path
        save_path (str): Directory to save the processed image
        ratio (tuple): Target width and height
        resize_type (int): Resize method (1=Crop, 2=Pad, 3=Stretch)
    
    Returns:
        str: Path of the resized image
    """
    # Open image
    with Image.open(image_path) as img:
        # Get original image size
        orig_width, orig_height = img.size
        # Get target image size(ratio)
        target_width, target_height = ratio

        if resize_type == 1:  # Cropping (Centered)
            # Calculate scaling factors
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

            resized_img = img.crop((left, top, right, bottom))
            resized_img = resized_img.resize((target_width, target_height), Image.LANCZOS)
        
        elif resize_type == 2:  # Padding (Centered)
            # Calculate scaling factors
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

            resized_img = ImageOps.expand(img, (pad_left, pad_top, pad_right, pad_bottom))
            resized_img = resized_img.resize((target_width, target_height), Image.LANCZOS)

        elif resize_type == 3:  # Stretching
            resized_img = img.resize((target_width, target_height), Image.LANCZOS)
        
        else:
            raise ValueError("Invalid resize type. Must be 1, 2, or 3.")

        # Generate new filename
        directory, filename = os.path.split(image_path)
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

    return new_path

def grayscale_to_lithophane(input_filepath, output_filepath=None, 
                           base_height=2.0, max_height=4.0, 
                           min_thickness=0.5):
    """
    Convert a grayscale image to a lithophane STL file.
    
    Parameters:
    -----------
    input_filepath : str
        Path to input grayscale image file
    output_filepath : str, optional
        Path for output STL file. If None, uses input path with .stl extension
    base_height : float, optional
        Base height of the lithophane in mm (default: 2.0)
    max_height : float, optional
        Maximum additional height from base in mm (default: 4.0)
    min_thickness : float, optional
        Minimum thickness of thinnest parts in mm (default: 0.5)
        
    Returns:
    --------
    str
        Path to the generated STL file
    """
    # Verify input file exists
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"Input file not found: {input_filepath}")
        
    # Set output filepath if not provided
    if output_filepath is None:
        output_filepath = os.path.splitext(input_filepath)[0] + '.stl'
    
    # Read and verify grayscale image
    img = Image.open(input_filepath)
    if img.mode != 'L':
        img = img.convert('L')
    
    # Convert image to numpy array and normalize to [0, 1]
    img_array = np.array(img).astype(float) / 255.0
    
    # Get image dimensions
    height, width = img_array.shape
    
    # Create vertices and faces for the top surface
    top_vertices = []
    top_faces = []
    vertex_count = 0
    
    # Generate vertices for the top surface
    for y in range(height):
        for x in range(width):
            z = base_height + (1 - img_array[y, x]) * max_height + min_thickness
            top_vertices.append([x, y, z])
    
    # Generate faces for the top surface
    for y in range(height - 1):
        for x in range(width - 1):
            # Get vertex indices for current quad
            v1 = y * width + x
            v2 = v1 + 1
            v3 = (y + 1) * width + x
            v4 = v3 + 1
            
            # Create two triangles
            top_faces.append([v1, v2, v3])
            top_faces.append([v2, v4, v3])
    
    top_vertices = np.array(top_vertices)
    top_faces = np.array(top_faces)
    
    # Create bottom surface vertices (at z=0)
    bottom_vertices = np.array([[x, y, 0] for y in range(height) for x in range(width)])
    
    # Create bottom surface faces (inverted orientation compared to top)
    bottom_faces = []
    for y in range(height - 1):
        for x in range(width - 1):
            v1 = y * width + x
            v2 = v1 + 1
            v3 = (y + 1) * width + x
            v4 = v3 + 1
            
            bottom_faces.append([v1, v3, v2])
            bottom_faces.append([v2, v3, v4])
    
    bottom_faces = np.array(bottom_faces)
    
    # Create side walls
    side_faces = []
    
    # Front and back walls
    for x in range(width - 1):
        # Front wall
        v1 = x
        v2 = x + 1
        v3 = x + len(top_vertices)
        v4 = v3 + 1
        side_faces.extend([[v1, v3, v2], [v2, v3, v4]])
        
        # Back wall
        v1 = x + (height - 1) * width
        v2 = v1 + 1
        v3 = v1 + len(top_vertices)
        v4 = v3 + 1
        side_faces.extend([[v1, v2, v3], [v2, v4, v3]])
    
    # Left and right walls
    for y in range(height - 1):
        # Left wall
        v1 = y * width
        v2 = (y + 1) * width
        v3 = v1 + len(top_vertices)
        v4 = v2 + len(top_vertices)
        side_faces.extend([[v1, v3, v2], [v2, v3, v4]])
        
        # Right wall
        v1 = (y + 1) * width - 1
        v2 = (y + 2) * width - 1
        v3 = v1 + len(top_vertices)
        v4 = v2 + len(top_vertices)
        side_faces.extend([[v1, v2, v3], [v2, v4, v3]])
    
    side_faces = np.array(side_faces)
    
    # Combine all vertices and adjust faces indices
    all_vertices = np.vstack([top_vertices, bottom_vertices])
    all_faces = np.vstack([
        top_faces,
        bottom_faces + len(top_vertices),
        side_faces
    ])
    
    # Create the final mesh
    lithophane = mesh.Mesh(np.zeros(len(all_faces), dtype=mesh.Mesh.dtype))
    
    # Add vertices for each face
    for i in range(len(all_faces)):
        for j in range(3):
            lithophane.vectors[i][j] = all_vertices[all_faces[i][j]]
    
    # Save the mesh
    lithophane.save(output_filepath)
    
    return output_filepath

class LithophaneGenerator(QMainWindow):
    def __init__(self, parent=None):
        super(LithophaneGenerator, self).__init__(parent)
        self.setWindowTitle("Lithophane Generator")
        self.resize(1200, 800)

        # Initialize variables
        self.current_image_path = None
        self.processed_image_path = None
        self.stl_path = None
        
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
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create temporary directory: {str(e)}")
            self.temp_dir = None

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
        self.width_spin.setRange(1, 1000)
        self.width_spin.setValue(100)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 1000)
        self.height_spin.setValue(100)
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

        # Lithophane parameters
        param_group = QGroupBox("Lithophane Parameters")
        param_layout = QVBoxLayout()
        
        # Base height
        base_layout = QHBoxLayout()
        self.base_height = QDoubleSpinBox()
        self.base_height.setRange(0.1, 10.0)
        self.base_height.setValue(2.0)
        self.base_height.setSingleStep(0.1)
        base_layout.addWidget(QLabel("Base Height (mm):"))
        base_layout.addWidget(self.base_height)
        param_layout.addLayout(base_layout)

        # Max height
        max_layout = QHBoxLayout()
        self.max_height = QDoubleSpinBox()
        self.max_height.setRange(0.1, 20.0)
        self.max_height.setValue(4.0)
        self.max_height.setSingleStep(0.1)
        max_layout.addWidget(QLabel("Max Height (mm):"))
        max_layout.addWidget(self.max_height)
        param_layout.addLayout(max_layout)

        # Min thickness
        min_layout = QHBoxLayout()
        self.min_thickness = QDoubleSpinBox()
        self.min_thickness.setRange(0.1, 5.0)
        self.min_thickness.setValue(0.5)
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
        self.preview_label.setMinimumSize(300, 300)
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

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)")
        
        if file_path:
            self.current_image_path = file_path
            self.image_label.setText(os.path.basename(file_path))
            
            # Show original image in preview
            self.update_image_preview(file_path)
            
            # Enable convert button
            self.convert_button.setEnabled(True)
            # Disable export button until new conversion
            self.export_button.setEnabled(False)

    def update_image_preview(self, image_path):
        # Load and resize image for preview
        img = Image.open(image_path)
        img.thumbnail((300, 300), Image.Resampling.LANCZOS)
        
        # Convert PIL image to QPixmap
        if img.mode == "RGB":
            qim = QImage(img.tobytes("raw", "RGB"), img.size[0], img.size[1], QImage.Format_RGB888)
        else:  # Grayscale
            qim = QImage(img.tobytes("raw", "L"), img.size[0], img.size[1], QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(qim)
        self.preview_label.setPixmap(pixmap)

    def convert_to_lithophane(self):
        if not self.current_image_path:
            return

        try:
            # Convert to grayscale
            gray_path = os.path.join(self.temp_dir, "grayscale.png")
            gray_img = rgb2gray(self.current_image_path, self.temp_dir, invert=False)

            # Resize image
            resize_type = self.resize_method.currentIndex() + 1
            ratio = (self.width_spin.value(), self.height_spin.value())
            resized_path = resize_image(gray_path, self.temp_dir, ratio, resize_type)
            self.processed_image_path = resized_path

            # Update preview with processed image
            self.update_image_preview(resized_path)

            # Convert to lithophane
            self.stl_path = os.path.join(self.temp_dir, "lithophane.stl")
            grayscale_to_lithophane(
                resized_path, 
                self.stl_path,
                base_height=self.base_height.value(),
                max_height=self.max_height.value(),
                min_thickness=self.min_thickness.value()
            )

            # Update 3D preview
            self.load_stl_to_viewer(self.stl_path)
            
            # Enable export button
            self.export_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def load_stl_to_viewer(self, stl_path):
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

    def export_stl(self):
        if not self.stl_path:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save STL File", "", "STL Files (*.stl)")
        
        if save_path:
            if not save_path.lower().endswith('.stl'):
                save_path += '.stl'
            
            try:
                # Copy STL file to selected location
                with open(self.stl_path, 'rb') as src, open(save_path, 'wb') as dst:
                    dst.write(src.read())
                QMessageBox.information(self, "Success", "STL file saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save STL file: {str(e)}")

    def closeEvent(self, event):
        # Cleanup temporary files
        import shutil
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")

        # Properly close the VTK widget
        self.vtk_widget.Finalize()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LithophaneGenerator()
    window.show()
    sys.exit(app.exec_())