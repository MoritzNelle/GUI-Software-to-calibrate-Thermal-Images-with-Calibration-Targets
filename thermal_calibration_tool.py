"""
Thermal Image Calibration Tool for DJI M4T Drone
=================================================
Corrects thermal drift in uncooled microbolometer sensors using the Empirical Line Method.

Author: SmartWheat Project
Date: 2026
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import piexif
from datetime import datetime
import cv2
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import rasterio
from rasterio.transform import from_bounds
from scipy import stats
import colorsys


# =============================================================================
# COLOR PALETTE FOR TARGETS
# =============================================================================

def generate_distinct_colors(n: int) -> List[str]:
    """Generate n visually distinct colors as hex strings."""
    colors = []
    for i in range(n):
        hue = i / n
        # Use high saturation and value for visibility
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')
    return colors


# Pre-generate a palette of 10 distinct colors
COLOR_PALETTE = [
    '#E63946',  # Red
    '#2A9D8F',  # Teal
    '#E9C46A',  # Yellow
    '#264653',  # Dark Blue
    '#F4A261',  # Orange
    '#9B5DE5',  # Purple
    '#00F5D4',  # Cyan
    '#F15BB5',  # Pink
    '#00BBF9',  # Light Blue
    '#FEE440',  # Bright Yellow
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CalibrationTarget:
    """Represents a single calibration target."""
    name: str
    color: str = '#FF0000'
    polygon: List[Tuple[int, int]] = field(default_factory=list)
    drone_mean: float = 0.0
    drone_std: float = 0.0
    ground_truth: float = 0.0
    
    def is_complete(self) -> bool:
        return len(self.polygon) == 4 and self.ground_truth != 0.0


@dataclass
class CalibrationEvent:
    """Represents a calibration event with timestamp and targets."""
    image_path: str
    timestamp: datetime
    targets: List[CalibrationTarget] = field(default_factory=list)
    slope_m: float = 1.0
    intercept_b: float = 0.0
    r_squared: float = 0.0
    
    def __post_init__(self):
        # Initialize with 2 default targets if none provided
        if not self.targets:
            self.targets = [
                CalibrationTarget('Target 1', COLOR_PALETTE[0]),
                CalibrationTarget('Target 2', COLOR_PALETTE[1]),
            ]
    
    def add_target(self) -> CalibrationTarget:
        """Add a new target with the next available color."""
        idx = len(self.targets)
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        target = CalibrationTarget(f'Target {idx + 1}', color)
        self.targets.append(target)
        return target
    
    def remove_target(self, index: int) -> bool:
        """Remove target at index. Maintain minimum of 2 targets."""
        if len(self.targets) <= 2:
            return False
        if 0 <= index < len(self.targets):
            self.targets.pop(index)
            return True
        return False


@dataclass  
class EmissivitySettings:
    """Emissivity correction settings."""
    enabled: bool = False
    target_emissivity: float = 0.95
    sky_temperature: float = -40.0  # Celsius


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_datetime_from_exif(image_path: str) -> Optional[datetime]:
    """Extract DateTimeOriginal from image EXIF data using piexif."""
    try:
        img = Image.open(image_path)
        if hasattr(img, '_getexif') and img._getexif():
            exif_data = piexif.load(img.info.get('exif', b''))
            if piexif.ExifIFD.DateTimeOriginal in exif_data.get('Exif', {}):
                dt_str = exif_data['Exif'][piexif.ExifIFD.DateTimeOriginal].decode('utf-8')
                return datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
            elif piexif.ImageIFD.DateTime in exif_data.get('0th', {}):
                dt_str = exif_data['0th'][piexif.ImageIFD.DateTime].decode('utf-8')
                return datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
    except Exception as e:
        print(f"Warning: Could not extract EXIF from {image_path}: {e}")
    
    # Fallback: use file modification time
    return datetime.fromtimestamp(os.path.getmtime(image_path))


def load_thermal_image(image_path: str) -> np.ndarray:
    """Load thermal image as float32 array (assumes already in Celsius)."""
    try:
        # Try rasterio first for TIFF files
        with rasterio.open(image_path) as src:
            data = src.read(1).astype(np.float32)
            return data
    except:
        pass
    
    # Fallback to PIL
    img = Image.open(image_path)
    return np.array(img, dtype=np.float32)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE for better visualization of thermal images."""
    # Normalize to 0-255 for CLAHE
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)
    
    return enhanced


def apply_inferno_colormap(image: np.ndarray) -> np.ndarray:
    """Apply Inferno colormap to thermal image for visualization."""
    # Normalize to 0-255
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    
    # Apply inferno colormap (COLORMAP_INFERNO = 11)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    
    # Convert BGR to RGB for PIL/Tkinter
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    return colored


def get_polygon_inner_region(polygon: List[Tuple[int, int]], 
                              image_shape: Tuple[int, int],
                              inner_fraction: float = 0.8) -> np.ndarray:
    """
    Create a mask for the inner portion of a polygon.
    Returns mask of inner 80% (default) to avoid edge contamination.
    """
    # Create full polygon mask
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    if inner_fraction >= 1.0:
        return mask
    
    # Calculate centroid
    M = cv2.moments(pts)
    if M['m00'] == 0:
        return mask
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    # Scale polygon towards centroid
    scale = np.sqrt(inner_fraction)
    inner_pts = []
    for px, py in polygon:
        nx = int(cx + (px - cx) * scale)
        ny = int(cy + (py - cy) * scale)
        inner_pts.append((nx, ny))
    
    inner_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    inner_pts = np.array(inner_pts, dtype=np.int32)
    cv2.fillPoly(inner_mask, [inner_pts], 255)
    
    return inner_mask


def extract_roi_statistics(image: np.ndarray, 
                           polygon: List[Tuple[int, int]],
                           inner_fraction: float = 0.8) -> Tuple[float, float]:
    """Extract mean and standard deviation from inner region of polygon."""
    mask = get_polygon_inner_region(polygon, image.shape, inner_fraction)
    
    # Extract values where mask is non-zero
    values = image[mask > 0]
    
    if len(values) == 0:
        return 0.0, 0.0
    
    return float(np.mean(values)), float(np.std(values))


def apply_emissivity_correction(T_app: np.ndarray, 
                                 emissivity: float,
                                 T_sky: float) -> np.ndarray:
    """
    Apply emissivity correction to apparent temperature.
    T_obj = (T_app^4 - (1-ε)·T_sky^4) / ε)^(1/4)
    
    Note: Temperatures must be in Kelvin for Stefan-Boltzmann law.
    """
    # Convert to Kelvin
    T_app_K = T_app + 273.15
    T_sky_K = T_sky + 273.15
    
    # Apply correction
    T_obj_K4 = (T_app_K**4 - (1 - emissivity) * T_sky_K**4) / emissivity
    
    # Handle negative values (numerical safety)
    T_obj_K4 = np.maximum(T_obj_K4, 0)
    
    T_obj_K = np.power(T_obj_K4, 0.25)
    
    # Convert back to Celsius
    return T_obj_K - 273.15


def compute_linear_regression(drone_temps: List[float], 
                               ground_truths: List[float]) -> Tuple[float, float, float]:
    """
    Compute linear regression: T_gun = m * T_drone + b
    Returns: (slope m, intercept b, r-squared)
    """
    if len(drone_temps) < 2:
        return 1.0, 0.0, 0.0
    
    x = np.array(drone_temps)
    y = np.array(ground_truths)
    
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    
    return slope, intercept, r_value**2


def interpolate_coefficients(timestamp: datetime,
                              event_start: CalibrationEvent,
                              event_end: Optional[CalibrationEvent]) -> Tuple[float, float]:
    """
    Interpolate m and b coefficients based on timestamp.
    If only one calibration event exists, use those coefficients directly.
    """
    if event_end is None:
        return event_start.slope_m, event_start.intercept_b
    
    # Calculate time fractions
    total_duration = (event_end.timestamp - event_start.timestamp).total_seconds()
    
    if total_duration <= 0:
        return event_start.slope_m, event_start.intercept_b
    
    elapsed = (timestamp - event_start.timestamp).total_seconds()
    fraction = max(0, min(1, elapsed / total_duration))
    
    # Linear interpolation
    m_interp = event_start.slope_m + fraction * (event_end.slope_m - event_start.slope_m)
    b_interp = event_start.intercept_b + fraction * (event_end.intercept_b - event_start.intercept_b)
    
    return m_interp, b_interp


def save_calibrated_geotiff(image: np.ndarray,
                             output_path: str,
                             source_path: Optional[str] = None) -> None:
    """Save calibrated image as 32-bit Float GeoTIFF."""
    height, width = image.shape[:2]
    
    # Try to copy georeferencing from source
    transform = None
    crs = None
    
    if source_path:
        try:
            with rasterio.open(source_path) as src:
                transform = src.transform
                crs = src.crs
        except:
            pass
    
    # Default transform if none found
    if transform is None:
        transform = from_bounds(0, 0, width, height, width, height)
    
    # Write GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        compress='LZW'
    ) as dst:
        dst.write(image.astype(np.float32), 1)


# =============================================================================
# POLYGON SELECTION WIDGET
# =============================================================================

class PolygonSelector:
    """Interactive 4-point polygon selection tool."""
    
    def __init__(self, canvas: tk.Canvas, callback=None):
        self.canvas = canvas
        self.callback = callback
        self.points = []
        self.point_ids = []
        self.line_ids = []
        self.polygon_id = None
        self.active = False
        self.color = '#00FF00'
        self.tag = ""
        
    def start_selection(self, color: str = '#00FF00', tag: str = ""):
        """Start polygon selection mode."""
        self.clear()
        self.active = True
        self.color = color
        self.tag = tag
        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas.config(cursor='crosshair')
        
    def _on_click(self, event):
        """Handle click events during selection."""
        if not self.active:
            return
            
        if len(self.points) < 4:
            # Add point
            x, y = event.x, event.y
            self.points.append((x, y))
            
            # Draw point
            r = 5
            point_id = self.canvas.create_oval(
                x-r, y-r, x+r, y+r,
                fill=self.color, outline='white', width=2,
                tags=self.tag
            )
            self.point_ids.append(point_id)
            
            # Draw line to previous point
            if len(self.points) > 1:
                x1, y1 = self.points[-2]
                line_id = self.canvas.create_line(
                    x1, y1, x, y,
                    fill=self.color, width=2,
                    tags=self.tag
                )
                self.line_ids.append(line_id)
            
            # Complete polygon on 4th point
            if len(self.points) == 4:
                self._complete_polygon()
                
    def _complete_polygon(self):
        """Complete the polygon and trigger callback."""
        # Draw closing line
        x1, y1 = self.points[-1]
        x2, y2 = self.points[0]
        line_id = self.canvas.create_line(
            x1, y1, x2, y2,
            fill=self.color, width=2,
            tags=self.tag
        )
        self.line_ids.append(line_id)
        
        # Fill polygon (semi-transparent effect via stipple)
        flat_points = [coord for point in self.points for coord in point]
        self.polygon_id = self.canvas.create_polygon(
            flat_points,
            fill='', outline=self.color, width=3,
            tags=self.tag
        )
        
        self.active = False
        self.canvas.unbind('<Button-1>')
        self.canvas.config(cursor='')
        
        if self.callback:
            self.callback(self.points.copy())
            
    def clear(self):
        """Clear all drawn elements."""
        for pid in self.point_ids:
            self.canvas.delete(pid)
        for lid in self.line_ids:
            self.canvas.delete(lid)
        if self.polygon_id:
            self.canvas.delete(self.polygon_id)
            
        self.points = []
        self.point_ids = []
        self.line_ids = []
        self.polygon_id = None
        self.active = False
        
    def get_points(self) -> List[Tuple[int, int]]:
        return self.points.copy()


# =============================================================================
# TARGET ROW WIDGET
# =============================================================================

class TargetRowWidget:
    """A single row in the target list with color-coded entry fields."""
    
    def __init__(self, parent: tk.Frame, index: int, color: str, 
                 on_draw: callable, on_clear: callable, on_remove: callable):
        self.index = index
        self.color = color
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.X, padx=2, pady=2)
        
        # Color indicator (small colored square)
        self.color_label = tk.Label(self.frame, bg=color, width=2, relief='raised')
        self.color_label.pack(side=tk.LEFT, padx=2)
        
        # Target name label
        self.name_label = ttk.Label(self.frame, text=f"T{index+1}", width=3)
        self.name_label.pack(side=tk.LEFT, padx=2)
        
        # Draw polygon button
        self.draw_btn = ttk.Button(self.frame, text="Draw", width=5,
                                   command=lambda: on_draw(self.index))
        self.draw_btn.pack(side=tk.LEFT, padx=2)
        
        # Clear polygon button  
        self.clear_btn = ttk.Button(self.frame, text="Clear", width=5,
                                    command=lambda: on_clear(self.index))
        self.clear_btn.pack(side=tk.LEFT, padx=2)
        
        # Ground truth entry with colored background
        self.gt_entry = tk.Entry(self.frame, width=8, bg=self._lighten_color(color), 
                                 fg='black', font=('TkDefaultFont', 9, 'bold'))
        self.gt_entry.pack(side=tk.LEFT, padx=2)
        
        # Mean/Std label
        self.stats_label = ttk.Label(self.frame, text="Mean: -", width=20)
        self.stats_label.pack(side=tk.LEFT, padx=2)
        
        # Remove button
        self.remove_btn = ttk.Button(self.frame, text="×", width=2,
                                     command=lambda: on_remove(self.index))
        self.remove_btn.pack(side=tk.RIGHT, padx=2)
        
    def _lighten_color(self, hex_color: str) -> str:
        """Lighten a hex color for use as background."""
        # Parse hex
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Lighten by mixing with white
        factor = 0.6
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def set_stats(self, mean: float, std: float):
        """Update the statistics label."""
        if mean != 0:
            self.stats_label.config(text=f"Mean: {mean:.2f}°C, σ: {std:.2f}")
        else:
            self.stats_label.config(text="Mean: -")
            
    def get_ground_truth(self) -> float:
        """Get the ground truth value from the entry."""
        try:
            return float(self.gt_entry.get())
        except ValueError:
            return 0.0
            
    def set_ground_truth(self, value: float):
        """Set the ground truth value in the entry."""
        self.gt_entry.delete(0, tk.END)
        if value != 0:
            self.gt_entry.insert(0, f"{value:.1f}")
            
    def destroy(self):
        """Remove this widget."""
        self.frame.destroy()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class ThermalCalibrationApp:
    """Main application for thermal image calibration."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Thermal Image Calibration Tool - DJI M4T")
        self.root.geometry("1500x950")
        
        # State variables
        self.calibration_events: List[CalibrationEvent] = []
        self.current_event_index = 0
        self.current_image: np.ndarray = None
        self.display_image: np.ndarray = None
        self.image_scale = 1.0
        self.survey_folder = ""
        self.output_folder = ""
        
        # Emissivity settings
        self.emissivity_settings = EmissivitySettings()
        
        # Target row widgets (per image)
        self.target_rows: List[TargetRowWidget] = []
        
        # Polygon selector
        self.polygon_selector = None
        self.active_target_index = 0
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the main UI layout."""
        # Main container with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Calibration Setup
        self.tab_calibration = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_calibration, text="1. Calibration Setup")
        self._setup_calibration_tab()
        
        # Tab 2: Batch Processing
        self.tab_processing = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_processing, text="2. Batch Processing")
        self._setup_processing_tab()
        
    def _setup_calibration_tab(self):
        """Setup calibration tab UI."""
        # Left panel: Controls (scrollable)
        left_container = ttk.Frame(self.tab_calibration, width=420)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_container.pack_propagate(False)
        
        # Create scrollable frame
        canvas_scroll = tk.Canvas(left_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas_scroll.yview)
        self.left_frame = ttk.Frame(canvas_scroll)
        
        self.left_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        )
        
        canvas_scroll.create_window((0, 0), window=self.left_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # --- Load Calibration Images ---
        load_frame = ttk.LabelFrame(self.left_frame, text="Load Calibration Images")
        load_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(load_frame, text="Load Calibration Images", 
                   command=self._load_calibration_images).pack(fill=tk.X, padx=5, pady=5)
        
        self.calib_listbox = tk.Listbox(load_frame, height=5, exportselection=False)
        self.calib_listbox.pack(fill=tk.X, padx=5, pady=5)
        self.calib_listbox.bind('<<ListboxSelect>>', self._on_calib_select)
        
        # Current image info label
        self.current_image_label = ttk.Label(load_frame, text="No image selected", 
                                              font=('TkDefaultFont', 9, 'bold'))
        self.current_image_label.pack(padx=5, pady=2)
        
        # --- Visualization Options ---
        viz_frame = ttk.LabelFrame(self.left_frame, text="Visualization")
        viz_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.viz_mode = tk.StringVar(value="inferno")
        ttk.Radiobutton(viz_frame, text="Inferno Colormap", 
                        variable=self.viz_mode, value="inferno",
                        command=self._update_display).pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(viz_frame, text="CLAHE Enhanced", 
                        variable=self.viz_mode, value="clahe",
                        command=self._update_display).pack(anchor=tk.W, padx=5)
        
        # --- Targets Section ---
        targets_frame = ttk.LabelFrame(self.left_frame, text="Calibration Targets (min 2)")
        targets_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Header row
        header = ttk.Frame(targets_frame)
        header.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(header, text="Color", width=5).pack(side=tk.LEFT)
        ttk.Label(header, text="ID", width=3).pack(side=tk.LEFT)
        ttk.Label(header, text="Actions", width=12).pack(side=tk.LEFT, padx=10)
        ttk.Label(header, text="GT (°C)", width=8).pack(side=tk.LEFT)
        ttk.Label(header, text="Drone Stats", width=20).pack(side=tk.LEFT)
        
        # Target rows container
        self.targets_container = ttk.Frame(targets_frame)
        self.targets_container.pack(fill=tk.X, padx=5, pady=5)
        
        # Add target button
        self.add_target_btn = ttk.Button(targets_frame, text="+ Add Target", 
                                          command=self._add_target)
        self.add_target_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # --- Calculate Button ---
        ttk.Button(self.left_frame, text="Calculate Coefficients for This Image", 
                   command=self._calculate_coefficients).pack(fill=tk.X, padx=5, pady=10)
        
        # --- Coefficients Display ---
        coeff_frame = ttk.LabelFrame(self.left_frame, text="Current Image Coefficients")
        coeff_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.coeff_label = ttk.Label(coeff_frame, text="m = -, b = -, R² = -",
                                     font=('TkDefaultFont', 10, 'bold'))
        self.coeff_label.pack(padx=5, pady=5)
        
        # --- Emissivity Settings ---
        emis_frame = ttk.LabelFrame(self.left_frame, text="Emissivity Correction")
        emis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.emis_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(emis_frame, text="Enable Emissivity Correction",
                        variable=self.emis_enabled).pack(anchor=tk.W, padx=5)
        
        emis_row1 = ttk.Frame(emis_frame)
        emis_row1.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(emis_row1, text="Target ε:").pack(side=tk.LEFT)
        self.emis_value = ttk.Entry(emis_row1, width=10)
        self.emis_value.insert(0, "0.95")
        self.emis_value.pack(side=tk.LEFT, padx=5)
        
        emis_row2 = ttk.Frame(emis_frame)
        emis_row2.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(emis_row2, text="T_sky (°C):").pack(side=tk.LEFT)
        self.tsky_value = ttk.Entry(emis_row2, width=10)
        self.tsky_value.insert(0, "-40")
        self.tsky_value.pack(side=tk.LEFT, padx=5)
        
        # Right panel: Image canvas
        right_frame = ttk.Frame(self.tab_calibration)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas with scrollbars
        canvas_frame = ttk.Frame(right_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='gray20')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Temperature readout
        self.temp_label = ttk.Label(right_frame, text="Temperature: - °C",
                                    font=('TkDefaultFont', 10))
        self.temp_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.canvas.bind('<Motion>', self._on_mouse_move)
        
        # Initialize polygon selector
        self.polygon_selector = PolygonSelector(self.canvas)
        
    def _setup_processing_tab(self):
        """Setup batch processing tab UI."""
        # Settings frame
        settings_frame = ttk.LabelFrame(self.tab_processing, text="Processing Settings")
        settings_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Survey folder
        folder_row1 = ttk.Frame(settings_frame)
        folder_row1.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(folder_row1, text="Survey Images Folder:").pack(side=tk.LEFT)
        self.survey_entry = ttk.Entry(folder_row1, width=60)
        self.survey_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(folder_row1, text="Browse", 
                   command=self._browse_survey_folder).pack(side=tk.LEFT)
        
        # Output folder
        folder_row2 = ttk.Frame(settings_frame)
        folder_row2.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(folder_row2, text="Output Folder:").pack(side=tk.LEFT)
        self.output_entry = ttk.Entry(folder_row2, width=60)
        self.output_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(folder_row2, text="Browse", 
                   command=self._browse_output_folder).pack(side=tk.LEFT)
        
        # Calibration summary
        summary_frame = ttk.LabelFrame(self.tab_processing, text="Calibration Summary")
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.summary_text = tk.Text(summary_frame, height=10, state=tk.DISABLED)
        self.summary_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Progress
        progress_frame = ttk.LabelFrame(self.tab_processing, text="Processing Progress")
        progress_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                            maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack(padx=5, pady=2)
        
        # Process button
        ttk.Button(self.tab_processing, text="Start Batch Processing", 
                   command=self._start_batch_processing).pack(pady=10)
        
        # Log display
        log_frame = ttk.LabelFrame(self.tab_processing, text="Processing Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.log_text = tk.Text(log_frame, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
    # =========================================================================
    # TARGET MANAGEMENT
    # =========================================================================
    
    def _rebuild_target_rows(self):
        """Rebuild the target row widgets based on current event's targets."""
        # Clear existing rows
        for row in self.target_rows:
            row.destroy()
        self.target_rows.clear()
        
        if not self.calibration_events:
            return
            
        event = self.calibration_events[self.current_event_index]
        
        # Create a row for each target
        for i, target in enumerate(event.targets):
            row = TargetRowWidget(
                self.targets_container,
                index=i,
                color=target.color,
                on_draw=self._start_polygon_selection,
                on_clear=self._clear_polygon,
                on_remove=self._remove_target
            )
            # Set existing values
            row.set_ground_truth(target.ground_truth)
            row.set_stats(target.drone_mean, target.drone_std)
            
            self.target_rows.append(row)
            
    def _add_target(self):
        """Add a new target to the current calibration event."""
        if not self.calibration_events:
            messagebox.showwarning("No Image", "Please load calibration images first.")
            return
            
        event = self.calibration_events[self.current_event_index]
        
        if len(event.targets) >= len(COLOR_PALETTE):
            messagebox.showwarning("Maximum Targets", 
                f"Maximum of {len(COLOR_PALETTE)} targets allowed.")
            return
            
        event.add_target()
        self._rebuild_target_rows()
        
    def _remove_target(self, index: int):
        """Remove a target from the current calibration event."""
        if not self.calibration_events:
            return
            
        event = self.calibration_events[self.current_event_index]
        
        if len(event.targets) <= 2:
            messagebox.showwarning("Minimum Targets", 
                "At least 2 targets are required for calibration.")
            return
            
        # Confirm removal
        result = messagebox.askyesno("Remove Target", 
            f"Remove Target {index + 1}? This will delete any polygon and ground truth data.")
        
        if result:
            event.remove_target(index)
            self._rebuild_target_rows()
            self._update_display()
        
    def _start_polygon_selection(self, target_index: int):
        """Start polygon selection for a specific target."""
        if not self.calibration_events or self.current_image is None:
            messagebox.showwarning("No Image", "Please load and select a calibration image first.")
            return
            
        self.active_target_index = target_index
        event = self.calibration_events[self.current_event_index]
        target = event.targets[target_index]
        
        # Clear existing polygon
        target.polygon = []
        target.drone_mean = 0.0
        target.drone_std = 0.0
        
        # Delete old polygon drawings
        self.canvas.delete(f"polygon_{target_index}")
        
        # Setup callback
        def on_complete(points):
            self._on_polygon_complete(target_index, points)
        
        self.polygon_selector.callback = on_complete
        self.polygon_selector.start_selection(target.color, f"polygon_{target_index}")
        
        messagebox.showinfo("Polygon Selection", 
            f"Click 4 points to define Target {target_index + 1} polygon.\n"
            "Points will be connected automatically.")
        
    def _clear_polygon(self, target_index: int):
        """Clear the polygon for a specific target."""
        if not self.calibration_events:
            return
            
        event = self.calibration_events[self.current_event_index]
        
        if target_index < len(event.targets):
            target = event.targets[target_index]
            target.polygon = []
            target.drone_mean = 0.0
            target.drone_std = 0.0
            
            self.canvas.delete(f"polygon_{target_index}")
            
            if target_index < len(self.target_rows):
                self.target_rows[target_index].set_stats(0, 0)
                
    def _on_polygon_complete(self, target_index: int, points: List[Tuple[int, int]]):
        """Handle completed polygon selection."""
        if not self.calibration_events:
            return
            
        # Convert display coordinates back to image coordinates
        img_points = [(int(x / self.image_scale), int(y / self.image_scale)) 
                      for x, y in points]
        
        event = self.calibration_events[self.current_event_index]
        
        if target_index < len(event.targets):
            target = event.targets[target_index]
            target.polygon = img_points
            
            # Extract statistics
            mean_val, std_val = extract_roi_statistics(
                self.current_image, img_points, inner_fraction=0.8
            )
            target.drone_mean = mean_val
            target.drone_std = std_val
            
            # Update UI
            if target_index < len(self.target_rows):
                self.target_rows[target_index].set_stats(mean_val, std_val)
                
    # =========================================================================
    # CALIBRATION IMAGE MANAGEMENT
    # =========================================================================
    
    def _load_calibration_images(self):
        """Load calibration images and sort by timestamp."""
        filepaths = filedialog.askopenfilenames(
            title="Select Calibration Images",
            filetypes=[
                ("Image files", "*.tif *.tiff *.jpg *.jpeg *.png"),
                ("TIFF files", "*.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not filepaths:
            return
        
        # Create calibration events and sort by timestamp
        self.calibration_events = []
        
        for fp in filepaths:
            timestamp = extract_datetime_from_exif(fp)
            event = CalibrationEvent(
                image_path=fp,
                timestamp=timestamp or datetime.now()
            )
            self.calibration_events.append(event)
        
        # Sort by timestamp
        self.calibration_events.sort(key=lambda e: e.timestamp)
        
        # Update listbox
        self.calib_listbox.delete(0, tk.END)
        for i, event in enumerate(self.calibration_events):
            name = os.path.basename(event.image_path)
            time_str = event.timestamp.strftime('%H:%M:%S')
            if len(self.calibration_events) == 1:
                label = "Only"
            elif i == 0:
                label = "Start"
            elif i == len(self.calibration_events) - 1:
                label = "End"
            else:
                label = f"Mid{i}"
            self.calib_listbox.insert(tk.END, f"[{label}] {time_str} - {name}")
        
        # Select first image
        if self.calibration_events:
            self.calib_listbox.selection_set(0)
            self._load_image(0)
            
        self._update_summary()
            
    def _on_calib_select(self, event):
        """Handle calibration image selection."""
        # Save current ground truths before switching
        self._save_current_ground_truths()
        
        selection = self.calib_listbox.curselection()
        if selection:
            self._load_image(selection[0])
            
    def _save_current_ground_truths(self):
        """Save ground truth values from UI to current event."""
        if not self.calibration_events or not self.target_rows:
            return
            
        event = self.calibration_events[self.current_event_index]
        
        for i, row in enumerate(self.target_rows):
            if i < len(event.targets):
                event.targets[i].ground_truth = row.get_ground_truth()
            
    def _load_image(self, index: int):
        """Load and display a calibration image."""
        self.current_event_index = index
        event = self.calibration_events[index]
        
        # Update current image label
        name = os.path.basename(event.image_path)
        time_str = event.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        self.current_image_label.config(text=f"Image {index+1}/{len(self.calibration_events)}: {name}\n{time_str}")
        
        # Load raw thermal data
        self.current_image = load_thermal_image(event.image_path)
        
        # Rebuild target rows for this image
        self._rebuild_target_rows()
        
        # Update coefficients display
        self._update_coeff_display()
        
        self._update_display()
        
    def _update_coeff_display(self):
        """Update the coefficients display for current event."""
        if not self.calibration_events:
            self.coeff_label.config(text="m = -, b = -, R² = -")
            return
            
        event = self.calibration_events[self.current_event_index]
        
        if event.r_squared > 0:
            self.coeff_label.config(
                text=f"m = {event.slope_m:.4f}, b = {event.intercept_b:.4f}, R² = {event.r_squared:.4f}"
            )
        else:
            self.coeff_label.config(text="m = -, b = -, R² = - (not calculated)")
        
    def _restore_polygons(self):
        """Restore drawn polygons for current event."""
        if not self.calibration_events:
            return
            
        event = self.calibration_events[self.current_event_index]
        
        # Redraw existing polygons
        for i, target in enumerate(event.targets):
            if len(target.polygon) == 4:
                self._draw_polygon(i, target.polygon, target.color)
                
    def _draw_polygon(self, index: int, points: List[Tuple[int, int]], color: str):
        """Draw a completed polygon on the canvas."""
        if len(points) != 4:
            return
            
        tag = f"polygon_{index}"
        
        # Scale points to display coordinates
        scaled_points = [(int(x * self.image_scale), int(y * self.image_scale)) 
                         for x, y in points]
        
        # Draw polygon outline
        flat_points = [coord for point in scaled_points for coord in point]
        self.canvas.create_polygon(flat_points, fill='', outline=color, width=3, tags=tag)
        
        # Draw corner points
        r = 5
        for x, y in scaled_points:
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline='white',
                                   width=2, tags=tag)
        
        # Draw target label
        cx = sum(p[0] for p in scaled_points) // 4
        cy = sum(p[1] for p in scaled_points) // 4
        self.canvas.create_text(cx, cy, text=f"T{index+1}", fill='white', 
                               font=('TkDefaultFont', 12, 'bold'), tags=tag)
        
    def _update_display(self):
        """Update the image display based on visualization mode."""
        if self.current_image is None:
            return
        
        # Apply visualization
        if self.viz_mode.get() == "inferno":
            self.display_image = apply_inferno_colormap(self.current_image)
        else:
            enhanced = apply_clahe(self.current_image)
            self.display_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        # Scale to fit canvas
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 100:
            canvas_w = 900
        if canvas_h < 100:
            canvas_h = 700
            
        img_h, img_w = self.display_image.shape[:2]
        
        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        self.image_scale = min(scale_w, scale_h, 1.0)
        
        new_w = int(img_w * self.image_scale)
        new_h = int(img_h * self.image_scale)
        
        resized = cv2.resize(self.display_image, (new_w, new_h), 
                            interpolation=cv2.INTER_LINEAR)
        
        # Convert to PhotoImage
        pil_img = Image.fromarray(resized)
        self.photo = ImageTk.PhotoImage(pil_img)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Restore polygons
        self._restore_polygons()
        
    def _on_mouse_move(self, event):
        """Display temperature at cursor position."""
        if self.current_image is None:
            return
            
        # Convert to image coordinates
        x = int(event.x / self.image_scale)
        y = int(event.y / self.image_scale)
        
        h, w = self.current_image.shape[:2]
        
        if 0 <= x < w and 0 <= y < h:
            temp = self.current_image[y, x]
            self.temp_label.config(text=f"Temperature: {temp:.2f} °C  |  Pixel: ({x}, {y})")
        else:
            self.temp_label.config(text="Temperature: - °C")
            
    def _calculate_coefficients(self):
        """Calculate linear regression coefficients for current calibration event."""
        if not self.calibration_events:
            messagebox.showwarning("No Image", "Please load calibration images first.")
            return
            
        # Save ground truths from UI
        self._save_current_ground_truths()
        
        event = self.calibration_events[self.current_event_index]
        
        # Collect data points
        drone_temps = []
        ground_truths = []
        
        for target in event.targets:
            if target.drone_mean != 0 and target.ground_truth != 0:
                drone_temps.append(target.drone_mean)
                ground_truths.append(target.ground_truth)
        
        if len(drone_temps) < 2:
            messagebox.showwarning("Insufficient Data", 
                "Need at least 2 targets with both ROI polygon and ground truth values.\n\n"
                "Please:\n"
                "1. Draw polygons around targets\n"
                "2. Enter IR Gun temperatures in the GT fields")
            return
        
        # Calculate regression
        m, b, r2 = compute_linear_regression(drone_temps, ground_truths)
        
        event.slope_m = m
        event.intercept_b = b
        event.r_squared = r2
        
        # Update display
        self._update_coeff_display()
        self._update_summary()
        
        messagebox.showinfo("Coefficients Calculated", 
            f"Linear regression complete for image {self.current_event_index + 1}:\n\n"
            f"Slope (m) = {m:.4f}\n"
            f"Intercept (b) = {b:.4f}\n"
            f"R² = {r2:.4f}\n\n"
            f"Equation: T_corrected = {m:.4f} × T_drone + {b:.4f}")
        
    def _update_summary(self):
        """Update calibration summary in processing tab."""
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        
        if not self.calibration_events:
            self.summary_text.insert(tk.END, "No calibration images loaded.\n")
        else:
            self.summary_text.insert(tk.END, f"Calibration Events: {len(self.calibration_events)}\n")
            self.summary_text.insert(tk.END, "=" * 60 + "\n\n")
            
            for i, event in enumerate(self.calibration_events):
                if len(self.calibration_events) == 1:
                    label = "ONLY"
                elif i == 0:
                    label = "START"
                elif i == len(self.calibration_events) - 1:
                    label = "END"
                else:
                    label = f"MID-{i}"
                    
                self.summary_text.insert(tk.END, f"[{label}] {event.timestamp}\n")
                self.summary_text.insert(tk.END, f"  File: {os.path.basename(event.image_path)}\n")
                self.summary_text.insert(tk.END, f"  Targets: {len(event.targets)}\n")
                
                if event.r_squared > 0:
                    self.summary_text.insert(tk.END, 
                        f"  m = {event.slope_m:.4f}, b = {event.intercept_b:.4f}, R² = {event.r_squared:.4f}\n")
                else:
                    self.summary_text.insert(tk.END, "  Coefficients: NOT CALCULATED\n")
                self.summary_text.insert(tk.END, "\n")
                
            self.summary_text.insert(tk.END, "-" * 60 + "\n")
            if len(self.calibration_events) == 1:
                self.summary_text.insert(tk.END, "Mode: Single calibration (constant m, b for all images)\n")
            else:
                self.summary_text.insert(tk.END, "Mode: Temporal interpolation (m, b vary by timestamp)\n")
                
        self.summary_text.config(state=tk.DISABLED)
        
    # =========================================================================
    # BATCH PROCESSING METHODS
    # =========================================================================
    
    def _browse_survey_folder(self):
        """Browse for survey images folder."""
        folder = filedialog.askdirectory(title="Select Survey Images Folder")
        if folder:
            self.survey_entry.delete(0, tk.END)
            self.survey_entry.insert(0, folder)
            
    def _browse_output_folder(self):
        """Browse for output folder."""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, folder)
            
    def _log(self, message: str):
        """Add message to processing log."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def _start_batch_processing(self):
        """Start batch processing of survey images."""
        # Save any pending ground truths
        self._save_current_ground_truths()
        
        # Validate inputs
        if not self.calibration_events:
            messagebox.showerror("Error", "No calibration events loaded.")
            return
            
        # Check if any calibration has been done
        calibrated_events = [e for e in self.calibration_events if e.r_squared > 0]
        if not calibrated_events:
            messagebox.showerror("Error", 
                "No calibration coefficients calculated.\n\n"
                "Please calculate coefficients for at least one calibration image.")
            return
            
        survey_folder = self.survey_entry.get()
        if not survey_folder or not os.path.isdir(survey_folder):
            messagebox.showerror("Error", "Invalid survey images folder.")
            return
            
        output_folder = self.output_entry.get()
        if not output_folder:
            output_folder = os.path.join(survey_folder, "calibrated")
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, output_folder)
            
        os.makedirs(output_folder, exist_ok=True)
        
        # Get emissivity settings
        self.emissivity_settings.enabled = self.emis_enabled.get()
        try:
            self.emissivity_settings.target_emissivity = float(self.emis_value.get())
            self.emissivity_settings.sky_temperature = float(self.tsky_value.get())
        except ValueError:
            pass
        
        # Find survey images
        extensions = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')
        survey_images = []
        
        for ext in extensions:
            survey_images.extend(Path(survey_folder).glob(f'*{ext}'))
            survey_images.extend(Path(survey_folder).glob(f'*{ext.upper()}'))
        
        # Remove calibration images from survey list
        calib_paths = set(os.path.abspath(e.image_path) for e in self.calibration_events)
        survey_images = [p for p in survey_images 
                         if os.path.abspath(str(p)) not in calib_paths]
        
        if not survey_images:
            messagebox.showerror("Error", "No survey images found.")
            return
            
        self._log(f"Found {len(survey_images)} survey images to process.")
        self._log(f"Using {len(calibrated_events)} calibration event(s).")
        self._log(f"Calibration mode: {'Interpolation' if len(calibrated_events) > 1 else 'Single'}")
        
        if self.emissivity_settings.enabled:
            self._log(f"Emissivity correction: ε={self.emissivity_settings.target_emissivity}, "
                     f"T_sky={self.emissivity_settings.sky_temperature}°C")
        
        # Prepare calibration events (use only those with calculated coefficients)
        event_start = calibrated_events[0]
        event_end = calibrated_events[-1] if len(calibrated_events) > 1 else None
        
        # Processing log data
        log_data = []
        
        # Process each image
        total = len(survey_images)
        
        for i, img_path in enumerate(sorted(survey_images)):
            try:
                filename = os.path.basename(img_path)
                self._log(f"Processing: {filename}")
                
                # Extract timestamp
                timestamp = extract_datetime_from_exif(str(img_path)) or datetime.now()
                
                # Get interpolated coefficients
                m, b = interpolate_coefficients(timestamp, event_start, event_end)
                
                # Load image
                img_data = load_thermal_image(str(img_path))
                
                # Apply emissivity correction if enabled
                if self.emissivity_settings.enabled:
                    img_data = apply_emissivity_correction(
                        img_data,
                        self.emissivity_settings.target_emissivity,
                        self.emissivity_settings.sky_temperature
                    )
                
                # Apply linear calibration: T_corrected = m * T_drone + b
                calibrated = m * img_data + b
                
                # Generate output filename
                stem = Path(filename).stem
                output_name = f"{stem}_calibrated.tif"
                output_path = os.path.join(output_folder, output_name)
                
                # Save as 32-bit Float GeoTIFF
                save_calibrated_geotiff(calibrated, output_path, str(img_path))
                
                # Log entry
                log_data.append({
                    'filename': filename,
                    'timestamp': timestamp.isoformat(),
                    'slope_m': m,
                    'intercept_b': b,
                    'emissivity_corrected': self.emissivity_settings.enabled,
                    'output_file': output_name
                })
                
                # Update progress
                progress = (i + 1) / total * 100
                self.progress_var.set(progress)
                self.progress_label.config(text=f"Processing: {i+1}/{total}")
                self.root.update()
                
            except Exception as e:
                self._log(f"ERROR processing {img_path}: {str(e)}")
                
        # Save processing log
        log_df = pd.DataFrame(log_data)
        log_path = os.path.join(output_folder, "processing_log.csv")
        log_df.to_csv(log_path, index=False)
        
        self._log(f"\nProcessing complete!")
        self._log(f"Output folder: {output_folder}")
        self._log(f"Processing log: {log_path}")
        
        self.progress_label.config(text="Complete!")
        
        messagebox.showinfo("Processing Complete", 
            f"Successfully processed {len(log_data)} images.\n\n"
            f"Output folder: {output_folder}\n"
            f"Processing log: processing_log.csv")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    style.theme_use('clam')
    
    app = ThermalCalibrationApp(root)
    
    root.mainloop()


if __name__ == "__main__":
    main()
