"""
Image Calibration Tool (Thermal & Multispectral)
=================================================
Corrects sensor drift using the Empirical Line Method.
Supports thermal images (temperature calibration) and multispectral
single-band images (reflectance calibration without irradiance sensor).
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageDraw
import piexif
from datetime import datetime
import cv2
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import rasterio
from rasterio.transform import from_bounds
from scipy import stats
import io

# Try to import matplotlib for regression plot
try:
    import matplotlib
    matplotlib.use('TkAgg')  # interactive backend for Tk
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


# =============================================================================
# COLOR PALETTE FOR TARGETS
# =============================================================================

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
        if not self.targets:
            self.targets = [
                CalibrationTarget('Target 1', COLOR_PALETTE[0]),
                CalibrationTarget('Target 2', COLOR_PALETTE[1]),
            ]
    
    def add_target(self) -> CalibrationTarget:
        idx = len(self.targets)
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        target = CalibrationTarget(f'Target {idx + 1}', color)
        self.targets.append(target)
        return target
    
    def remove_target(self, index: int, min_targets: int = 2) -> bool:
        if len(self.targets) <= min_targets:
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
    sky_temperature: float = -40.0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _parse_datetime_string(dt_str: str) -> Optional[datetime]:
    """Parse a datetime string using multiple formats, returning a naive datetime."""
    if not dt_str or not dt_str.strip():
        return None
    dt_str = dt_str.strip()
    # Normalize trailing 'Z' to +00:00 for fromisoformat
    if dt_str.endswith('Z'):
        dt_str = dt_str[:-1] + '+00:00'
    # Try fromisoformat first (handles ISO8601 including timezone)
    try:
        dt = datetime.fromisoformat(dt_str)
        # Strip timezone to keep all datetimes naive & consistent
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except Exception:
        pass
    # Try common alternative formats
    for fmt in ('%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S.%f%z',
                '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f',
                '%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S',
                '%Y:%m:%d %H:%M:%S.%f'):
        try:
            dt = datetime.strptime(dt_str, fmt)
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt
        except Exception:
            continue
    return None


def extract_datetime_from_exif(image_path: str) -> Optional[datetime]:
    """Extract DateTime from image metadata.
    Tries (in order):
     1. XMP xmp:CreateDate within XMP packet boundaries (handles ISO8601)
     2. piexif DateTimeOriginal / DateTime
     3. rasterio TIFF tags (works well with thermal GeoTIFFs)
     4. Pillow _getexif DateTimeOriginal / DateTime
     5. file modification time as last resort
    All returned datetimes are timezone-naive for consistent arithmetic.
    """
    import re
    import os

    # 1) Try XMP CreateDate – only search within XMP packet to avoid
    #    false matches in binary pixel data of large thermal TIFFs.
    try:
        with open(image_path, 'rb') as f:
            content = f.read()
        # Locate XMP packet boundaries
        xmp_start = content.find(b'<?xpacket begin')
        xmp_end = content.find(b'<?xpacket end')
        if xmp_start >= 0 and xmp_end > xmp_start:
            xmp_data = content[xmp_start:xmp_end + 200]
        else:
            # No packet markers – restrict search to first 256 KB
            # (TIFF/JPEG metadata is at the beginning of the file)
            xmp_data = content[:262144]
        m = re.search(rb'xmp:CreateDate="([^"]+)"', xmp_data)
        if not m:
            m = re.search(rb'CreateDate="([^"]+)"', xmp_data)
        if m:
            dt_str = m.group(1).decode('utf-8', errors='ignore').strip()
            result = _parse_datetime_string(dt_str)
            if result:
                return result
    except Exception:
        pass

    # 2) Try piexif if available
    try:
        import piexif
        exif_dict = piexif.load(str(image_path))
        dt_bytes = None
        if 'Exif' in exif_dict and piexif.ExifIFD.DateTimeOriginal in exif_dict['Exif']:
            dt_bytes = exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal]
        elif '0th' in exif_dict and piexif.ImageIFD.DateTime in exif_dict['0th']:
            dt_bytes = exif_dict['0th'][piexif.ImageIFD.DateTime]
        if dt_bytes:
            if isinstance(dt_bytes, bytes):
                dt_str = dt_bytes.decode('utf-8', errors='ignore').strip()
            else:
                dt_str = str(dt_bytes)
            result = _parse_datetime_string(dt_str)
            if result:
                return result
    except Exception:
        pass

    # 3) Try rasterio TIFF tags (reliable for thermal GeoTIFFs)
    try:
        with rasterio.open(image_path) as src:
            tags = src.tags()
            for key in ('TIFFTAG_DATETIME', 'DateTime', 'datetime',
                        'TIFFTAG_DATETIMEORIGINAL', 'DateTimeOriginal'):
                dt_str = tags.get(key)
                if dt_str:
                    result = _parse_datetime_string(dt_str)
                    if result:
                        return result
    except Exception:
        pass

    # 4) Pillow _getexif fallback
    try:
        img = Image.open(image_path)
        exif = getattr(img, "_getexif", None)
        if exif:
            raw = exif()
            if raw:
                from PIL.ExifTags import TAGS
                for tag, val in raw.items():
                    name = TAGS.get(tag, tag)
                    if name in ('DateTimeOriginal', 'DateTime'):
                        result = _parse_datetime_string(str(val))
                        if result:
                            return result
    except Exception:
        pass

    # 5) Last resort: file modification time
    try:
        return datetime.fromtimestamp(os.path.getmtime(image_path))
    except Exception:
        return None


def load_thermal_image(image_path: str) -> np.ndarray:
    """Load thermal image as float32 array (assumes already in Celsius)."""
    try:
        with rasterio.open(image_path) as src:
            data = src.read(1).astype(np.float32)
            return data
    except:
        pass
    img = Image.open(image_path)
    return np.array(img, dtype=np.float32)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE for better visualization of thermal images."""
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)
    return enhanced


def apply_inferno_colormap(image: np.ndarray) -> np.ndarray:
    """Apply Inferno colormap to thermal image for visualization."""
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


def get_polygon_inner_region(polygon: List[Tuple[int, int]], 
                              image_shape: Tuple[int, int],
                              inner_fraction: float = 0.8) -> np.ndarray:
    """Create a mask for the inner portion of a polygon."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    if inner_fraction >= 1.0:
        return mask
    
    M = cv2.moments(pts)
    if M['m00'] == 0:
        return mask
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
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
    values = image[mask > 0]
    if len(values) == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def apply_emissivity_correction(T_app: np.ndarray, 
                                 emissivity: float,
                                 T_sky: float) -> np.ndarray:
    """Apply emissivity correction using Stefan-Boltzmann law."""
    T_app_K = T_app + 273.15
    T_sky_K = T_sky + 273.15
    T_obj_K4 = (T_app_K**4 - (1 - emissivity) * T_sky_K**4) / emissivity
    T_obj_K4 = np.maximum(T_obj_K4, 0)
    T_obj_K = np.power(T_obj_K4, 0.25)
    return T_obj_K - 273.15


def compute_linear_regression(drone_temps: List[float], 
                               ground_truths: List[float]) -> Tuple[float, float, float]:
    """Compute linear regression: T_gun = m * T_drone + b"""
    if len(drone_temps) < 2:
        return 1.0, 0.0, 0.0
    x = np.array(drone_temps)
    y = np.array(ground_truths)
    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    return slope, intercept, r_value**2


def interpolate_coefficients(timestamp: datetime,
                              event_start: CalibrationEvent,
                              event_end: Optional[CalibrationEvent]) -> Tuple[float, float]:
    """Interpolate m and b coefficients based on timestamp."""
    if event_end is None:
        return event_start.slope_m, event_start.intercept_b
    
    total_duration = (event_end.timestamp - event_start.timestamp).total_seconds()
    if total_duration <= 0:
        return event_start.slope_m, event_start.intercept_b
    
    elapsed = (timestamp - event_start.timestamp).total_seconds()
    fraction = max(0, min(1, elapsed / total_duration))
    
    m_interp = event_start.slope_m + fraction * (event_end.slope_m - event_start.slope_m)
    b_interp = event_start.intercept_b + fraction * (event_end.intercept_b - event_start.intercept_b)
    return m_interp, b_interp


def save_calibrated_geotiff(image: np.ndarray,
                             output_path: str,
                             source_path: Optional[str] = None) -> None:
    """Save calibrated image as 32-bit Float GeoTIFF."""
    height, width = image.shape[:2]
    transform = None
    crs = None
    
    if source_path:
        try:
            with rasterio.open(source_path) as src:
                transform = src.transform
                crs = src.crs
        except:
            pass
    
    if transform is None:
        transform = from_bounds(0, 0, width, height, width, height)
    
    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=height, width=width, count=1,
        dtype=np.float32, crs=crs, transform=transform, compress='LZW'
    ) as dst:
        dst.write(image.astype(np.float32), 1)


def hex_to_rgba(hex_color: str, alpha: int = 25) -> Tuple[int, int, int, int]:
    """Convert hex color to RGBA tuple with alpha."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (r, g, b, alpha)


def lighten_color(hex_color: str) -> str:
    """Lighten a hex color for use as background."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    factor = 0.6
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f'#{r:02x}{g:02x}{b:02x}'


# =============================================================================
# TOOLTIP WIDGET
# =============================================================================

class ToolTip:
    """Simple tooltip widget."""
    
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)
        
    def show(self, event=None):
        if self.tipwindow:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                        background="#FFFFDD", relief=tk.SOLID, borderwidth=1,
                        font=("TkDefaultFont", 9), wraplength=300)
        label.pack()
        
    def hide(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


# =============================================================================
# TARGET ROW WIDGET
# =============================================================================

class TargetRowWidget:
    """A single row in the target list with color-coded entry fields."""
    
    def __init__(self, parent: tk.Frame, index: int, color: str, 
                 on_draw: callable, on_clear: callable, on_remove: callable,
                 on_gt_change: callable):
        self.index = index
        self.color = color
        self.on_gt_change = on_gt_change
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.X, padx=2, pady=2)
        self.frame.columnconfigure(5, weight=1)  # stats column stretches
        
        # Color indicator
        self.color_label = tk.Label(self.frame, bg=color, width=2, relief='raised')
        self.color_label.grid(row=0, column=0, padx=2, sticky='w')
        
        # Target name label
        self.name_label = ttk.Label(self.frame, text=f"T{index+1}", width=3)
        self.name_label.grid(row=0, column=1, padx=2, sticky='w')
        
        # Buttons sub-frame keeps Draw+Clear together
        btn_frame = ttk.Frame(self.frame)
        btn_frame.grid(row=0, column=2, padx=2, sticky='w')
        self.draw_btn = ttk.Button(btn_frame, text="Draw", width=5,
                                   command=lambda: on_draw(self.index))
        self.draw_btn.pack(side=tk.LEFT, padx=1)
        self.clear_btn = ttk.Button(btn_frame, text="Clear", width=5,
                                    command=lambda: on_clear(self.index))
        self.clear_btn.pack(side=tk.LEFT, padx=1)
        
        # Ground truth entry with colored background
        self.gt_var = tk.StringVar()
        self.gt_var.trace_add('write', self._on_gt_modified)
        self.gt_entry = tk.Entry(self.frame, width=9, bg=lighten_color(color), 
                                 fg='black', font=('TkDefaultFont', 9, 'bold'),
                                 textvariable=self.gt_var)
        self.gt_entry.grid(row=0, column=3, padx=2, sticky='w')
        
        # Mean/Std label
        self.stats_label = ttk.Label(self.frame, text="Mean: -")
        self.stats_label.grid(row=0, column=4, padx=2, sticky='w')
        
        # Remove button
        self.remove_btn = ttk.Button(self.frame, text="×", width=2,
                                     command=lambda: on_remove(self.index))
        self.remove_btn.grid(row=0, column=5, padx=2, sticky='e')
        
    def _on_gt_modified(self, *args):
        """Called when ground truth entry is modified."""
        self.on_gt_change()
    
    def set_stats(self, mean: float, std: float, unit: str = "°C"):
        if mean != 0:
            self.stats_label.config(text=f"Mean: {mean:.2f}{unit}, σ: {std:.2f}")
        else:
            self.stats_label.config(text="Mean: -")
            
    def get_ground_truth(self) -> float:
        try:
            return float(self.gt_entry.get())
        except ValueError:
            return 0.0
            
    def set_ground_truth(self, value: float):
        self.gt_var.set(f"{value:.1f}" if value != 0 else "")
            
    def destroy(self):
        self.frame.destroy()


# =============================================================================
# REGRESSION PLOT DIALOG
# =============================================================================

class RegressionPlotDialog:
    """Dialog to show scatter of drone mean temps vs. ground truth and fitted regression.

    Expects:
      - parent: Tk parent
      - drone_temps: list of floats (means extracted from ROIs)
      - ground_truths: list of floats (IR gun measurements)
      - labels: optional list of strings for each point (e.g. target names or image ids)
    """

    def __init__(self, parent, drone_temps: List[float], ground_truths: List[float],
                 labels: Optional[List[str]] = None, title: str = "Regression: Drone vs Ground Truth",
                 xlabel: str = "Drone ROI mean temperature (\u00b0C)",
                 ylabel: str = "Ground Truth (IR gun) temperature (\u00b0C)"):
        if not HAS_MATPLOTLIB:
            from tkinter import messagebox
            messagebox.showerror("Matplotlib required",
                                 "Matplotlib is not installed. Install it to view regression plots.")
            return

        self.parent = parent
        self.drone = list(drone_temps)
        self.ground = list(ground_truths)
        self.labels = labels or [f"T{i+1}" for i in range(len(self.drone))]
        self.xlabel = xlabel
        self.ylabel = ylabel

        # Create dialog window
        self.win = tk.Toplevel(parent)
        self.win.title(title)
        self.win.geometry("740x520")
        self.win.transient(parent)
        self.win.grab_set()

        # Figure
        self.fig, self.ax = plt.subplots(figsize=(7, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Optional toolbar
        try:
            toolbar = NavigationToolbar2Tk(self.canvas, self.win)
            toolbar.update()
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
        except Exception:
            pass

        # Plot content
        self._draw_plot()

        # Buttons frame
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill='x', pady=6)
        save_btn = ttk.Button(btn_frame, text="Save Plot as PNG", command=self._save_png)
        save_btn.pack(side='right', padx=6)
        close_btn = ttk.Button(btn_frame, text="Close", command=self.win.destroy)
        close_btn.pack(side='right')

    def _draw_plot(self):
        self.ax.clear()
        if len(self.drone) == 0 or len(self.ground) == 0 or len(self.drone) != len(self.ground):
            self.ax.text(0.5, 0.5, "Insufficient or mismatched data for regression",
                         ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return

        x = np.array(self.drone)
        y = np.array(self.ground)

        # Scatter with color coding (cycle through palette)
        colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(x))]
        for xi, yi, lbl, col in zip(x, y, self.labels, colors):
            self.ax.scatter(xi, yi, color=col, edgecolor='k', s=80, alpha=0.9)
            self.ax.annotate(lbl, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=9)

        # Fit line using scipy.stats.linregress
        try:
            res = stats.linregress(x, y)
            slope, intercept, r_val = res.slope, res.intercept, res.rvalue
            x_range = np.linspace(np.min(x) - 0.5, np.max(x) + 0.5, 200)
            y_fit = slope * x_range + intercept
            self.ax.plot(x_range, y_fit, color='black', linestyle='--', linewidth=1.5, label='Fit')
            eq_text = f"y = {slope:.4f} x + {intercept:.4f}\nR² = {r_val**2:.4f}"
            # Place equation text in upper left
            self.ax.text(0.02, 0.98, eq_text, transform=self.ax.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        except Exception:
            # fallback: no fit
            pass

        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title("Drone vs Ground Truth Regression")
        self.ax.grid(alpha=0.3)
        self.ax.set_aspect('auto')
        self.fig.tight_layout()
        self.canvas.draw()

    def _save_png(self):
        from tkinter import filedialog
        out = filedialog.asksaveasfilename(parent=self.win, defaultextension=".png",
                                           filetypes=[("PNG image", "*.png")],
                                           title="Save regression plot")
        if out:
            try:
                self.fig.savefig(out, dpi=200)
            except Exception:
                pass


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class ThermalCalibrationApp:
    """Main application for thermal / multispectral image calibration."""
    
    # Mode constants
    MODE_THERMAL = "thermal"
    MODE_MSI = "msi"
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image Calibration Tool – Thermal / MSI")
        self.root.geometry("1500x950")
        
        # Sensor mode
        self.sensor_mode = tk.StringVar(value=self.MODE_THERMAL)
        
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
        
        # Target row widgets
        self.target_rows: List[TargetRowWidget] = []
        
        # Polygon selection state
        self.drawing_polygon = False
        self.polygon_points = []
        self.active_target_index = 0
        
        # Status message
        self.status_var = tk.StringVar(value="Load calibration images to begin")
        
        self._setup_ui()
        
    # -- helpers that return mode-dependent text --------------------------
    def _is_thermal(self) -> bool:
        return self.sensor_mode.get() == self.MODE_THERMAL

    def _is_single_point(self) -> bool:
        """True when MSI single-point (K = refl / DN) calibration is active."""
        return (not self._is_thermal()) and self.msi_single_point.get()

    def _unit(self) -> str:
        return "°C" if self._is_thermal() else "DN"

    def _gt_label(self) -> str:
        return "GT (°C)" if self._is_thermal() else "GT (Refl 0–1)"

    def _readout_idle(self) -> str:
        u = self._unit()
        return f"Value: - {u} | Move cursor over image"

    def _update_targets_tooltip(self):
        """Refresh the targets help tooltip for current mode."""
        if self._is_thermal():
            text = ("For each target:\n"
                    "1. Click 'Draw' to draw a 4-point polygon\n"
                    "2. Click 4 corners on the image\n"
                    "3. Enter the IR Gun temperature (°C) in the GT field\n\n"
                    "Coefficients update automatically when ≥2 complete targets.")
        else:
            text = ("For each reflectance panel:\n"
                    "1. Click 'Draw' to draw a 4-point polygon\n"
                    "2. Click 4 corners on the panel in the image\n"
                    "3. Enter the known reflectance as a fraction\n"
                    "   (0–1, e.g. 50 %  →  0.5) in the GT field\n\n"
                    "Calibrate one band at a time (load single-band images).")
        # Recreate tooltip
        try:
            self.targets_help_btn.unbind("<Enter>")
            self.targets_help_btn.unbind("<Leave>")
        except Exception:
            pass
        ToolTip(self.targets_help_btn, text)

    def _on_mode_switch(self):
        """Called when user toggles between Thermal and MSI mode."""
        is_thermal = self._is_thermal()

        # Update window title
        mode_str = "Thermal" if is_thermal else "MSI"
        self.root.title(f"Image Calibration Tool – {mode_str}")

        # Show / hide emissivity frame (thermal only)
        if is_thermal:
            self.emis_frame.pack(fill=tk.X, padx=5, pady=5)
        else:
            self.emis_frame.pack_forget()

        # Show / hide MSI sub-options
        if is_thermal:
            self.msi_options_frame.pack_forget()
        else:
            # Insert after mode frame, before Step 1
            self.msi_options_frame.pack(fill=tk.X, padx=5, pady=5,
                                        after=self.mode_frame)
        self._on_single_point_toggle()

        # Update header label in targets section
        self.gt_header_label.config(text=self._gt_label())

        # Update tooltip
        self._update_targets_tooltip()

        # Update pixel readout
        self.temp_label.config(text=self._readout_idle())

        # Update stats labels in target rows
        if self.calibration_events:
            event = self.calibration_events[self.current_event_index]
            for i, row in enumerate(self.target_rows):
                if i < len(event.targets):
                    row.set_stats(event.targets[i].drone_mean,
                                  event.targets[i].drone_std, self._unit())

        self.status_var.set(f"Switched to {mode_str} mode.")

    def _on_single_point_toggle(self):
        """Show/hide multi-target controls depending on calibration method."""
        single = self._is_single_point()

        # Keep target count consistent with selected calibration method.
        changed = self._normalize_targets_for_mode()

        if single:
            # In single-point mode only one target is needed
            self.targets_frame.config(text="Step 2: Define CRP Target")
            self.add_target_btn.pack_forget()
            self.coeff_frame.config(text="Calibration Coefficient K (auto-updated)")
            # No regression curve for a single data point
            self.plot_btn.pack_forget()
            self.plot_help_label.pack_forget()
        else:
            self.targets_frame.config(text="Step 2: Define Targets (min 2)")
            self.add_target_btn.pack(fill=tk.X, padx=5, pady=5)
            self.coeff_frame.config(text="Calibration Coefficients (auto-updated)")
            self.plot_btn.pack(side=tk.LEFT)
            self.plot_help_label.pack(side=tk.LEFT, padx=5)

        if changed:
            self._rebuild_target_rows()
            self._update_display()

        self._auto_calculate_coefficients()
        self._update_coeff_display()

    def _normalize_targets_for_mode(self) -> bool:
        """Ensure target count matches mode constraints. Returns True if modified."""
        if not self.calibration_events:
            return False

        changed = False
        if self._is_single_point():
            # Single-point MSI must have exactly one CRP target.
            for event in self.calibration_events:
                if not event.targets:
                    event.targets = [CalibrationTarget('Target 1', COLOR_PALETTE[0])]
                    changed = True
                elif len(event.targets) > 1:
                    event.targets = [event.targets[0]]
                    changed = True
        else:
            # Thermal and MSI multi-point modes require at least two targets.
            for event in self.calibration_events:
                while len(event.targets) < 2:
                    event.add_target()
                    changed = True

        if changed:
            self.active_target_index = None
        return changed

    def _setup_ui(self):
        """Setup the main UI layout."""
        # ---- top bar: status only ----
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=5, pady=2)

        self.status_label = ttk.Label(top_frame, textvariable=self.status_var,
                                      font=('TkDefaultFont', 10, 'italic'),
                                      foreground='#0066CC')
        self.status_label.pack(side=tk.LEFT, padx=5)

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
        left_container = ttk.Frame(self.tab_calibration, width=480)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_container.pack_propagate(False)
        
        # Create scrollable frame
        canvas_scroll = tk.Canvas(left_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas_scroll.yview)
        self.left_frame = ttk.Frame(canvas_scroll)
        
        self.left_frame.bind("<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")))
        
        canvas_scroll.create_window((0, 0), window=self.left_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # --- Sensor Mode (at top of left panel) ---
        self.mode_frame = ttk.LabelFrame(self.left_frame, text="Sensor Mode")
        self.mode_frame.pack(fill=tk.X, padx=5, pady=5)
        mode_row = ttk.Frame(self.mode_frame)
        mode_row.pack(fill=tk.X, padx=5, pady=4)
        ttk.Radiobutton(mode_row, text="Thermal", variable=self.sensor_mode,
                        value=self.MODE_THERMAL,
                        command=self._on_mode_switch).pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(mode_row, text="Multispectral (MSI)", variable=self.sensor_mode,
                        value=self.MODE_MSI,
                        command=self._on_mode_switch).pack(side=tk.LEFT, padx=6)

        # --- MSI sub-options (hidden by default) ---
        self.msi_single_point = tk.BooleanVar(value=True)
        self.msi_options_frame = ttk.LabelFrame(self.left_frame, text="MSI Calibration Method")
        # (not packed yet – shown only when MSI is active)
        sp_row = ttk.Frame(self.msi_options_frame)
        sp_row.pack(fill=tk.X, padx=5, pady=4)
        ttk.Radiobutton(sp_row, text="Single-Point (1 CRP)",
                        variable=self.msi_single_point, value=True,
                        command=self._on_single_point_toggle).pack(anchor=tk.W)
        ttk.Radiobutton(sp_row, text="Multi-Point ELM (≥2 targets)",
                        variable=self.msi_single_point, value=False,
                        command=self._on_single_point_toggle).pack(anchor=tk.W)
        sp_help = ttk.Label(self.msi_options_frame, text="❓", cursor="hand2",
                            foreground='#0066CC')
        sp_help.pack(anchor=tk.E, padx=5)
        ToolTip(sp_help,
                "Single-Point: K = ρ_CRP / DN_CRP – requires only one\n"
                "calibration panel. Good for agricultural drones that\n"
                "auto-correct internal offsets (b≈0).\n\n"
                "Multi-Point ELM: ρ = m·DN + b – requires ≥2 panels\n"
                "of different reflectance for a full linear fit.")
        
        # --- Load Calibration Images ---
        load_frame = ttk.LabelFrame(self.left_frame, text="Step 1: Load Calibration Images")
        load_frame.pack(fill=tk.X, padx=5, pady=5)
        
        btn_row = ttk.Frame(load_frame)
        btn_row.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(btn_row, text="Load Calibration Images", 
                   command=self._load_calibration_images).pack(side=tk.LEFT)
        
        # Help icon
        help_btn = ttk.Label(btn_row, text="❓", cursor="hand2", foreground='#0066CC')
        help_btn.pack(side=tk.LEFT, padx=5)
        ToolTip(help_btn, "Select one or more calibration images containing your reference targets.\n"
                         "Images will be sorted by timestamp automatically.\n"
                         "Each image needs its own ground truth measurements.")
        
        self.calib_listbox = tk.Listbox(load_frame, height=5, exportselection=False,
                                        font=('TkDefaultFont', 9))
        self.calib_listbox.pack(fill=tk.X, padx=5, pady=5)
        self.calib_listbox.bind('<<ListboxSelect>>', self._on_calib_select)
        
        # Hint label
        hint_label = ttk.Label(load_frame, 
            text="⚠ Click an image above to select it, then draw polygons and enter GT values.",
            font=('TkDefaultFont', 8), foreground='#666666', wraplength=380)
        hint_label.pack(padx=5, pady=2)
        
        # Current image info
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
        self.targets_frame = ttk.LabelFrame(self.left_frame, text="Step 2: Define Targets (min 2)")
        self.targets_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Header row – use a grid so columns align with the target rows
        header = ttk.Frame(self.targets_frame)
        header.pack(fill=tk.X, padx=5, pady=2)
        header.columnconfigure(5, weight=1)  # stats column stretches
        ttk.Label(header, text="Col",  width=3).grid(row=0, column=0, padx=2, sticky='w')
        ttk.Label(header, text="ID",   width=3).grid(row=0, column=1, padx=2, sticky='w')
        ttk.Label(header, text="Actions", width=11).grid(row=0, column=2, padx=2, sticky='w')
        self.gt_header_label = ttk.Label(header, text=self._gt_label(), width=9)
        self.gt_header_label.grid(row=0, column=3, padx=2, sticky='w')
        ttk.Label(header, text="Drone Stats").grid(row=0, column=4, padx=2, sticky='w')
        
        help_btn2 = ttk.Label(header, text="❓", cursor="hand2", foreground='#0066CC')
        help_btn2.grid(row=0, column=5, padx=2, sticky='e')
        self.targets_help_btn = help_btn2
        self._update_targets_tooltip()
        
        # Target rows container
        self.targets_container = ttk.Frame(self.targets_frame)
        self.targets_container.pack(fill=tk.X, padx=5, pady=5)
        
        # Add target button
        self.add_target_btn = ttk.Button(self.targets_frame, text="+ Add Target", 
                                          command=self._add_target)
        self.add_target_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # --- Coefficients Display ---
        self.coeff_frame = ttk.LabelFrame(self.left_frame, text="Calibration Coefficients (auto-updated)")
        self.coeff_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.coeff_label = ttk.Label(self.coeff_frame, text="m = -, b = -, R² = -",
                                     font=('TkDefaultFont', 10, 'bold'))
        self.coeff_label.pack(padx=5, pady=5)
        
        # View regression button
        btn_row2 = ttk.Frame(self.coeff_frame)
        btn_row2.pack(fill=tk.X, padx=5, pady=5)
        self.plot_btn = ttk.Button(btn_row2, text="📊 View Regression Plot", 
                                   command=self._show_regression_plot)
        self.plot_btn.pack(side=tk.LEFT)
        
        self.plot_help_label = ttk.Label(btn_row2, text="❓", cursor="hand2", foreground='#0066CC')
        self.plot_help_label.pack(side=tk.LEFT, padx=5)
        ToolTip(self.plot_help_label, "Opens a plot showing:\n"
                          "• Your data points (drone temp vs ground truth)\n"
                          "• The regression line fit\n"
                          "• R² value indicating fit quality\n\n"
                          "Use this to visually validate your calibration.")
        
        # --- Emissivity Settings (thermal only) ---
        self.emis_frame = ttk.LabelFrame(self.left_frame, text="Emissivity Correction (Optional)")
        self.emis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.emis_enabled = tk.BooleanVar(value=False)
        emis_check = ttk.Checkbutton(self.emis_frame, text="Enable Emissivity Correction",
                        variable=self.emis_enabled)
        emis_check.pack(anchor=tk.W, padx=5)
        
        help_btn4 = ttk.Label(self.emis_frame, text="❓", cursor="hand2", foreground='#0066CC')
        help_btn4.pack(anchor=tk.E, padx=5)
        ToolTip(help_btn4, "Corrects for non-blackbody emission using:\n"
                          "T_obj = ⁴√[(T_app⁴ - (1-ε)·T_sky⁴) / ε]\n\n"
                          "Only needed if targets have known emissivity different from 1.0")
        
        emis_row1 = ttk.Frame(self.emis_frame)
        emis_row1.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(emis_row1, text="Target ε:").pack(side=tk.LEFT)
        self.emis_value = ttk.Entry(emis_row1, width=10)
        self.emis_value.insert(0, "0.95")
        self.emis_value.pack(side=tk.LEFT, padx=5)
        
        emis_row2 = ttk.Frame(self.emis_frame)
        emis_row2.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(emis_row2, text="T_sky (°C):").pack(side=tk.LEFT)
        self.tsky_value = ttk.Entry(emis_row2, width=10)
        self.tsky_value.insert(0, "-40")
        self.tsky_value.pack(side=tk.LEFT, padx=5)
        
        # Right panel: Image canvas
        right_frame = ttk.Frame(self.tab_calibration)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas
        canvas_frame = ttk.Frame(right_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='gray20')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Pixel readout
        self.temp_label = ttk.Label(right_frame, text=self._readout_idle(),
                                    font=('TkDefaultFont', 10))
        self.temp_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.canvas.bind('<Motion>', self._on_mouse_move)
        self.canvas.bind('<Button-1>', self._on_canvas_click)
        
    def _setup_processing_tab(self):
        """Setup batch processing tab UI."""
        # Settings frame
        settings_frame = ttk.LabelFrame(self.tab_processing, text="Processing Settings")
        settings_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Survey images – folder OR file selection
        folder_row1 = ttk.Frame(settings_frame)
        folder_row1.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(folder_row1, text="Survey Images:").pack(side=tk.LEFT)
        self.survey_entry = ttk.Entry(folder_row1, width=55)
        self.survey_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(folder_row1, text="Folder",
                   command=self._browse_survey_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(folder_row1, text="Files…",
                   command=self._browse_survey_files).pack(side=tk.LEFT, padx=2)
        
        # Filename filter
        filter_row = ttk.Frame(settings_frame)
        filter_row.pack(fill=tk.X, padx=5, pady=3)
        ttk.Label(filter_row, text="Filename filter:").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        self.filter_entry = ttk.Entry(filter_row, width=20, textvariable=self.filter_var)
        self.filter_entry.pack(side=tk.LEFT, padx=5)
        filter_help = ttk.Label(filter_row, text="❓", cursor="hand2", foreground='#0066CC')
        filter_help.pack(side=tk.LEFT)
        ToolTip(filter_help, "Only process files whose name contains this text.\n"
                             "Useful for MSI band selection, e.g. '_1' for band 1.\n"
                             "Leave empty to process all images in the folder.")
        
        # Output folder
        folder_row2 = ttk.Frame(settings_frame)
        folder_row2.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(folder_row2, text="Output Folder:").pack(side=tk.LEFT)
        self.output_entry = ttk.Entry(folder_row2, width=60)
        self.output_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(folder_row2, text="Browse", 
                   command=self._browse_output_folder).pack(side=tk.LEFT)
        
        # Explicit file list (hidden storage for Files… selection)
        self._selected_survey_files: List[str] = []
        
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
        
        log_scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        log_scrollbar.place(relx=1.0, rely=0, relheight=1.0, anchor='ne')
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        
    # =========================================================================
    # POLYGON DRAWING
    # =========================================================================
    
    def _on_canvas_click(self, event):
        """Handle canvas click for polygon drawing."""
        if not self.drawing_polygon or self.current_image is None:
            return
            
        # Add point
        x, y = event.x, event.y
        self.polygon_points.append((x, y))
        
        event_data = self.calibration_events[self.current_event_index]
        target = event_data.targets[self.active_target_index]
        color = target.color
        tag = f"polygon_{self.active_target_index}"
        
        # Draw point
        r = 5
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline='white', 
                               width=2, tags=tag)
        
        # Draw line to previous point
        if len(self.polygon_points) > 1:
            x1, y1 = self.polygon_points[-2]
            self.canvas.create_line(x1, y1, x, y, fill=color, width=2, tags=tag)
        
        # Complete polygon on 4th point
        if len(self.polygon_points) == 4:
            self._complete_polygon_drawing()
            
    def _complete_polygon_drawing(self):
        """Complete the polygon and update data."""
        self.drawing_polygon = False
        self.canvas.config(cursor='')
        
        event_data = self.calibration_events[self.current_event_index]
        target = event_data.targets[self.active_target_index]
        color = target.color
        tag = f"polygon_{self.active_target_index}"
        
        # Draw closing line
        x1, y1 = self.polygon_points[-1]
        x2, y2 = self.polygon_points[0]
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2, tags=tag)
        
        # Convert to image coordinates
        img_points = [(int(x / self.image_scale), int(y / self.image_scale)) 
                      for x, y in self.polygon_points]
        target.polygon = img_points
        
        # Extract statistics
        mean_val, std_val = extract_roi_statistics(
            self.current_image, img_points, inner_fraction=0.8
        )
        target.drone_mean = mean_val
        target.drone_std = std_val
        
        # Update UI
        if self.active_target_index < len(self.target_rows):
            self.target_rows[self.active_target_index].set_stats(mean_val, std_val, self._unit())
        
        # Redraw display to show filled polygon with label
        self._update_display()
        
        u = self._unit()
        self.status_var.set(f"Polygon T{self.active_target_index+1} complete. "
                           f"Mean: {mean_val:.2f}{u}. Enter ground truth value.")
        
        # Auto-calculate coefficients
        self._auto_calculate_coefficients()
        
        self.polygon_points = []
        
    # =========================================================================
    # TARGET MANAGEMENT
    # =========================================================================
    
    def _rebuild_target_rows(self):
        """Rebuild the target row widgets based on current event's targets."""
        for row in self.target_rows:
            row.destroy()
        self.target_rows.clear()
        
        if not self.calibration_events:
            return
            
        event = self.calibration_events[self.current_event_index]
        
        for i, target in enumerate(event.targets):
            row = TargetRowWidget(
                self.targets_container,
                index=i,
                color=target.color,
                on_draw=self._start_polygon_selection,
                on_clear=self._clear_polygon,
                on_remove=self._remove_target,
                on_gt_change=self._on_gt_change
            )
            row.set_ground_truth(target.ground_truth)
            row.set_stats(target.drone_mean, target.drone_std, self._unit())
            self.target_rows.append(row)
            
    def _add_target(self):
        """Add a new target to the current calibration event."""
        if not self.calibration_events:
            self.status_var.set("Please load calibration images first.")
            return

        if self._is_single_point():
            self.status_var.set("Single-point MSI uses exactly one CRP target.")
            return
            
        event = self.calibration_events[self.current_event_index]
        
        if len(event.targets) >= len(COLOR_PALETTE):
            self.status_var.set(f"Maximum of {len(COLOR_PALETTE)} targets allowed.")
            return
            
        event.add_target()
        self._rebuild_target_rows()
        self.status_var.set(f"Added Target {len(event.targets)}. Draw polygon and enter GT.")
        
    def _remove_target(self, index: int):
        """Remove a target from the current calibration event."""
        if not self.calibration_events:
            return
            
        event = self.calibration_events[self.current_event_index]
        
        min_targets = 1 if self._is_single_point() else 2
        if len(event.targets) <= min_targets:
            self.status_var.set(f"Minimum of {min_targets} target(s) required.")
            return
            
        event.remove_target(index, min_targets=min_targets)
        self._rebuild_target_rows()
        self._update_display()
        self._auto_calculate_coefficients()
        
    def _start_polygon_selection(self, target_index: int):
        """Start polygon selection for a specific target."""
        if not self.calibration_events or self.current_image is None:
            self.status_var.set("Please load and select a calibration image first.")
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
        
        # Start drawing mode
        self.drawing_polygon = True
        self.polygon_points = []
        self.canvas.config(cursor='crosshair')
        
        self.status_var.set(f"Drawing T{target_index+1}: Click 4 points on the image to define polygon.")
        
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
                
            self._auto_calculate_coefficients()
            self.status_var.set(f"Cleared T{target_index+1} polygon.")
            
    def _on_gt_change(self):
        """Called when ground truth entry is modified."""
        if not self.calibration_events:
            return
        self._save_current_ground_truths()
        self._auto_calculate_coefficients()
        
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
        
        self.calibration_events = []
        
        for fp in filepaths:
            timestamp = extract_datetime_from_exif(fp)
            event = CalibrationEvent(
                image_path=fp,
                timestamp=timestamp or datetime.now()
            )
            self.calibration_events.append(event)

        # Apply current mode constraints to newly loaded events.
        self._normalize_targets_for_mode()
        
        self.calibration_events.sort(key=lambda e: e.timestamp)
        
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
        
        if self.calibration_events:
            self.calib_listbox.selection_set(0)
            self._load_image(0)
            
        self._update_summary()
        self.status_var.set(f"Loaded {len(self.calibration_events)} calibration image(s). "
                           "Click an image to select it, then draw targets.")
            
    def _on_calib_select(self, event):
        """Handle calibration image selection."""
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
        
        name = os.path.basename(event.image_path)
        time_str = event.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        self.current_image_label.config(
            text=f"▶ Image {index+1}/{len(self.calibration_events)}: {name}\n{time_str}")
        
        self.current_image = load_thermal_image(event.image_path)
        self._rebuild_target_rows()
        self._update_coeff_display()
        self._update_display()
        
        self.status_var.set(f"Selected image {index+1}. Draw polygons and enter GT for THIS image.")
        
    def _update_coeff_display(self):
        """Update the coefficients display for current event."""
        if not self.calibration_events:
            if self._is_single_point():
                self.coeff_label.config(text="K = - (define CRP target)")
            else:
                self.coeff_label.config(text="m = -, b = -, R² = -")
            return
            
        event = self.calibration_events[self.current_event_index]
        
        if self._is_single_point():
            if event.r_squared > 0:
                self.coeff_label.config(text=f"K = {event.slope_m:.8f}")
            else:
                self.coeff_label.config(text="K = - (define CRP target)")
        elif event.r_squared > 0:
            self.coeff_label.config(
                text=f"m = {event.slope_m:.4f}, b = {event.intercept_b:.4f}, R² = {event.r_squared:.4f}"
            )
        else:
            self.coeff_label.config(text="m = -, b = -, R² = - (need ≥2 targets with GT)")
        
    def _draw_filled_polygon(self, index: int, points: List[Tuple[int, int]], color: str):
        """Draw a filled polygon with translucent color and label."""
        if len(points) != 4:
            return
            
        tag = f"polygon_{index}"
        
        # Scale points to display coordinates
        scaled_points = [(int(x * self.image_scale), int(y * self.image_scale)) 
                         for x, y in points]
        
        # Create translucent fill using PIL
        # We'll overlay this on the canvas using a PhotoImage
        flat_points = [coord for point in scaled_points for coord in point]
        
        # Draw polygon outline
        self.canvas.create_polygon(flat_points, fill='', outline=color, width=3, 
                                   tags=tag, stipple='gray25')
        
        # Draw a semi-transparent fill (using stipple pattern as approximation)
        self.canvas.create_polygon(flat_points, fill=color, outline='', 
                                   stipple='gray12', tags=tag)
        
        # Draw corner points
        r = 5
        for x, y in scaled_points:
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline='white',
                                   width=2, tags=tag)
        
        # Draw target label in center
        cx = sum(p[0] for p in scaled_points) // 4
        cy = sum(p[1] for p in scaled_points) // 4
        
        # Background rectangle for label
        self.canvas.create_rectangle(cx-12, cy-10, cx+12, cy+10, 
                                    fill='white', outline=color, width=2, tags=tag)
        self.canvas.create_text(cx, cy, text=f"T{index+1}", fill=color, 
                               font=('TkDefaultFont', 10, 'bold'), tags=tag)
        
    def _update_display(self):
        """Update the image display based on visualization mode."""
        if self.current_image is None:
            return
        
        if self.viz_mode.get() == "inferno":
            self.display_image = apply_inferno_colormap(self.current_image)
        else:
            enhanced = apply_clahe(self.current_image)
            self.display_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
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
        
        pil_img = Image.fromarray(resized)
        self.photo = ImageTk.PhotoImage(pil_img)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Redraw all polygons for current event
        if self.calibration_events:
            event = self.calibration_events[self.current_event_index]
            for i, target in enumerate(event.targets):
                if len(target.polygon) == 4:
                    self._draw_filled_polygon(i, target.polygon, target.color)
        
    def _on_mouse_move(self, event):
        """Display value at cursor position."""
        if self.current_image is None:
            return
            
        x = int(event.x / self.image_scale)
        y = int(event.y / self.image_scale)
        
        h, w = self.current_image.shape[:2]
        u = self._unit()
        
        if 0 <= x < w and 0 <= y < h:
            val = self.current_image[y, x]
            self.temp_label.config(text=f"Value: {val:.2f} {u} | Pixel: ({x}, {y})")
        else:
            self.temp_label.config(text=self._readout_idle())
            
    def _auto_calculate_coefficients(self):
        """Automatically calculate coefficients when enough data is available."""
        if not self.calibration_events:
            return
            
        event = self.calibration_events[self.current_event_index]
        
        drone_vals = []
        ground_truths = []
        
        for target in event.targets:
            if target.drone_mean != 0 and target.ground_truth != 0:
                drone_vals.append(target.drone_mean)
                ground_truths.append(target.ground_truth)
        
        if self._is_single_point():
            # K = ρ_CRP / DN_CRP  (slope=K, intercept=0)
            if len(drone_vals) >= 1:
                K = ground_truths[0] / drone_vals[0]
                event.slope_m = K
                event.intercept_b = 0.0
                event.r_squared = 1.0  # perfect by definition
                self.status_var.set(f"K = {K:.8f}  (ρ={ground_truths[0]}, DN={drone_vals[0]:.2f})")
            else:
                event.slope_m = 1.0
                event.intercept_b = 0.0
                event.r_squared = 0.0
        else:
            if len(drone_vals) >= 2:
                m, b, r2 = compute_linear_regression(drone_vals, ground_truths)
                event.slope_m = m
                event.intercept_b = b
                event.r_squared = r2
                self.status_var.set(f"Coefficients updated: m={m:.4f}, b={b:.4f}, R²={r2:.4f}")
            else:
                event.slope_m = 1.0
                event.intercept_b = 0.0
                event.r_squared = 0.0
        
        self._update_coeff_display()
        self._update_summary()
        
    def _show_regression_plot(self):
        """Show the regression plot dialog."""
        if not self.calibration_events:
            self.status_var.set("No calibration data to plot.")
            return
            
        event = self.calibration_events[self.current_event_index]
        
        # Check if we have data
        valid_targets = sum(1 for t in event.targets if t.drone_mean != 0 and t.ground_truth != 0)
        if valid_targets < 2:
            self.status_var.set("Need at least 2 complete targets to show regression plot.")
            return
            
        drone_temps = [t.drone_mean for t in event.targets if t.is_complete()]
        ground_truths = [t.ground_truth for t in event.targets if t.is_complete()]
        labels = [t.name for t in event.targets if t.is_complete()]
        
        if len(drone_temps) < 2:
            messagebox.showwarning("Not Enough Data", 
                                 "At least two complete targets (with polygon and ground truth) are needed to plot regression.")
            return
            
        RegressionPlotDialog(self.root, drone_temps, ground_truths, labels,
                             title=f"Regression for {Path(event.image_path).name}",
                             xlabel="Drone ROI mean (°C)" if self._is_thermal() else "Drone ROI mean DN",
                             ylabel="Ground Truth (°C)" if self._is_thermal() else "Ground Truth Reflectance (0–1)")
        
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
                    if self._is_single_point():
                        self.summary_text.insert(tk.END,
                            f"  ✓ K = {event.slope_m:.8f}\n")
                    else:
                        self.summary_text.insert(tk.END, 
                            f"  ✓ m = {event.slope_m:.4f}, b = {event.intercept_b:.4f}, R² = {event.r_squared:.4f}\n")
                else:
                    self.summary_text.insert(tk.END, "  ✗ Coefficients: NOT READY\n")
                self.summary_text.insert(tk.END, "\n")
                
            self.summary_text.insert(tk.END, "-" * 60 + "\n")
            if self._is_single_point():
                self.summary_text.insert(tk.END, "Method: Single-Point  ρ_pixel = K × DN_pixel\n")
            elif len(self.calibration_events) == 1:
                self.summary_text.insert(tk.END, "Mode: Single calibration (constant m, b)\n")
            else:
                self.summary_text.insert(tk.END, "Mode: Temporal interpolation (m, b vary by timestamp)\n")
                
        self.summary_text.config(state=tk.DISABLED)
        
    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================
    
    def _browse_survey_folder(self):
        folder = filedialog.askdirectory(title="Select Survey Images Folder")
        if folder:
            self._selected_survey_files.clear()
            self.survey_entry.delete(0, tk.END)
            self.survey_entry.insert(0, folder)

    def _browse_survey_files(self):
        """Let user pick individual images instead of a whole folder."""
        filepaths = filedialog.askopenfilenames(
            title="Select Survey Images",
            filetypes=[
                ("Image files", "*.tif *.tiff *.jpg *.jpeg *.png"),
                ("All files", "*.*")
            ]
        )
        if filepaths:
            self._selected_survey_files = list(filepaths)
            self.survey_entry.delete(0, tk.END)
            # Show filenames so the user can verify the selection
            names = [os.path.basename(p) for p in filepaths]
            display = "; ".join(names)
            self.survey_entry.insert(0, display)
            self.survey_entry.xview_moveto(0)  # scroll to start

    def _browse_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, folder)
            
    def _log(self, message: str):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def _start_batch_processing(self):
        """Start batch processing of survey images."""
        self._save_current_ground_truths()
        
        if not self.calibration_events:
            self.status_var.set("Error: No calibration events loaded.")
            return
            
        calibrated_events = [e for e in self.calibration_events if e.r_squared > 0]
        if not calibrated_events:
            self.status_var.set("Error: No calibration coefficients calculated.")
            return
        
        # --- Collect survey images from folder or explicit file list ---
        extensions = {'.tif', '.tiff', '.jpg', '.jpeg', '.png'}
        
        if self._selected_survey_files:
            # User picked specific files via "Files…" button
            survey_images = {Path(p) for p in self._selected_survey_files
                            if Path(p).suffix.lower() in extensions}
            # Derive a default output folder from first file's parent
            default_out_base = Path(self._selected_survey_files[0]).parent
        else:
            survey_folder = self.survey_entry.get()
            if not survey_folder or not os.path.isdir(survey_folder):
                self.status_var.set("Error: Invalid survey images folder.")
                return
            survey_images = set()
            for f in Path(survey_folder).iterdir():
                if f.is_file() and f.suffix.lower() in extensions:
                    survey_images.add(f)
            default_out_base = Path(survey_folder)
        
        # Apply filename filter
        name_filter = self.filter_var.get().strip()
        if name_filter:
            survey_images = {p for p in survey_images if name_filter in p.name}
        
        output_folder = self.output_entry.get()
        if not output_folder:
            output_folder = str(default_out_base / "calibrated")
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, output_folder)
            
        os.makedirs(output_folder, exist_ok=True)
        
        # Emissivity only applies in thermal mode
        if self._is_thermal():
            self.emissivity_settings.enabled = self.emis_enabled.get()
            try:
                self.emissivity_settings.target_emissivity = float(self.emis_value.get())
                self.emissivity_settings.sky_temperature = float(self.tsky_value.get())
            except ValueError:
                pass
        else:
            self.emissivity_settings.enabled = False

        # Remove calibration images
        calib_paths = {Path(e.image_path).resolve() for e in self.calibration_events}
        survey_images = [p for p in survey_images if p.resolve() not in calib_paths]
        survey_images = sorted(survey_images, key=lambda p: p.name)
        
        if not survey_images:
            self.status_var.set("Error: No survey images found.")
            return
        
        mode_str = "Thermal" if self._is_thermal() else "MSI"
        single_pt = self._is_single_point()
        self._log(f"Mode: {mode_str}" + (" (Single-Point K)" if single_pt else ""))
        if name_filter:
            self._log(f"Filename filter: '{name_filter}'")
        self._log(f"Found {len(survey_images)} survey images to process.")
        self._log(f"Using {len(calibrated_events)} calibration event(s).")
        if single_pt:
            self._log(f"Calibration method: Single-Point  K = ρ_CRP / DN_CRP")
        else:
            self._log(f"Calibration mode: {'Interpolation' if len(calibrated_events) > 1 else 'Single'}")
        
        if self.emissivity_settings.enabled:
            self._log(f"Emissivity correction: ENABLED\n"
                     f"  Target ε={self.emissivity_settings.target_emissivity}, "
                     f"T_sky={self.emissivity_settings.sky_temperature}°C")
        
        event_start = calibrated_events[0]
        event_end = calibrated_events[-1] if len(calibrated_events) > 1 else None
        
        # Detailed processing log for transparency
        log_data = []
        total = len(survey_images)
        
        for i, img_path in enumerate(survey_images):
            try:
                filename = os.path.basename(img_path)
                self._log(f"Processing: {filename}")
                
                timestamp = extract_datetime_from_exif(str(img_path)) or datetime.now()
                self._log(f"  Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                m, b = interpolate_coefficients(timestamp, event_start, event_end)
                
                img_data = load_thermal_image(str(img_path))
                original_mean = float(np.mean(img_data))
                original_min = float(np.min(img_data))
                original_max = float(np.max(img_data))
                
                if self.emissivity_settings.enabled:
                    img_data = apply_emissivity_correction(
                        img_data,
                        self.emissivity_settings.target_emissivity,
                        self.emissivity_settings.sky_temperature
                    )
                
                # Apply calibration
                calibrated = m * img_data + b
                
                calibrated_mean = float(np.mean(calibrated))
                calibrated_min = float(np.min(calibrated))
                calibrated_max = float(np.max(calibrated))
                
                stem = Path(filename).stem
                output_name = f"{stem}_calibrated.tif"
                output_path = os.path.join(output_folder, output_name)
                
                save_calibrated_geotiff(calibrated, output_path, str(img_path))
                
                # Detailed log entry
                u = 'celsius' if self._is_thermal() else 'reflectance'
                entry = {
                    'input_filename': filename,
                    'output_filename': output_name,
                    'timestamp_extracted': timestamp.isoformat(),
                    'emissivity_corrected': self.emissivity_settings.enabled,
                    'emissivity_value': self.emissivity_settings.target_emissivity if self.emissivity_settings.enabled else None,
                    'sky_temp_celsius': self.emissivity_settings.sky_temperature if self.emissivity_settings.enabled else None,
                    f'original_mean_{u}': round(original_mean, 4),
                    f'original_min_{u}': round(original_min, 4),
                    f'original_max_{u}': round(original_max, 4),
                    f'calibrated_mean_{u}': round(calibrated_mean, 4),
                    f'calibrated_min_{u}': round(calibrated_min, 4),
                    f'calibrated_max_{u}': round(calibrated_max, 4),
                    f'mean_shift_{u}': round(calibrated_mean - original_mean, 4),
                }
                if single_pt:
                    entry['K'] = round(m, 10)
                    entry['formula_applied'] = f"Refl = {m:.10f} * DN"
                else:
                    entry['slope_m'] = round(m, 6)
                    entry['intercept_b'] = round(b, 6)
                    entry['formula_applied'] = f"Out = {m:.6f} * In + {b:.6f}"
                log_data.append(entry)
                
                progress = (i + 1) / total * 100
                self.progress_var.set(progress)
                self.progress_label.config(text=f"Processing: {i+1}/{total}")
                self.root.update_idletasks()
                
            except Exception as e:
                self._log(f"ERROR processing {img_path}: {str(e)}")
                
        # Save detailed processing log
        log_df = pd.DataFrame(log_data)
        log_path = os.path.join(output_folder, "processing_log.csv")
        log_df.to_csv(log_path, index=False)
        
        # Save human-readable summary
        summary_path = os.path.join(output_folder, "processing_summary.txt")
        mode_str = "Thermal" if self._is_thermal() else "MSI (Multispectral)"
        with open(summary_path, 'w') as f:
            f.write(f"IMAGE CALIBRATION PROCESSING SUMMARY ({mode_str})\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Processing Date: {datetime.now().isoformat()}\n")
            f.write(f"Sensor Mode: {mode_str}\n")
            if single_pt:
                f.write("Calibration Method: Single-Point (K = ρ_CRP / DN_CRP)\n")
            if name_filter:
                f.write(f"Filename Filter: '{name_filter}'\n")
            f.write(f"Total Images Processed: {len(log_data)}\n\n")
            
            f.write("CALIBRATION SETTINGS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Calibration Events Used: {len(calibrated_events)}\n")
            for i, evt in enumerate(calibrated_events):
                f.write(f"  Event {i+1}: {evt.timestamp}\n")
                f.write(f"    m = {evt.slope_m:.6f}, b = {evt.intercept_b:.6f}, R² = {evt.r_squared:.4f}\n")
            
            if len(calibrated_events) > 1:
                f.write(f"\nInterpolation Mode: Linear between Start and End\n")
            else:
                f.write(f"\nCalibration Mode: Single (constant coefficients)\n")
                
            if self.emissivity_settings.enabled:
                f.write(f"\nEmissivity Correction: ENABLED\n")
                f.write(f"  Target ε = {self.emissivity_settings.target_emissivity}\n")
                f.write(f"  Sky Temp = {self.emissivity_settings.sky_temperature}°C\n")
            else:
                f.write(f"\nEmissivity Correction: DISABLED\n")
                
            f.write("\n\nTRANSFORMATION EQUATION\n")
            f.write("-" * 40 + "\n")
            if single_pt:
                f.write("Single-Point Calibration:\n")
                f.write("  Reflectance = K × DN\n")
                f.write("  where K = ρ_CRP / DN_CRP\n\n")
            elif self._is_thermal():
                f.write("For each pixel value T_drone (in Celsius):\n")
                f.write("  T_calibrated = m × T_drone + b\n\n")
            else:
                f.write("For each pixel DN value:\n")
                f.write("  Reflectance = m × DN + b\n\n")
            f.write("Where coefficients are interpolated based on image timestamp.\n")
            f.write("See processing_log.csv for exact values per image.\n")
        
        self._log(f"\nProcessing complete!")
        self._log(f"Output folder: {output_folder}")
        self._log(f"Detailed log: processing_log.csv")
        self._log(f"Summary: processing_summary.txt")
        
        # Display processing_log.csv contents in the log window
        try:
            self._log(f"\n{'=' * 70}")
            self._log("PROCESSING LOG (processing_log.csv)")
            self._log(f"{'=' * 70}")
            csv_text = log_df.to_string(index=False)
            self.log_text.insert(tk.END, csv_text + "\n")
            self.log_text.see(tk.END)
            self.root.update_idletasks()
        except Exception:
            pass
        
        self.progress_label.config(text="Complete!")
        self.status_var.set(f"Batch processing complete. {len(log_data)} images processed.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    app = ThermalCalibrationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
