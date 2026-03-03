# Thermal Image Calibration Tool

A Python GUI application for correcting thermal drift in uncooled microbolometer sensors (DJI M4T) using the Empirical Line Method.

## Why This Tool?

Uncooled microbolometer sensors in drones like the DJI Mavic 4 Thermal experience thermal drift during operation. This means the raw temperature readings change over time even for the same surface temperature. This tool corrects for drift by:

1. Measuring known reference targets (e.g. plastic calibration panels) with an IR thermometer
2. Comparing these "ground truth" temperatures to what the drone measured
3. Computing a correction formula: `T_corrected = m × T_drone + b`
4. Applying this correction to all survey images

## Features

- **Dynamic Targets**: Add as many calibration targets as needed (minimum 2)
- **Color-Coded UI**: Each target has a unique color for easy identification
- **Per-Image Ground Truth**: Enter IR gun readings separately for each calibration image
- **Auto-Updating Coefficients**: m, b, R² update automatically as you enter data
- **Regression Plot Viewer**: Visual confirmation of calibration quality
- **Temporal Interpolation**: Linearly interpolate coefficients between start/end calibration
- **Tooltip Help System**: Hover over ❓ icons for guidance
- **32-bit Float GeoTIFF Output**: Full precision scientific data
- **Detailed Processing Log**: CSV with exact transformations applied to each image

## Installation

```bash
pip install -r requirements.txt
```

## Step-by-Step Workflow

### Phase 1: Field Work (Before/After Flight)

**Before your drone flight:**
1. Place 2-4 plastic calibration targets in the scene (different colors recommended)
2. Take a thermal image of the targets with your drone
3. Immediately measure each target with your IR thermometer gun
4. Record the IR gun temperatures on paper

**After your drone flight:** (if drift correction is important)
1. Take another thermal image of the targets
2. Measure again with the IR gun
3. Record these end-of-flight temperatures

### Phase 2: Software Calibration

**Step 1: Launch the tool**
```bash
python thermal_calibration_tool.py
```

**Step 2: Load calibration images**
1. In **Tab 1 (Calibration Setup)**, click **"Load Calibration Images"**
2. Select your calibration image(s) - the ones showing the plastic targets
3. Images will be sorted by timestamp automatically
4. The listbox shows `[Start]`, `[End]`, or `[Only]` labels

**Step 3: Draw polygons for each target**

For the **first (or only) calibration image**:
1. Click the image in the listbox to select it
2. For Target T1:
   - Click the **"Draw"** button next to T1
   - Your cursor changes to a crosshair
   - Click **4 points** around the target (going around the corners)
   - The polygon closes automatically after the 4th click
   - A translucent fill and "T1" label appear
3. In the colored **GT (°C)** field, type the IR gun temperature for that target
4. Repeat for T2, T3, etc.

> **Tip:** Click **"+ Add Target"** if you have more than 2 targets

**Step 4: Check the coefficients**
- As you enter ground truth values, coefficients auto-update
- You'll see: `m = 1.0234, b = -0.45, R² = 0.9987`
- `R² > 0.99` indicates excellent calibration
- Click **"📊 View Regression Plot"** to see the fit visually

**Step 5: For multi-image calibration (recommended)**
- Click the second image in the listbox (e.g., `[End]`)
- Draw new polygons 
- Enter the **new** IR gun readings for that image
- Each image gets its own m, b coefficients

### Phase 3: Batch Processing

Switch to **Tab 2 (Batch Processing)**:

**Step 1: Set folders**
- **Survey Images Folder**: Where your drone flight images are stored
- **Output Folder**: Where calibrated images will be saved (auto-creates if missing)

**Step 2: Optional emissivity settings**
- If your targets are not perfect blackbodies, enable emissivity correction
- Default: disabled

**Step 3: Process**
1. Enter the Input and Output directories
1. Click **"Start Batch Processing"**
2. Each image is calibrated based on its timestamp:
   - Before start calibration: uses start coefficients
   - Between start/end: interpolated coefficients
   - After end calibration: uses end coefficients

### Phase 4: Verify Output

Check the output folder for:
- `*_calibrated.tif` - Your corrected thermal images (32-bit Float GeoTIFF)
- `processing_log.csv` - Detailed record with:
  - Exact m, b values applied to each image
  - Original vs calibrated temperature statistics
  - Formula applied: `T_out = 1.0234 * T_in + (-0.45)`
- `processing_summary.txt` - Human-readable summary

## Output Files

| File | Purpose |
|------|---------|
| `*_calibrated.tif` | 32-bit Float GeoTIFF with corrected temperatures |
| `processing_log.csv` | Machine-readable log for audit trail |
| `processing_summary.txt` | Human-readable summary of settings |

## Mathematical Background

### Empirical Line Method

The tool solves a linear regression between drone measurements and ground truth:

```
T_corrected = m × T_drone + b
```

Where:
- `T_drone`: Mean temperature from inner 80% of target ROI
- `T_gun`: Ground truth IR gun measurement
- `m`, `b`: Calibration coefficients

### Temporal Interpolation

For flights with calibration at start ($t_0$) and end ($t_1$):

```
m(t) = m₀ + (t - t₀)/(t₁ - t₀) × (m₁ - m₀)
b(t) = b₀ + (t - t₀)/(t₁ - t₀) × (b₁ - b₀)
```

### Emissivity Correction (Optional)

Corrects apparent temperature using Stefan-Boltzmann law:

$$T_{obj} = \sqrt[4]{\frac{T_{app}^4 - (1 - \epsilon) \cdot T_{sky}^4}{\epsilon}}$$

Where:
- `ε`: Target emissivity (default: 0.95)
- `T_sky`: Reflected sky temperature (default: -40°C)

## File Format Support

### Input
- TIFF (.tif, .tiff)
- JPEG (.jpg, .jpeg)
- PNG (.png)

### Output
- 32-bit Float GeoTIFF with LZW compression
- Preserves georeferencing if present in source