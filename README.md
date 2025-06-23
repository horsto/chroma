# Chroma
Chromatophore processing

see `code/`
![Chroma Demo](gifs/differentgroups.gif)


### Usage

The scripts `code/process_video.py` and `code/process_video_batch.py` process either on video or a folder of videos and save results into a new folder under `video_name_export` next to the original video file. The export folders contain 
- ... _masks.zarr: Zarr archive containing masks across frames in the resolution of the analyzed video 
- ... _cell_tracking.parquet: Parquet archive containing region specific info across frames for each region in the extracted masks

The notebook `load_data.ipynb` shows an overview of methods to load + process the exported data. It shows how to interpolate timeseries data (area mesaurements) and how to cluster chromatophores with kmeans on one loaded video (cluster n = 3).
The function `load_video_zarr` take a video_path as input and returns the zarr_path (masks)

```python
# Load data 
import pandas as pd # Use pandas for reading parquet files back in 
from process_video import load_video_zarr

video_path = 'your_video_path.mp4'
zarr_path, parquet_path, video, mask_zarr = load_video_zarr(Path(video_path),
                                                            remove_previous_zarr=False
                                                            )

export_path = parquet_path.parent
num_frames = video.shape[0]
image_height = video.shape[1]
image_width = video.shape[2]


chroma_data = pd.read_parquet(parquet_path)
```


Columns in parquet dataframe: 

- `label`: Unique identifier for each detected region *BEFORE* distance sorting (*ignore this!*)
- `frame`: Frame index (as in original video)
- `region_id`: Unique identifier for the region across frames *AFTER* sorting. Matches the label of each region in corresponding mask at this frame.
- `area`: Number of pixels within the region.
- `area_bbox`: Area of the bounding box surrounding the region.
- `area_convex`: Area of the convex hull of the region.
- `centroid-0`: Y-coordinate of the region's centroid.
- `centroid-1`: X-coordinate of the region's centroid.
- `orientation`: Angle of the region's major axis.
- `eccentricity`: Measure of how elongated the region is.
- `solidity`: Ratio of region area to its convex hull area.
- `extent`: Ratio of region area to bounding box area.
- `major_axis_length`: Length of the major axis of the region.
- `minor_axis_length`: Length of the minor axis of the region.
- `corner_top_left_mean_l`: Mean lightness (L) in the top-left corner of the frame (for laser stim. extraction).
- `corner_top_right_mean_l`: Mean lightness (L) in the top-right corner of the frame (for laser stim. extraction).
- `corner_bottom_left_meanl`: Mean lightness (L) in the bottom-left corner of the frame (for laser stim. extraction).
- `corner_bottom_right_mean_l`: Mean lightness (L) in the bottom-right corner of the frame (for laser stim. extraction).
- `mean_l`: Mean lightness (L) value of the region.
- `mean_a`: Mean green-red (a) value of the region.
- `mean_b`: Mean blue-yellow (b) value of the region.
- `mean_hues`: Mean hue value of the region.
- `mean_sats`: Mean saturation value of the region.
- `mean_vals`: Mean value (brightness) of the region.
- centroid_dist: Distance from the region's centroid the identified reference point.


#### Extract laser onset / offset indices

```python
threshold_rel_laser = 1.005 # e.g. 1.25 x median is threshold for detecting laser stim 
# chroma_data is loaded parquet dataframe
frame_data = chroma_data.groupby('frame').mean(numeric_only=True).reset_index()

# Extract all laser indices, laser pulse on and off 
# Get average and peaks 
average_of_corners = np.nanmean(np.stack([frame_data['corner_top_left_mean_l'],
                                          frame_data['corner_top_right_mean_l'],
                                          frame_data['corner_bottom_left_mean_l'],
                                          frame_data['corner_bottom_right_mean_l']]), 
                                          axis=0
                                          )
average_of_corners /= np.nanmedian(average_of_corners)
laser_indices = np.argwhere(average_of_corners>threshold_rel_laser)
print(f'Found {len(laser_indices)} laser pulses')
print(f'Laser pulses visible at indices: {laser_indices}')
# Extract laser onsets 
gap_idx = np.where(np.diff(laser_indices.squeeze()) > 1)[0]
# Onset of each pulse train is the first index after each gap (plus the very first pulse)
laser_onsets = laser_indices.squeeze()[np.insert(gap_idx + 1, 0, 0)]
print("Laser pulse train onsets at frames:", laser_onsets)
# Extract offsets (last index before each gap, plus the last pulse)
laser_offsets = laser_indices.squeeze()[np.append(gap_idx, len(laser_indices.squeeze()) - 1)]
print("Laser pulse train offsets at frames:", laser_offsets)


```

