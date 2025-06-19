
from pathlib import Path
import argparse
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
from skimage import color, measure

# Custom imports
from napari_pyav._reader import FastVideoReader
from octron.sam2_octron.helpers.sam2_zarr import create_image_zarr, load_image_zarr



test_video_path = '/Users/horst/Library/CloudStorage/GoogleDrive-hobenhaus@gmail.com/My Drive/OCTRON/Project folders - in progress/chromatophores/original videos/resized/0N3A1505_resized.mp4'
cp_model = '/Users/horst/Library/CloudStorage/GoogleDrive-hobenhaus@gmail.com/My Drive/OCTRON/Project folders - in progress/chromatophores/chromatophore_cp_models/cpsam_20250618_090504_it400'

def ensure_package(pkg):
    import sys
    import subprocess
    try:
        __import__(pkg.split('[')[0])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        
        
ensure_package("pyarrow")
ensure_package("fastparquet")
ensure_package("cellpose[gui]")
from cellpose import models

##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
    
def load_video_zarr(video_path,
                    remove_previous_zarr=False,
                    ):
    """
    Load a video file and prepare associated Zarr 
    and Parquet file paths for cell mask segmentation and tracking.

    Parameters
    ----------
    video_path : pathlib.Path
        Path to the input video file.
    remove_previous_zarr : bool
        If a zarr archive is found, load or remove it?

    Returns
    -------
    zarr_path : pathlib.Path
        Path to the Zarr file for storing cell masks.
    parquet_path : pathlib.Path
        Path to the Parquet file for storing cell tracking data.
    video : FastVideoReader
        Video reader object for accessing video frames.
    mask_zarr : zarr.core.Array or similar
        Zarr array object for cell masks, either newly created or loaded from disk.
    """

    video = FastVideoReader(str(video_path))
    num_frames = video.shape[0]
    image_height = video.shape[1]
    image_width = video.shape[2]
    

    export_path = video_path.parent / f"{video_path.stem}_export"
    zarr_path = export_path / f"{video_path.stem}_cell_masks.zarr"
    parquet_path = export_path / f"{video_path.stem}_cell_tracking.parquet"

    # Checks paths / zarr 
    if not export_path.exists():
        export_path.mkdir(parents=True)
    
    if zarr_path.exists() and remove_previous_zarr:
        shutil.rmtree(zarr_path)
        print('Removed previously saved zarr archive.')
        
    if not zarr_path.exists():
        mask_zarr = create_image_zarr(zarr_path=zarr_path,
                    num_frames=num_frames,
                    image_height=image_height,
                    image_width=image_width,
                    chunk_size=250,
                    fill_value=-1,
                    dtype='int32',
                    num_ch=None,
                    verbose=True,
                    )
    else:
        print(f'Loading existing zarr mask file')
        mask_zarr, status_zarr = load_image_zarr(zarr_path, 
                                            num_frames=num_frames, 
                                            image_height=image_height, 
                                            image_width=image_width, 
                                            )
        assert status_zarr, f'Failed to load zarr archive'
    return zarr_path, parquet_path, video, mask_zarr


def process_frames(video, 
                   mask_zarr, 
                   model, 
                   parquet_path,
                   max_frames,
                   ):
    """
    Processes each frame of a video to extract and track regions 

    For each frame, the function:
      - Converts the frame to LAB color space.
      - Computes average L channel values in the four corners (for laser stimulation tracking).
      - Applies the segmentation model to extract masks and region properties.
      - Computes mean LAB values for each segmented region.
      - Tracks regions across frames by matching centroids within a distance cutoff.
      - Assigns persistent region IDs to tracked regions.
      - Relabels masks with region IDs and saves them into a Zarr archive.
      - Aggregates all region properties into a master DataFrame and saves it as a Parquet file.

    Parameters
    ----------
    video : FastVideoReader
        Video reader object for accessing video frames.
    mask_zarr : zarr.core.Array
        Zarr array object for storing relabeled masks.
    model : object
        Loaded CellPose model.
    parquet_path : str or Path
        Path to save the parquet file (master pandas dataframe) under.
    max_frames : int, optional
        Maximum number of frames to process. If None, processes all frames.

    Returns
    -------
    master_df : pandas.DataFrame
        Aggregated region properties for all frames.

    """

    dist_cutoff = 10 # Distance cutoff to recognize two  chromatophores as same

    for frame_idx in tqdm(range(video.shape[0]), unit='frame', desc='Extracting masks'):
        
        frame = video[frame_idx]
        frame_lab = color.rgb2lab(frame, 
                                illuminant='D65', 
                                observer='2'
        )
        frame_hsv = color.rgb2hsv(frame)                        
        
        l = frame_lab[:,:,0]
        a = frame_lab[:,:,1]
        b = frame_lab[:,:,2]
    
        hue = frame_hsv[:,:,0]
        sat = frame_hsv[:,:,1]
        val = frame_hsv[:,:,2]
        
        # Compute the average L channel value in 50x50 regions at the four corners
        region_size = 50
        h, w = l.shape
        corners = {
            'top_left': l[:region_size, :region_size],
            'top_right': l[:region_size, w-region_size:w],
            'bottom_left': l[h-region_size:h, :region_size],
            'bottom_right': l[h-region_size:h, w-region_size:w]
        }
        corner_means = {}
        for name, region in corners.items():
            corner_means[name] = region.mean()

        imgs = [frame]
        masks, _, _ = model.eval(
            imgs,
        )
        cell_mask = masks[0]
        #no_cells = len(np.unique(cell_mask)) - 1   # Exluding '0' (background)
        props = measure.regionprops_table(
            cell_mask, 
            properties=(
                'label',
                'area',
                'area_bbox',
                'area_convex',
                'coords',
                'centroid',
                'orientation',
                'eccentricity',
                'solidity',
                'extent',
                'major_axis_length',
                'minor_axis_length'
            )
        )
        props_df = pd.DataFrame(props)
        # Add corner mean L values to props_df
        for corner_name, mean_val in corner_means.items():
            props_df[f'corner_{corner_name}_mean_l'] = mean_val

        # Extract l,a,b values 
        mean_l = []
        mean_a = []
        mean_b = []
        mean_hues = []
        mean_sats = []
        mean_vals = []
        for coords in props['coords']:
            # coords is an array of (row, col) pairs
            coords_arr = np.array(coords)
            l_vals = l[coords_arr[:, 0], coords_arr[:, 1]]
            a_vals = a[coords_arr[:, 0], coords_arr[:, 1]]
            b_vals = b[coords_arr[:, 0], coords_arr[:, 1]]
            
            hue_vals = hue[coords_arr[:, 0], coords_arr[:, 1]]
            sat_vals = sat[coords_arr[:, 0], coords_arr[:, 1]]
            val_vals = val[coords_arr[:, 0], coords_arr[:, 1]]
            
            mean_l.append(np.mean(l_vals))
            mean_a.append(np.mean(a_vals))
            mean_b.append(np.mean(b_vals))
            mean_hues.append(np.mean(hue_vals))
            mean_sats.append(np.mean(sat_vals))
            mean_vals.append(np.mean(val_vals))
            
        props_df['mean_l'] = mean_l
        props_df['mean_a'] = mean_a
        props_df['mean_b'] = mean_b
        props_df['mean_hues'] = mean_hues
        props_df['mean_sats'] = mean_sats
        props_df['mean_vals'] = mean_vals
        
        props_df = props_df.drop(columns=['coords'])

        # Keep a global list of all seen centroids and their region_ids
        if frame_idx == 0:
            global_centroids = []
            global_region_ids = []
            master_df = props_df.copy()
            master_df['frame'] = frame_idx
            master_df['region_id'] = np.arange(len(props_df))
            next_region_id = len(props_df)
            # Store initial centroids and ids
            global_centroids.extend(props_df[['centroid-0', 'centroid-1']].values.tolist())
            global_region_ids.extend(master_df['region_id'].tolist())
        else:
            curr_centroids = props_df[['centroid-0', 'centroid-1']].values
            assigned = np.full(len(props_df), -1, dtype=int)
            centroid_dists = np.full(len(props_df), np.nan)

            dists = cdist(curr_centroids, np.array(global_centroids))
            used_global = set()
            used_curr = set()

            # assign the closest available pair under cutoff
            pairs = [
                (i, j, dists[i, j])
                for i in range(dists.shape[0])
                for j in range(dists.shape[1])
                if dists[i, j] <= dist_cutoff
            ]
            pairs.sort(key=lambda x: x[2])  # sort by distance

            for i, j, dist in pairs:
                if i in used_curr or j in used_global:
                    continue
                assigned[i] = global_region_ids[j]
                centroid_dists[i] = dist
                # Update global_centroids to the average of the old and new centroid
                old_centroid = np.array(global_centroids[j])
                new_centroid = curr_centroids[i]
                avg_centroid = (old_centroid + new_centroid) / 2
                global_centroids[j] = avg_centroid.tolist()
                used_curr.add(i)
                used_global.add(j)

            # Assign new IDs to unmatched regions
            for i in range(len(props_df)):
                if assigned[i] == -1:
                    assigned[i] = next_region_id
                    next_region_id += 1

            props_df['region_id'] = assigned
            props_df['frame'] = frame_idx
            props_df['centroid_dist'] = centroid_dists

            # Update global centroids and ids with new regions only
            for i, rid in enumerate(assigned):
                if rid >= len(global_region_ids):
                    global_centroids.append(curr_centroids[i])
                    global_region_ids.append(rid)

            master_df = pd.concat([master_df, props_df], ignore_index=True)
            
            # Take care of the cell_mask
            # Relabel cell_mask with new region_id from props_df
            relabel_mask = np.zeros_like(cell_mask, dtype=np.int32)
            for idx, row in props_df.iterrows():
                mask_label = row['label']
                region_id = row['region_id']
                relabel_mask[cell_mask == mask_label] = region_id

            # Save relabeled mask to zarr
            mask_zarr[frame_idx] = relabel_mask
        
        if max_frames is not None and frame_idx >= max_frames: 
            break
        
    
    # Safe to parquet file
    master_df.to_parquet(parquet_path)
    
    print('\nSaved masks to zarr archive and tracking details to parquet.')
    print(f'Parquet saved under: "{parquet_path.as_posix()}"')
    return master_df

def main(video_path,
         model_path,
         max_frames
         ):
    
    """
    
    Parameters
    ----------
    video_path : str or Path
        Path to the input video file to be processed.
    model_path : str or Path
        Path to the pretrained Cellpose model to use for segmentation.
    max_frames : int or None
        Maximum number of frames to process from the video. 
        If None, all frames are processed.
        
    Returns
    -------
    video : numpy.ndarray
        The loaded video data as a NumPy array.
    zarr_path : Path
        Path to the Zarr file containing processed video data.
    mask_zarr : zarr.core.Array
        Zarr array containing segmentation masks for the video frames.
    master_df : pandas.DataFrame
        DataFrame containing extracted features or results from the processed frames.
    parquet_path : Path
        Path to the Parquet file containing the results DataFrame.
    """
    
    video_path = Path(video_path)
    model_path = Path(model_path)
    assert video_path.exists(), f'Video file not found: {model_path.as_posix()}'
    assert model_path.exists(), f'Model path not found: {video_path.as_posix()}'

    print(f"Using video: {video_path.name}")
    print(f"Using Cellpose model: {model_path.name}")
    if max_frames:
        print(f"Will process up to {max_frames} frames\n")
    else:
        print("Will process all frames\n")
    
    print("Loading CellPose model")
    model = models.CellposeModel(
                    pretrained_model=model_path,
                    gpu=True,
                    )
    print("Processing video")
    zarr_path, parquet_path, video, mask_zarr = load_video_zarr(video_path,
                                                                remove_previous_zarr=True
                                                                )
    master_df = process_frames(video=video, 
                               mask_zarr=mask_zarr, 
                               model=model, 
                               parquet_path=parquet_path,
                               max_frames=max_frames
                              )

    return video, zarr_path, mask_zarr, master_df, parquet_path

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Process a video with Cellpose segmentation.")
    parser.add_argument("--video", type=str, help="Path to the input video file")
    parser.add_argument("--model", type=str, help="Path to the Cellpose model")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")

    args = parser.parse_args()

    video_path = args.video if args.video else test_video_path
    model_path = args.model if args.model else cp_model
    max_frames = args.max_frames

    

    #### START PROCESSING ##################################################################################
    main(video_path, model_path, max_frames)