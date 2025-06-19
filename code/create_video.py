import cv2
import numpy as np
from tqdm import tqdm

def generate_label_masks_and_frames(master_df, mask_zarr, video, kmeans_labels):
    """
    Yields for each valid frame index:
        - frame_idx
        - actual frame (RGB)
        - dict of {label: mask} for each kmeans label
    """
    image_height = mask_zarr.shape[1]
    image_width = mask_zarr.shape[2]
    valid_frame_indices = np.where(mask_zarr[:,0,0] > -1)[0]
    for frame_idx in valid_frame_indices:
        mask = mask_zarr[frame_idx]
        frame_df = master_df[master_df['frame'] == frame_idx]
        if frame_df.empty:
            continue
        frame = video[frame_idx]
        label_masks = {}
        for label in kmeans_labels:
            region_ids = frame_df[frame_df['kmeans_label'] == label]['region_id'].values
            out_img = np.zeros((image_height, image_width), dtype=np.uint8)
            for rid in region_ids:
                if rid == 0:
                    continue
                out_img[mask == rid] = 255
            label_masks[label] = out_img
        yield frame_idx, frame, label_masks

def create_mp4(master_df, mask_zarr, video, export_path, fps=20):
    """
    Creates MP4 videos visualizing k-means clustering 
    results for each label and exports the actual video frames.
    """
    assert export_path.exists(), f'Export path "{export_path.as_posix()}" not found'
    assert fps > 0 
    
    image_height = mask_zarr.shape[1]
    image_width = mask_zarr.shape[2]
    kmeans_labels = master_df['kmeans_label'].dropna().unique().astype(int)
    kmeans_labels.sort()

    video_out_dir = export_path / "kmeans_videos"
    video_out_dir.mkdir(exist_ok=True)

    video_writers = {}
    for label in kmeans_labels:
        out_path = video_out_dir / f"kmeans_label_{label}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writers[label] = cv2.VideoWriter(
            str(out_path), fourcc, fps, (image_width, image_height), isColor=False
        )

    frame_out_path = video_out_dir / "actual_frame.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_writer = cv2.VideoWriter(
        str(frame_out_path), fourcc, fps, (image_width, image_height)
    )

    exported_indices = []
    for frame_idx, frame, label_masks in tqdm(
        generate_label_masks_and_frames(master_df, mask_zarr, video, kmeans_labels),
        desc="Exporting kmeans videos"
    ):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_writer.write(frame_bgr)
        for label, out_img in label_masks.items():
            video_writers[label].write(out_img)
        exported_indices.append(frame_idx)

    for vw in video_writers.values():
        vw.release()
    frame_writer.release()
    print(f'Videos saved under\n{video_out_dir.as_posix()}, each containing {len(exported_indices)} frames.')