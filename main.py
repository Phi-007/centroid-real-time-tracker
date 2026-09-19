import cv2
import time
from pathlib import Path

from indexer import Indexer
from tracker import Tracker
from visualize import Visualize
from yolov10_detector import YOLOv10Warpper

# ---------------------------
# Example main loop (process frames in a folder)
# ---------------------------
def main(frames_folder: Path,
         output_video: Path,
         cell_size: int = 64,
         max_distance: float = 80.0,
         max_missed: int = 5):
    # get frame list (assumes frames are image files sorted by name)
    frames = sorted([p for p in frames_folder.iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg')])
    if len(frames) == 0:
        raise RuntimeError("No frames found in folder: " + str(frames_folder))

    # read first frame to get image dimensions
    test_img = cv2.imread(str(frames[0]))
    height, width = test_img.shape[:2]

    indexer = Indexer(image_width=width, image_height=height, cell_size=cell_size)
    tracker = Tracker(indexer=indexer, max_distance=max_distance, max_missed=max_missed)
    visualizer = Visualize(width, height, cell_size, output_video, fps=25)

    detector = YOLOv10Warpper("/path/to/yolov10_model.pt")  # Adjust the path to your YOLOv10 model
    
    for frame_idx, fpath in enumerate(frames, start=0):
        if frame_idx == 300:
            break
        img = cv2.imread(str(fpath))
        start_t = time.time()
        detections = detector.detect(img)  # list of (x,y,w,h,class_id)
        # Update tracker with detections
        dets_for_tracker = [(x,y,w,h,cid) for (x,y,w,h,cid) in detections]
        dets_info, tracks_vis = tracker.update(dets_for_tracker, frame_idx)

        dets_vis = []
        for (x,y,w,h,cid) in detections:
            dets_vis.append({'box':(x,y,w,h), 'class_id':cid})

        frame_out = visualizer.draw(img, dets_vis, tracks_vis)
        visualizer.write_frame(frame_out)
        end_t = time.time()
        print(f"Frame {frame_idx} processed: {len(detections)} dets, {len(tracks_vis)} tracks, elapsed {end_t-start_t:.3f}s")

    visualizer.release()
    print("Output video saved to:", output_video)


if __name__ == "__main__":
    # adjust these paths
    frames_folder = Path("/path/to/frames_folder")  # Folder containing extracted frames     
    output_video = Path("output.mp4")

    main(frames_folder, output_video,
         cell_size=64, max_distance=100.0, max_missed=5)