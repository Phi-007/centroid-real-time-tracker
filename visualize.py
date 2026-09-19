import cv2
from pathlib import Path

# ---------------------------
# Visualize: draw grid, dots, labels, write video
# ---------------------------
class Visualize:
    def __init__(self, image_width: int, image_height: int, cell_size: int, output_path: Path, fps: int = 25):
        self.image_width = image_width
        self.image_height = image_height
        self.cell_size = cell_size
        self.output_path = Path(output_path)
        self.fps = fps
        self.writer = None

    def _init_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (self.image_width, self.image_height))

    def draw_grid(self, img):
        # vertical lines
        for x in range(0, self.image_width, self.cell_size):
            cv2.line(img, (x, 0), (x, self.image_height), (200,200,200), 1)
        # horizontal lines
        for y in range(0, self.image_height, self.cell_size):
            cv2.line(img, (0, y), (self.image_width, y), (200,200,200), 1)

    def draw(self, img, detections, tracks_vis):
        """
        detections: list of dicts for current detections (boxes) - for debug if needed
        tracks_vis: list returned from tracker.update containing id, cx, cy, class_id, updated flag
        """
        out = img.copy()
        self.draw_grid(out)

        # draw detections (boxes) - green rectangles
        for det in detections:
            box = det['box']
            x,y,w,h = box
            #cv2.rectangle(out, (int(x),int(y)), (int(x+w), int(y+h)), (0,255,0), 1)

        # draw tracks: centroid dot (green if updated by detector, red if predicted),
        # and above the dot the class id + track id
        for tr in tracks_vis:
            cx = int(round(tr['cx']))
            cy = int(round(tr['cy']))
            label = f"ID:{tr['id']} C:{tr['class_id']}"
            if tr['updated']:
                dot_color = (0,255,0)  # green, detection updated this frame
            else:
                dot_color = (0,0,255)  # red predicted-only
            # dot
            cv2.circle(out, (cx, cy), 4, dot_color, -1)
            # text slightly above
            cv2.putText(out, label, (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, dot_color, 1)
        return out

    def write_frame(self, frame_img):
        if self.writer is None:
            self._init_writer()
        self.writer.write(frame_img)

    def release(self):
        if self.writer:
            self.writer.release()
            self.writer = None