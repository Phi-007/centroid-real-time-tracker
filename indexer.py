import math 
from typing import List, Tuple

# ---------------------------
# Indexer: grid helpers
# ---------------------------
class Indexer:
    def __init__(self, image_width: int, image_height: int, cell_size: int):
        self.image_width = image_width
        self.image_height = image_height
        self.cell_size = cell_size
        # number of cells in each dimension
        self.width_cells = math.ceil(image_width / cell_size)
        self.height_cells = math.ceil(image_height / cell_size)

    def point_to_cell(self, x: float, y: float) -> Tuple[int, int, int]:
        """
        Return (row, col, cell_id).
        cell_id = row * width_cells + col
        clamps to image boundary.
        """
        col = int(x // self.cell_size)
        row = int(y // self.cell_size)
        # clamp
        col = max(0, min(self.width_cells - 1, col))
        row = max(0, min(self.height_cells - 1, row))
        cell_id = row * self.width_cells + col
        return row, col, cell_id

    def cell_id_to_rowcol(self, cell_id: int) -> Tuple[int, int]:
        row = cell_id // self.width_cells
        col = cell_id % self.width_cells
        return row, col

    def neighbors8(self, cell_id: int, include_self: bool = True) -> List[int]:
        """
        Return list of neighbor cell ids (8 neighbors). If include_self True also return cell_id itself.
        Edges are handled (fewer neighbors).
        """
        r, c = self.cell_id_to_rowcol(cell_id)
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < self.height_cells and 0 <= nc < self.width_cells:
                    nid = nr * self.width_cells + nc
                    if not include_self and nid == cell_id:
                        continue
                    neighbors.append(nid)
        return neighbors