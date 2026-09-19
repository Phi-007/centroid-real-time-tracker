from typing import List, Tuple
import math


class CountingSort:
    """Counting sort used as a subroutine for Radix sort."""
    @staticmethod
    def sort_by_digit(arr: List[Tuple[float, int, int]], exp: int, base: int = 10) -> List[Tuple[float, int, int]]:
        """
        Sorts arr by the digit represented by exp (in base 'base').
        Each element is a tuple (distance, det_idx, track_id).
        """
        n = len(arr)
        output = [None] * n
        count = [0] * base

        # Convert to integer representation
        for item in arr:
            index = int(item[0] // exp) % base
            count[index] += 1

        # Cumulative count
        for i in range(1, base):
            count[i] += count[i - 1]

        # Build output (stable sort)
        for i in range(n - 1, -1, -1):
            index = int(arr[i][0] // exp) % base
            output[count[index] - 1] = arr[i]
            count[index] -= 1

        return output


class RadixSort:
    """
    Radix Sort for sorting floating-point distances using Counting Sort.
    Works on list of tuples (distance, det_idx, track_id).
    """

    def __init__(self, base: int = 10, precision: int = 4):
        """
        :param base: Base used for counting sort (default 10).
        :param precision: Number of decimal digits to preserve for floats.
        """
        self.base = base
        self.precision = precision

    def _scale_to_int(self, arr: List[Tuple[float, int, int]]) -> List[Tuple[float, int, int]]:
        """Scale float distances to integers by multiplying by 10^precision."""
        scale = 10 ** self.precision
        return [(round(a[0] * scale), a[1], a[2]) for a in arr]

    def _scale_to_float(self, arr: List[Tuple[int, int, int]]) -> List[Tuple[float, int, int]]:
        """Rescale integers back to float distances."""
        scale = 10 ** self.precision
        return [(a[0] / scale, a[1], a[2]) for a in arr]

    def sort(self, arr: List[Tuple[float, int, int]]) -> List[Tuple[float, int, int]]:
        """Main radix sort logic."""
        if not arr:
            return arr

        # Scale to integers
        scaled = self._scale_to_int(arr)

        # Find max number for number of digits
        max_val = max(a[0] for a in scaled)
        exp = 1

        # Sort by each digit
        while max_val // exp > 0:
            scaled = CountingSort.sort_by_digit(scaled, exp, self.base)
            exp *= self.base

        # Convert back to float
        return self._scale_to_float(scaled)