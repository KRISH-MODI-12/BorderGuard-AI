import time
from dataclasses import dataclass

@dataclass
class BagState:
    first_seen: float | None = None
    last_seen: float | None = None
    last_center: tuple[int, int] | None = None
    missed_since: float | None = None
    alerted: bool = False

class UnattendedBagTracker:
    """Keeps the timer through short detector misses instead of restarting immediately."""
    def __init__(self):
        self.state = BagState()
        self.max_missed_seconds = 3.0
        self.movement_reset_pixels = 60

    def update(self, bag_box, now: float, threshold: int):
        if bag_box is None:
            if self.state.first_seen is not None:
                if self.state.missed_since is None:
                    self.state.missed_since = now
                if now - self.state.missed_since > self.max_missed_seconds:
                    self.reset()
            return self.status(now, threshold)

        x1, y1, x2, y2 = bag_box
        center = ((x1+x2)//2, (y1+y2)//2)
        self.state.missed_since = None
        if self.state.first_seen is None:
            self.state.first_seen = now
            self.state.last_center = center
        else:
            px, py = self.state.last_center or center
            movement = ((center[0]-px)**2 + (center[1]-py)**2) ** 0.5
            if movement > self.movement_reset_pixels:
                self.state.first_seen = now
                self.state.alerted = False
            self.state.last_center = center
        return self.status(now, threshold)

    def status(self, now, threshold):
        elapsed = 0 if self.state.first_seen is None else max(0, now-self.state.first_seen)
        return {"seconds": round(elapsed, 1), "threshold": threshold, "alert": elapsed >= threshold and self.state.first_seen is not None, "active": self.state.first_seen is not None}

    def reset(self):
        self.state = BagState()
