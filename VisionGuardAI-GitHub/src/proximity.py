"""Image-space proximity heuristic, not a physical-distance measurement."""
import math


def iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - overlap
    return overlap / union if union > 0 else 0


class ProximityMonitor:
    """Confirm one spatially consistent pair; rate-limit alerts camera-wide."""
    def __init__(self, overlap=.15, hold=1., cooldown=30., min_conf=.5, confirmations=3):
        self.overlap, self.hold, self.cooldown, self.min_conf = overlap, hold, cooldown, min_conf
        self.confirmations = confirmations
        self.pair = None
        self.since = None
        self.previous = None
        self.frames = 0
        self.last_alert = -math.inf
        self.status = 'Waiting for detections'

    def update(self, detections, now):
        # Rows: x1, y1, x2, y2, confidence, class_id. Windowsill alone
        # does not trigger: it is currently the weakest model class.
        rows = [r for r in detections if len(r) == 6 and all(math.isfinite(v) for v in r)
                and r[4] >= self.min_conf and r[2] > r[0] and r[3] > r[1]]
        pairs = []
        for child in (r[:4] for r in rows if r[5] == 0):
            for window in (r[:4] for r in rows if r[5] == 1):
                intersection = max(0, min(child[2], window[2])-max(child[0], window[0])) * max(0, min(child[3], window[3])-max(child[1], window[1]))
                fraction = intersection / ((child[2]-child[0])*(child[3]-child[1]))
                if fraction >= self.overlap:
                    pairs.append((child, window))
        # A missed frame, gap > 1 second, or changing pair resets confirmation.
        matched = next((p for p in pairs if self.pair and iou(p[0], self.pair[0]) >= .3
                        and iou(p[1], self.pair[1]) >= .3), None)
        continuous = matched is not None and self.previous is not None and 0 <= now-self.previous <= 1.
        if not pairs:
            self.pair, self.since, self.frames = None, None, 0
        elif not continuous:
            self.pair, self.since, self.frames = pairs[0], now, 1
        else:
            self.pair = matched
            self.frames += 1
        self.previous = now
        confirmed = self.pair is not None and self.frames >= 3 and now-self.since >= self.hold * self.confirmations
        alert = confirmed and now-self.last_alert >= self.cooldown
        if alert:
            self.last_alert = now
        children = [r for r in rows if r[5] == 0]
        windows = [r for r in rows if r[5] == 1]
        if not children or not windows:
            missing = ', '.join(name for name, present in [('child', children), ('window', windows)] if not present)
            self.status = f'Waiting: no {missing} above alert confidence {self.min_conf:.2f}'
        elif not pairs:
            self.status = f'Waiting: overlap below {self.overlap:.0%} of child box'
        elif alert:
            self.status = 'Proximity confirmed: alert triggered'
        elif confirmed:
            self.status = f'Cooldown: {max(0, self.cooldown-(now-self.last_alert)):.1f}s remaining'
        else:
            self.status = f'Confirming overlap: {min(self.confirmations, int((now-self.since)/self.hold))}/{self.confirmations} periods; {now-self.since:.1f}/{self.hold*self.confirmations:.1f}s'
        return alert
