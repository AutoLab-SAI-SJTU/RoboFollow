"""Route validation for Scene3 expert collection."""
import numpy as np


def check_training_route(samples, initial_z, route_mode):
    """Check the carried object's sampled route, including the avoided object."""
    if route_mode not in ("via", "around"):
        raise ValueError(f"Unknown route mode: {route_mode}")
    if not np.isfinite(initial_z):
        return {"passed": False, "reason": "invalid initial height", "samples": len(samples)}
    reached = False
    crossed_avoided = False
    for sample in samples:
        source, correct, wrong = (np.asarray(sample[key], dtype=float) for key in ("source", "correct", "wrong"))
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in (source, correct, wrong)):
            return {"passed": False, "reason": "invalid route sample", "samples": len(samples)}
        if source[2] - initial_z < 0.01:
            continue
        dx = abs(source[0] - correct[0])
        dy = abs(source[1] - correct[1])
        wrong_dy = abs(source[1] - wrong[1])
        reached |= bool(dx < 0.01 and dy < 0.06 and wrong_dy > 0.04 and dy < wrong_dy)
        crossed_avoided |= bool(abs(source[0] - wrong[0]) < 0.01 and wrong_dy < 0.04)
    passed = reached and (route_mode != "around" or not crossed_avoided)
    reason = "ok" if passed else "required route not reached"
    if route_mode == "around" and crossed_avoided:
        reason = "passed over the object that should be avoided"
    return {"passed": passed, "reason": reason, "samples": len(samples),
            "reached_route": reached, "crossed_avoided": crossed_avoided}
