"""Small numerical helpers shared by foveation StageB workspaces."""

import math


def cosine_schedule_value(start, end, step, total_steps):
    """Interpolate two scalar targets with the DP cosine schedule."""
    total_steps = max(1, int(total_steps))
    progress = min(1.0, max(0.0, float(step) / total_steps))
    progress = 0.5 - 0.5 * math.cos(math.pi * progress)
    return float(start) + (float(end) - float(start)) * progress


def projected_dual_ascent(lambda_value, expected_mean, target, learning_rate, budget_range):
    """Update one nonnegative budget multiplier from its global target gap."""
    if budget_range <= 0.0:
        return max(0.0, float(lambda_value))
    gap = (float(expected_mean) - float(target)) / float(budget_range)
    return max(0.0, float(lambda_value) + float(learning_rate) * gap)
