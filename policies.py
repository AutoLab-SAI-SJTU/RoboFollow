"""Model-independent adapter interface and rollout shape validation."""

import importlib
import numpy as np


def load_factory(spec, kwargs):
    module, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("Policy factory must be module:function")
    return getattr(importlib.import_module(module), name)(**kwargs)


class ValidatedPolicy:
    def __init__(self, policy):
        self.policy = policy

    def reset(self):
        return self.policy.reset()

    def set_instruction(self, instruction):
        return self.policy.set_instruction(instruction)

    def predict(self, observation):
        actions = np.asarray(self.policy.predict(observation), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) == 0:
            raise ValueError(f"Expected a nonempty [T, 14] action chunk, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Policy action contains NaN/Inf")
        return actions

    def close(self):
        if hasattr(self.policy, "close"):
            self.policy.close()


class HoldPolicy:
    """Hold the current joints for interface smoke tests; not a trained policy."""
    def __init__(self, chunk_size=5):
        self.chunk_size = int(chunk_size)

    def reset(self):
        pass

    def set_instruction(self, instruction):
        self.instruction = instruction

    def predict(self, observation):
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        return np.repeat(state[None], self.chunk_size, axis=0)
