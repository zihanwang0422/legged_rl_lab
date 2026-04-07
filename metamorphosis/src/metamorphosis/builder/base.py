from __future__ import annotations
import threading
import mujoco
from typing import Any, Dict, Type, Optional, TypeVar

from metamorphosis.utils.usd_utils import from_mjspec
from pxr import Usd


class SingletonMeta(type):
    """
    Thread-safe, per-class singleton metaclass.
    Each subclass gets its own unique instance.
    """
    _instances: Dict[Type, Any] = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # Fast path: instance exists
        if cls in cls._instances:
            raise RuntimeError(
                f"{cls.__name__} is a singleton class and can only be initialized once. "
                "Call `get_instance()` instead.")
        # Slow path: need to initialize under lock
        with cls._lock:
            obj = super().__call__(*args, **kwargs)  # calls __new__ and __init__
            cls._instances[cls] = obj
        return obj


T = TypeVar('T')

class BuilderBase(metaclass=SingletonMeta):

    def __init__(self):
        self.params = [] # list of parameters
    
    @classmethod
    def get_instance(cls) -> BuilderBase:
        return type(cls)._instances[cls]
    
    def sample_params(self, seed: int=-1) -> T:
        """Sample parameters for the builder."""
        raise NotImplementedError("sample_params is not implemented")
    
    def generate_mjspec(self, param: T) -> mujoco.MjSpec:
        """Generate Mujoco specification for the builder."""
        raise NotImplementedError("generate_mjspec is not implemented")

    def spawn(
        self,
        stage: Optional[Usd.Stage],
        prim_path: str,
        param: T,
    ):
        spec = self.generate_mjspec(param)
        self.params.append(param)
        return from_mjspec(stage, prim_path, spec)

