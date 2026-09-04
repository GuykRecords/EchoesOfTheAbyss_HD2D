"""rtvc -- realtime voice-conversion I/O, DSP and latency measurement.

Only pure-python/numpy modules are re-exported here.  ``rtvc.audio_io`` pulls
in sounddevice/PortAudio and is imported on demand so this package stays
importable on machines with no audio stack.
"""

__version__ = "0.1.0"

from . import dsp, engines, realtime  # noqa: F401

__all__ = ["dsp", "engines", "realtime", "__version__"]
