import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
# scripts/ is not a package; put it on the path so the helpers can be
# imported by name instead of exec'd from a file spec -- @dataclass needs
# its module present in sys.modules to resolve annotations.
sys.path.insert(0, os.path.join(_ROOT, "scripts"))


@pytest.fixture(scope="session", autouse=True)
def _scheduler_tick():
    """Hold a 1 ms scheduler tick for the whole test session.

    Several tests measure short sleeps.  On Windows the default tick is about
    15.6 ms, so a 5 ms budget measures 16 ms and a 2.67 ms feeder interval runs
    four times too slowly -- the tests fail while the code is fine.  ``main()``
    raises the tick for exactly this reason (see ``rtvc/timing.py``); the test
    session has to do the same or it is measuring a different machine than the
    one the tool runs on.

    A no-op everywhere except Windows.
    """
    from rtvc.timing import HighResolutionTimer

    with HighResolutionTimer() as clock:
        yield clock
