"""Clocks for the streamer: a real one for runs, a fake one for fast tests."""

import asyncio
import time


class RealClock:
    """Seconds since the call started, measured on the wall clock."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def reset(self) -> None:
        self._start = time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self._start

    async def sleep_until(self, t: float) -> None:
        """Wait until time t. Windows sleeps are coarse (~15 ms), so sleep most of
        the way and then yield in a tight loop for the last couple of ms."""
        delay = t - self.now()
        if delay > 0.003:
            await asyncio.sleep(delay - 0.002)
        while self.now() < t:
            await asyncio.sleep(0)


class FakeClock:
    """Jumps straight to the requested time, so a 40 s call runs instantly."""

    def __init__(self) -> None:
        self.t = 0.0

    def reset(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    async def sleep_until(self, t: float) -> None:
        self.t = max(self.t, t)
        await asyncio.sleep(0)
