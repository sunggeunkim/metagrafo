from scripts.start_server import wait_for_start


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _Server:
    """A started server that crashes at `exits_at` seconds, or never when None."""

    def __init__(self, clock: _Clock, exits_at: float | None = None, output: str = "") -> None:
        self._clock = clock
        self._exits_at = exits_at
        self._output = output

    def exited(self) -> bool:
        return self._exits_at is not None and self._clock.now >= self._exits_at

    def output(self) -> str:
        return self._output


def _health(clock: _Clock, answers_at: float | None):
    """/health as Windows answers it: a refused localhost connection takes about 2 s."""

    def healthy() -> bool:
        if answers_at is not None and clock.now >= answers_at:
            return True
        clock.now += 2.0
        return False

    return healthy


def test_a_server_that_answers_health_has_started() -> None:
    clock = _Clock()
    outcome = wait_for_start(_Server(clock), _health(clock, answers_at=0.0), clock, clock.sleep)
    assert outcome.started is True


def test_a_server_that_answers_health_only_after_a_minute_has_still_started() -> None:
    clock = _Clock()
    outcome = wait_for_start(_Server(clock), _health(clock, answers_at=60.0), clock, clock.sleep)
    assert outcome.started is True


def test_a_server_that_crashes_on_import_is_reported_within_ten_seconds_with_its_error() -> None:
    clock = _Clock()
    error = "ModuleNotFoundError: No module named 'httpx'\n"
    server = _Server(clock, exits_at=1.0, output=error)
    outcome = wait_for_start(server, _health(clock, answers_at=None), clock, clock.sleep)
    assert outcome.started is False
    assert error in outcome.output
    assert clock.now <= 10.0


def test_a_server_that_stays_up_but_never_answers_is_given_up_on_after_three_minutes() -> None:
    clock = _Clock()
    outcome = wait_for_start(_Server(clock), _health(clock, answers_at=None), clock, clock.sleep)
    assert outcome.started is False
    assert 180.0 <= clock.now <= 190.0
