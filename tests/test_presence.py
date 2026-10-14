from concurrent.futures import ThreadPoolExecutor

import pytest

from paperleet.presence import PresenceTracker, PresenceUnavailable, visitor_from_payload


def test_expiry_uses_last_heartbeat_and_expires_at_window_boundary():
    now = [0.0]
    tracker = PresenceTracker(clock=lambda: now[0])
    assert tracker.snapshot() == {"online": 0, "window_seconds": 90, "heartbeat_seconds": 25}
    tracker.heartbeat("first")
    now[0] = 30
    tracker.heartbeat("second")
    tracker.heartbeat("first")
    now[0] = 119.99
    assert tracker.snapshot()["online"] == 2
    now[0] = 120
    assert tracker.snapshot()["online"] == 0
    assert not tracker._visitors


def test_browser_and_authenticated_account_deduplication():
    tracker = PresenceTracker()
    assert tracker.heartbeat("browser1")["online"] == 1
    assert tracker.heartbeat("browser1")["online"] == 1
    assert tracker.heartbeat("browser2")["online"] == 2
    assert tracker.heartbeat("browser1", "alice")["online"] == 2
    assert tracker.heartbeat("browser2", "alice")["online"] == 1
    assert tracker.heartbeat("browser3", "bob")["online"] == 2


def test_login_logout_and_switch_replace_browser_identity():
    tracker = PresenceTracker()
    tracker.heartbeat("browser1", "alice")
    assert tracker.heartbeat("browser2", "alice")["online"] == 1
    assert tracker.heartbeat("browser1", "bob")["online"] == 2
    assert tracker.heartbeat("browser2", "bob")["online"] == 1
    assert tracker.heartbeat("browser1")["online"] == 2
    assert tracker.heartbeat("browser2")["online"] == 2
    assert tracker.heartbeat("browser1", "alice")["online"] == 2


def test_active_account_survives_another_browser_expiring():
    now = [0.0]
    tracker = PresenceTracker(clock=lambda: now[0])
    tracker.heartbeat("browser1", "alice")
    now[0] = 50
    tracker.heartbeat("browser2", "alice")
    now[0] = 100
    assert tracker.snapshot()["online"] == 1
    assert set(tracker._visitors) == {"browser2"}


def test_capacity_preserves_live_visitors_and_recovers_after_expiry():
    now = [0.0]
    tracker = PresenceTracker(clock=lambda: now[0], capacity=2)
    tracker.heartbeat("browser1")
    tracker.heartbeat("browser2")
    with pytest.raises(PresenceUnavailable):
        tracker.heartbeat("browser3")
    assert tracker.heartbeat("browser1", "alice")["online"] == 2
    now[0] = 90
    assert tracker.heartbeat("browser3")["online"] == 1
    assert len(tracker._visitors) == 1


def test_concurrent_heartbeats_and_snapshots_do_not_lose_visitors():
    tracker = PresenceTracker()

    def update(index):
        for _ in range(5):
            tracker.heartbeat(str(index), str(index // 2))
            tracker.snapshot()

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(update, range(100)))
    assert tracker.snapshot()["online"] == 50
    assert len(tracker._visitors) == 100


def test_trackers_are_scoped_to_one_server_instance():
    first, second = PresenceTracker(), PresenceTracker()
    first.heartbeat("browser")
    assert second.snapshot()["online"] == 0


@pytest.mark.parametrize("payload", [None, [], {}, {"visitor_id": None}, {"visitor_id": 42},
    {"visitor_id": ""}, {"visitor_id": "not-a-uuid"},
    {"visitor_id": "0" * 36}, {"visitor_id": "ab087ba6-e05d-4438-83dd-adf560b0a782\n"},
    {"visitor_id": "ab087ba6-e05d-4438-83dd-adf560b0a782", "user_id": "alice"}])
def test_only_a_uuid_is_accepted_from_client(payload):
    with pytest.raises(ValueError):
        visitor_from_payload(payload)


def test_uuid_case_is_normalized_for_deduplication():
    visitor = "AB087BA6-E05D-4438-83DD-ADF560B0A782"
    assert visitor_from_payload({"visitor_id": visitor}) == visitor.lower()
