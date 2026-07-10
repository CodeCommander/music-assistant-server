"""
Message parsing and construction for the YouTube Lounge API.

The Lounge browser-channel wire format frames each message as
``[AID, ["name", payload?]]`` inside chunked responses that also carry
length-prefix lines. Parsing is done the same way as the reference
yt-cast-receiver implementation: strip newlines and regex the message tuples
out, which conveniently skips the length prefixes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# https://github.com/patrickkfkan/yt-cast-receiver (Message.ts)
_MESSAGE_REGEX = re.compile(r'\[(\d+),\["(.+?)"(?:,(.*?))?\]\]')

# Player status codes understood by lounge senders
PLAYER_STATUS_IDLE = -1
PLAYER_STATUS_PLAYING = 1
PLAYER_STATUS_PAUSED = 2
PLAYER_STATUS_LOADING = 3
PLAYER_STATUS_STOPPED = 4

AUTOPLAY_UNSUPPORTED = "UNSUPPORTED"


@dataclass
class LoungeMessage:
    """A single message received from or destined for a lounge sender."""

    name: str
    payload: Any = field(default_factory=dict)
    # AID of the incoming message this replies to; None for unsolicited sends.
    # Nulled when the session's AID sequence resets (token refresh).
    aid: int | None = None


def parse_incoming(data: str) -> list[LoungeMessage]:
    """
    Parse raw browser-channel text into lounge messages.

    :param data: Unprocessed (possibly multi-line) text from the bind endpoint.
    """
    messages: list[LoungeMessage] = []
    for match in _MESSAGE_REGEX.finditer(data.replace("\n", "").replace("\r", "")):
        aid = int(match.group(1))
        name = match.group(2)
        payload: Any = None
        if match.group(3) is not None:
            try:
                parsed = json.loads(f"[{match.group(3)}]")
            except json.JSONDecodeError:
                continue
            payload = parsed[0] if isinstance(parsed, list) and len(parsed) == 1 else parsed
        messages.append(LoungeMessage(name=name, payload=payload, aid=aid))
    return messages


def now_playing(
    aid: int | None,
    *,
    video_id: str | None = None,
    status: int = PLAYER_STATUS_IDLE,
    position: float = 0,
    duration: float = 0,
    cpn: str | None = None,
    list_id: str | None = None,
    current_index: int | None = None,
    ctt: str | None = None,
) -> LoungeMessage:
    """Build a 'nowPlaying' message describing the current video and state."""
    payload: dict[str, Any] = {}
    if video_id:
        payload = {
            "videoId": video_id,
            "currentTime": position,
            "duration": duration,
            "state": status,
            "loadedTime": duration if status != PLAYER_STATUS_IDLE else 0,
            "seekableStartTime": 0,
            "seekableEndTime": duration,
        }
        if cpn:
            payload["cpn"] = cpn
        if list_id:
            payload["listId"] = list_id
        if current_index is not None:
            payload["currentIndex"] = current_index
        if ctt:
            payload["ctt"] = ctt
    return LoungeMessage("nowPlaying", payload, aid)


def on_state_change(
    aid: int | None,
    *,
    status: int,
    position: float,
    duration: float,
    cpn: str | None = None,
) -> LoungeMessage:
    """Build an 'onStateChange' message with the current player state."""
    payload: dict[str, Any] = {
        "state": status,
        "currentTime": position,
        "duration": duration,
        "loadedTime": duration if status != PLAYER_STATUS_IDLE else 0,
        "seekableStartTime": 0,
        "seekableEndTime": duration,
    }
    if cpn:
        payload["cpn"] = cpn
    return LoungeMessage("onStateChange", payload, aid)


def on_volume_changed(aid: int | None, *, level: int, muted: bool) -> LoungeMessage:
    """Build an 'onVolumeChanged' message."""
    return LoungeMessage("onVolumeChanged", {"volume": level, "muted": muted}, aid)


def on_autoplay_mode_changed(aid: int | None, mode: str = AUTOPLAY_UNSUPPORTED) -> LoungeMessage:
    """Build an 'onAutoplayModeChanged' message."""
    return LoungeMessage("onAutoplayModeChanged", {"autoplayMode": mode}, aid)


def on_has_previous_next_changed(
    aid: int | None, *, has_previous: bool, has_next: bool
) -> LoungeMessage:
    """Build an 'onHasPreviousNextChanged' message."""
    return LoungeMessage(
        "onHasPreviousNextChanged",
        {"hasPrevious": has_previous, "hasNext": has_next},
        aid,
    )


def lounge_screen_disconnected() -> LoungeMessage:
    """Build a 'loungeScreenDisconnected' message (sent when the screen goes away)."""
    return LoungeMessage("loungeScreenDisconnected", {})
