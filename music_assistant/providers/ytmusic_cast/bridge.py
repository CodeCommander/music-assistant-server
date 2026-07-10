"""
Bridge between inbound lounge messages and the Music Assistant queue.

Receives the queue the phone hands over on cast (setPlaylist: a list of YouTube
videoIds plus current index/position) and mirrors it onto the bound MA player's
queue. The videoIds ARE the ytmusic music provider's track ids, so resolution is
a direct provider lookup — the tracks then play natively through MA (no audio
ever comes from the phone). Transport commands (play/pause/seek/...) map to the
queue controller; volume maps to the player.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import EventType, MediaType, PlaybackState, QueueOption

from music_assistant.providers.ytmusic_cast.lounge.messages import (
    PLAYER_STATUS_IDLE,
    PLAYER_STATUS_PAUSED,
    PLAYER_STATUS_PLAYING,
    LoungeMessage,
    now_playing,
    on_has_previous_next_changed,
    on_state_change,
    on_volume_changed,
)

if TYPE_CHECKING:
    from music_assistant_models.event import MassEvent
    from music_assistant_models.media_items import Track

    from music_assistant.providers.ytmusic_cast import YTMusicCastProvider

# concurrent ytmusic get_track lookups for a cold cast queue (uncached upstream)
RESOLVE_CONCURRENCY = 8
# debounce for outbound state pushes (matches the reference's volume debounce)
STATE_PUSH_DEBOUNCE = 0.2
# window in which a player volume change matching a phone-set value is treated
# as the echo of that command rather than a change to report back
VOLUME_ECHO_WINDOW = 3.0


class CastQueueBridge:
    """Maps one cast session's queue and transport onto the bound MA player."""

    def __init__(self, provider: YTMusicCastProvider) -> None:
        """Initialize the bridge for the given provider instance."""
        self.provider = provider
        self.mass = provider.mass
        self.logger = provider.logger
        # the videoIds of the lounge queue, in lounge order
        self._video_ids: list[str] = []
        # lounge queue index -> MA queue index (differs when tracks fail to resolve)
        self._index_map: dict[int, int] = {}
        self._list_id: str | None = None
        self._ctt: str | None = None
        # guards concurrent setPlaylist handling (phone can resend on reconnect)
        self._load_lock = asyncio.Lock()
        # outbound state tracking
        self._unsubs: list[Any] = []
        self._last_video_id: str | None = None
        self._last_status: int | None = None
        self._last_position: float = 0
        self._last_volume_sent: int | None = None
        self._phone_volume: tuple[int, float] | None = None  # (level, monotonic ts)
        self._cpn: str = _new_cpn()

    @property
    def queue_id(self) -> str:
        """The MA queue id this bridge controls (== the bound player id)."""
        return self.provider.mass_player_id

    @property
    def session_active(self) -> bool:
        """Whether a cast queue has been loaded (state reporting is gated on this)."""
        return bool(self._video_ids)

    def start(self) -> None:
        """Subscribe to MA events for the bound player to mirror state to senders."""
        self._unsubs.append(
            self.mass.subscribe(
                self._on_ma_event,
                (
                    EventType.PLAYER_UPDATED,
                    EventType.QUEUE_UPDATED,
                    EventType.QUEUE_TIME_UPDATED,
                    EventType.QUEUE_ITEMS_UPDATED,
                ),
                id_filter=self.queue_id,
            )
        )

    def stop(self) -> None:
        """Unsubscribe from MA events."""
        for unsub in self._unsubs:
            unsub()
        self._unsubs.clear()

    async def handle_message(self, message: LoungeMessage) -> bool:
        """
        Act on an inbound lounge message; returns True when handled.

        :param message: The parsed lounge message.
        """
        payload = message.payload if isinstance(message.payload, dict) else {}
        match message.name:
            case "setPlaylist":
                await self._handle_set_playlist(payload)
            case "updatePlaylist":
                await self._handle_update_playlist(payload)
            case "play":
                await self.mass.player_queues.play(self.queue_id)
            case "pause":
                await self.mass.player_queues.pause(self.queue_id)
            case "stopVideo":
                await self.mass.player_queues.stop(self.queue_id)
            case "next":
                await self.mass.player_queues.next(self.queue_id)
            case "previous":
                await self.mass.player_queues.previous(self.queue_id)
            case "seekTo":
                position = int(float(payload.get("newTime") or 0))
                await self.mass.player_queues.seek(self.queue_id, position)
            case "setVolume":
                volume = int(float(payload.get("volume") or 0))
                self._phone_volume = (volume, time.monotonic())
                await self.mass.players.cmd_volume_set(self.queue_id, volume)
            case "getNowPlaying":
                await self._send_now_playing(message.aid)
            case "getVolume":
                await self._send_volume(message.aid)
            case _:
                return False
        return True

    async def _handle_set_playlist(self, payload: dict[str, Any]) -> None:
        """Load the cast queue onto the MA queue and start at the sender's position."""
        video_ids = [v for v in (payload.get("videoIds") or "").split(",") if v]
        current_index = int(payload.get("currentIndex") or 0)
        current_time = int(float(payload.get("currentTime") or 0))
        list_id = payload.get("listId")
        start_playing = (payload.get("playbackState") or "PLAYING") == "PLAYING"
        if not video_ids:
            self.logger.warning("setPlaylist without videoIds; ignoring")
            return

        async with self._load_lock:
            if video_ids == self._video_ids and self._list_id == list_id:
                # same queue, new position: the phone skipping tracks arrives here
                await self._play_lounge_index(current_index, current_time)
                return

            self.logger.info(
                "Loading cast queue: %s tracks, starting at index %s (position %ss)",
                len(video_ids),
                current_index,
                current_time,
            )
            tracks, index_map = await self._resolve_tracks(video_ids)
            if not tracks:
                self.logger.error("None of the cast queue's tracks could be resolved")
                return
            self._video_ids = video_ids
            self._index_map = index_map
            self._list_id = list_id
            self._ctt = payload.get("ctt")

            await self.mass.player_queues.play_media(
                self.queue_id,
                media=list(tracks),
                option=QueueOption.REPLACE,
            )
            if start_playing:
                await self._play_lounge_index(current_index, current_time)
            else:
                await self.mass.player_queues.pause(self.queue_id)

    async def _handle_update_playlist(self, payload: dict[str, Any]) -> None:
        """
        Handle queue edits from the phone.

        v1 keeps this simple: re-run the setPlaylist flow when the video list
        changed (REPLACE keeps playback position through play_index).
        """
        video_ids = [v for v in (payload.get("videoIds") or "").split(",") if v]
        if video_ids and video_ids != self._video_ids:
            self.logger.debug("updatePlaylist with changed videoIds; reloading queue")
            # keep playing the current track: reuse setPlaylist handling with
            # the current index if the payload does not carry one
            if "currentIndex" not in payload and self._video_ids:
                current = self._current_lounge_index()
                if current is not None:
                    payload = {**payload, "currentIndex": str(current)}
            await self._handle_set_playlist(payload)

    async def _resolve_tracks(self, video_ids: list[str]) -> tuple[list[Track], dict[int, int]]:
        """Resolve videoIds to ytmusic tracks concurrently, keeping an index map."""
        semaphore = asyncio.Semaphore(RESOLVE_CONCURRENCY)

        async def _resolve(video_id: str) -> Track | None:
            async with semaphore:
                try:
                    item = await self.mass.music.get_item(MediaType.TRACK, video_id, "ytmusic")
                except Exception as err:
                    self.logger.warning("Failed to resolve track %s: %s", video_id, err)
                    return None
                return item  # type: ignore[return-value]

        results = await asyncio.gather(*(_resolve(vid) for vid in video_ids))
        tracks: list[Track] = []
        index_map: dict[int, int] = {}
        for lounge_index, track in enumerate(results):
            if track is None:
                continue
            index_map[lounge_index] = len(tracks)
            tracks.append(track)
        if len(tracks) != len(video_ids):
            self.logger.warning(
                "Resolved %s/%s tracks of the cast queue", len(tracks), len(video_ids)
            )
        return tracks, index_map

    async def _play_lounge_index(self, lounge_index: int, seek_position: int = 0) -> None:
        """Jump the MA queue to the given lounge queue index."""
        ma_index = self._index_map.get(lounge_index)
        if ma_index is None:
            # requested track failed to resolve: fall back to the next resolvable one
            for candidate in range(lounge_index + 1, len(self._video_ids)):
                if (ma_index := self._index_map.get(candidate)) is not None:
                    seek_position = 0
                    break
            else:
                self.logger.warning("No resolvable track at or after index %s", lounge_index)
                return
        await self.mass.player_queues.play_index(
            self.queue_id, ma_index, seek_position=seek_position
        )

    def _current_lounge_index(self) -> int | None:
        """Return the lounge index of the MA queue's current track, if mapped."""
        queue = self.mass.player_queues.get(self.queue_id)
        if not queue or queue.current_index is None:
            return None
        for lounge_index, ma_index in self._index_map.items():
            if ma_index == queue.current_index:
                return lounge_index
        return None

    async def _on_ma_event(self, event: MassEvent) -> None:
        """Mirror MA player/queue state changes back to connected senders."""
        session = self.provider.lounge_session
        if not session or not session.running or not self.session_active:
            return
        if event.event == EventType.PLAYER_UPDATED:
            await self._maybe_send_volume()
        await self._push_state()

    async def _push_state(self) -> None:
        """Send state (and track change) updates, debounced."""
        session = self.provider.lounge_session
        assert session is not None
        status, position, duration, lounge_index = self._snapshot()
        video_id = (
            self._video_ids[lounge_index]
            if lounge_index is not None and lounge_index < len(self._video_ids)
            else None
        )
        track_changed = video_id != self._last_video_id
        state_changed = status != self._last_status
        seeked = abs(position - self._last_position) > 3
        self._last_position = position
        if not (track_changed or state_changed or seeked):
            return
        self._last_video_id = video_id
        self._last_status = status
        if track_changed:
            self._cpn = _new_cpn()
        messages: list[LoungeMessage] = []
        if track_changed and video_id:
            messages.append(self._build_now_playing(None, video_id, status, position, duration))
        messages.append(
            on_state_change(
                None, status=status, position=position, duration=duration, cpn=self._cpn
            )
        )
        if track_changed and lounge_index is not None:
            messages.append(
                on_has_previous_next_changed(
                    None,
                    has_previous=lounge_index > 0,
                    has_next=lounge_index < len(self._video_ids) - 1,
                )
            )
        await session.send(messages, defer=("state", STATE_PUSH_DEBOUNCE))

    async def _maybe_send_volume(self) -> None:
        """Report player volume changes, suppressing echoes of phone-set values."""
        session = self.provider.lounge_session
        assert session is not None
        player = self.mass.players.get_player(self.queue_id)
        if not player or player.volume_level is None:
            return
        level = int(player.volume_level)
        if level == self._last_volume_sent:
            return
        if self._phone_volume:
            phone_level, when = self._phone_volume
            if level == phone_level and time.monotonic() - when < VOLUME_ECHO_WINDOW:
                self._last_volume_sent = level
                return
        self._last_volume_sent = level
        await session.send(
            on_volume_changed(None, level=level, muted=level == 0),
            defer=("volume", STATE_PUSH_DEBOUNCE),
        )

    async def _send_now_playing(self, aid: int | None) -> None:
        """Answer a getNowPlaying probe with the real current state."""
        session = self.provider.lounge_session
        assert session is not None
        status, position, duration, lounge_index = self._snapshot()
        video_id = (
            self._video_ids[lounge_index]
            if lounge_index is not None and lounge_index < len(self._video_ids)
            else None
        )
        if video_id:
            await session.send(self._build_now_playing(aid, video_id, status, position, duration))
        else:
            await session.send(now_playing(aid))

    async def _send_volume(self, aid: int | None) -> None:
        """Answer a getVolume probe with the real player volume."""
        session = self.provider.lounge_session
        assert session is not None
        player = self.mass.players.get_player(self.queue_id)
        level = int(player.volume_level or 0) if player else 0
        self._last_volume_sent = level
        await session.send(on_volume_changed(aid, level=level, muted=level == 0))

    def _build_now_playing(
        self, aid: int | None, video_id: str, status: int, position: float, duration: float
    ) -> LoungeMessage:
        lounge_index = self._video_ids.index(video_id) if video_id in self._video_ids else None
        return now_playing(
            aid,
            video_id=video_id,
            status=status,
            position=position,
            duration=duration,
            cpn=self._cpn,
            list_id=self._list_id,
            current_index=lounge_index,
            ctt=self._ctt,
        )

    def _snapshot(self) -> tuple[int, float, float, int | None]:
        """Return (lounge status code, position, duration, lounge index) for the queue."""
        queue = self.mass.player_queues.get(self.queue_id)
        if not queue:
            return PLAYER_STATUS_IDLE, 0, 0, None
        match queue.state:
            case PlaybackState.PLAYING:
                status = PLAYER_STATUS_PLAYING
            case PlaybackState.PAUSED:
                status = PLAYER_STATUS_PAUSED
            case _:
                status = PLAYER_STATUS_IDLE
        position = float(queue.corrected_elapsed_time or 0)
        duration = 0.0
        if queue.current_item and queue.current_item.duration:
            duration = float(queue.current_item.duration)
        return status, position, duration, self._current_lounge_index()


def _new_cpn() -> str:
    """Generate a 16-char client playback nonce."""
    return secrets.token_urlsafe(12)
