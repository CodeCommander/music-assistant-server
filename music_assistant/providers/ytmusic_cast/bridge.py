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
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import MediaType, QueueOption

if TYPE_CHECKING:
    from music_assistant_models.media_items import Track

    from music_assistant.providers.ytmusic_cast import YTMusicCastProvider
    from music_assistant.providers.ytmusic_cast.lounge.messages import LoungeMessage

# concurrent ytmusic get_track lookups for a cold cast queue (uncached upstream)
RESOLVE_CONCURRENCY = 8


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
        # guards concurrent setPlaylist handling (phone can resend on reconnect)
        self._load_lock = asyncio.Lock()

    @property
    def queue_id(self) -> str:
        """The MA queue id this bridge controls (== the bound player id)."""
        return self.provider.mass_player_id

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
                await self.mass.players.cmd_volume_set(self.queue_id, volume)
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
