"""
YouTube Music Cast Receiver plugin for Music Assistant.

Makes a Music Assistant player appear as a cast target in the YouTube Music app.
Casting hands the phone's playback queue to Music Assistant via YouTube's Lounge
API; the tracks then play natively through the YouTube Music music provider (the
videoIds in a cast queue are the same ids the ytmusic provider uses), so no audio
ever flows from the phone.

Each provider instance is tied to exactly one Music Assistant player and appears
as its own device in the cast menu (multi-instance, like the Spotify Connect
plugin). Discovery is DIAL: an SSDP responder (shared across instances, see
ssdp.py) plus a small per-instance HTTP endpoint (dial.py).
"""

from __future__ import annotations

import asyncio
import hashlib
import socket
from typing import TYPE_CHECKING, cast

from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import ConfigEntryType, ProviderFeature

from music_assistant.constants import CONF_ENTRY_WARN_PREVIEW
from music_assistant.helpers.util import select_free_port
from music_assistant.models.plugin import PluginProvider
from music_assistant.providers.ytmusic_cast.bridge import CastQueueBridge
from music_assistant.providers.ytmusic_cast.dial import DialServer
from music_assistant.providers.ytmusic_cast.lounge.messages import (
    LoungeMessage,
    now_playing,
    on_volume_changed,
)
from music_assistant.providers.ytmusic_cast.lounge.session import LoungeSession, ScreenInfo
from music_assistant.providers.ytmusic_cast.ssdp import DialAdvertisement, SharedSsdpResponder

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

CONF_MASS_PLAYER_ID = "mass_player_id"
CONF_CAST_NAME = "cast_name"
CONF_PORT = "port"
CONF_SCREEN_ID = "screen_id"
CONF_KEEP_PLAYING_ON_DISCONNECT = "keep_playing_on_disconnect"

DEFAULT_CAST_NAME = "Music Assistant"
PORT_RANGE_START = 8790
PORT_RANGE_END = 8890

SUPPORTED_FEATURES: set[ProviderFeature] = set()


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return YTMusicCastProvider(mass, manifest, config, SUPPORTED_FEATURES)


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,  # noqa: ARG001
    action: str | None = None,  # noqa: ARG001
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """
    Return Config entries to setup this provider.

    instance_id: id of an existing provider instance (None if new instance setup).
    action: [optional] action key called from config entries UI.
    values: the (intermediate) raw values for config entries sent with the action.
    """
    # default the cast name to the selected player's name once one is picked
    cast_name_default = DEFAULT_CAST_NAME
    if (
        values
        and (player_id := values.get(CONF_MASS_PLAYER_ID))
        and (player := mass.players.get_player(str(player_id)))
    ):
        cast_name_default = player.display_name
    return (
        CONF_ENTRY_WARN_PREVIEW,
        ConfigEntry(
            key=CONF_MASS_PLAYER_ID,
            type=ConfigEntryType.STRING,
            options=[
                ConfigValueOption(x.player_id, title=x.display_name)
                for x in sorted(
                    mass.players.all_players(False, False), key=lambda p: p.display_name.lower()
                )
            ],
            required=True,
        ),
        ConfigEntry(
            key=CONF_CAST_NAME,
            type=ConfigEntryType.STRING,
            default_value=cast_name_default,
        ),
        ConfigEntry(
            key=CONF_KEEP_PLAYING_ON_DISCONNECT,
            type=ConfigEntryType.BOOLEAN,
            default_value=True,
        ),
        ConfigEntry(
            key=CONF_PORT,
            type=ConfigEntryType.INTEGER,
            default_value=None,
            required=False,
            hidden=True,
        ),
        ConfigEntry(
            key=CONF_SCREEN_ID,
            type=ConfigEntryType.STRING,
            default_value=None,
            required=False,
            hidden=True,
        ),
    )


class YTMusicCastProvider(PluginProvider):
    """Exposes a Music Assistant player as a YouTube Music cast target."""

    _dial_server: DialServer | None = None
    _lounge_session: LoungeSession | None = None
    _bridge: CastQueueBridge | None = None

    async def handle_async_init(self) -> None:
        """Start the DIAL endpoint and register the SSDP advertisement."""
        self.mass_player_id = cast("str", self.config.get_value(CONF_MASS_PLAYER_ID))
        self.cast_name = (
            cast("str | None", self.config.get_value(CONF_CAST_NAME)) or DEFAULT_CAST_NAME
        )
        # deterministic device uuid: stable across restarts and config wipes so the
        # phone's cast menu keeps recognizing this as the same device
        digest = hashlib.sha256(f"ytmusic_cast:{self.instance_id}".encode()).hexdigest()
        self.device_uuid = (
            f"{digest[0:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"
        )
        port = await self._resolve_port()
        self._dial_server = DialServer(
            port=port,
            publish_ip=str(self.mass.streams.publish_ip),
            device_uuid=self.device_uuid,
            friendly_name=self.cast_name,
            launch_callback=self._handle_dial_launch,
            logger=self.logger,
        )
        await self._dial_server.start()
        await SharedSsdpResponder.get().register(
            DialAdvertisement(
                device_uuid=self.device_uuid,
                location=self._dial_server.location,
                friendly_name=self.cast_name,
            ),
            self.logger,
        )
        self._lounge_session = LoungeSession(
            http_session=self.mass.http_session,
            screen=ScreenInfo(name=self.cast_name, device_id=self.device_uuid),
            get_screen_id=lambda: cast("str | None", self.config.get_value(CONF_SCREEN_ID)),
            set_screen_id=self._persist_screen_id,
            on_messages=self._handle_lounge_messages,
            on_terminate=self._handle_lounge_terminate,
            logger=self.logger,
        )
        self._bridge = CastQueueBridge(self)
        self.mass.create_task(self._start_lounge_session())
        self.logger.info(
            "Cast target '%s' advertising for player %s (DIAL port %s)",
            self.cast_name,
            self.mass_player_id,
            port,
        )

    async def unload(self, is_removed: bool = False) -> None:
        """Tear down lounge session, SSDP advertisement and DIAL endpoint."""
        if self._lounge_session:
            await self._lounge_session.end()
            self._lounge_session = None
        await SharedSsdpResponder.get().unregister(self.device_uuid)
        if self._dial_server:
            await self._dial_server.stop()
            self._dial_server = None

    async def _start_lounge_session(self) -> None:
        """Establish the lounge session, retrying a few times on startup failures."""
        assert self._lounge_session is not None
        for attempt in range(1, 4):
            try:
                await self._lounge_session.begin()
            except Exception as err:
                self.logger.warning(
                    "Failed to establish lounge session (attempt %s/3): %s", attempt, err
                )
                await asyncio.sleep(10 * attempt)
            else:
                return
        self.logger.error("Lounge session could not be established; casting will not work")

    async def _handle_dial_launch(self, params: dict[str, str]) -> None:
        """
        Handle a cast launch from a sender app.

        Registers the pairing code with the lounge session so the sender's
        connection completes through YouTube's cloud.
        """
        pairing_code = params.get("pairingCode", "")
        if not self._lounge_session:
            self.logger.warning("Cast launch received but lounge session does not exist")
            return
        if not self._lounge_session.running:
            # session may have died (or never come up): a cast is the perfect
            # moment to try bringing it back
            self.logger.info("Lounge session not running; starting it for this cast")
            try:
                await self._lounge_session.begin()
            except Exception as err:
                self.logger.error("Cannot handle cast: lounge session failed to start: %s", err)
                return
        try:
            await self._lounge_session.register_pairing_code(pairing_code)
        except Exception as err:
            self.logger.error("Failed to register pairing code: %s", err)

    async def _handle_lounge_messages(self, messages: list[LoungeMessage]) -> None:
        """
        Handle inbound lounge messages.

        Queue and transport messages go to the bridge; liveness probes are
        answered here. Everything else is logged for protocol visibility.
        """
        assert self._lounge_session is not None
        assert self._bridge is not None
        for message in messages:
            self.logger.debug(
                "LOUNGE message '%s' (AID=%s): %s", message.name, message.aid, message.payload
            )
            try:
                if await self._bridge.handle_message(message):
                    continue
            except Exception as err:
                self.logger.exception("Error handling '%s' message: %s", message.name, err)
                continue
            if message.name == "getNowPlaying":
                await self._lounge_session.send(now_playing(message.aid))
            elif message.name == "getVolume":
                await self._lounge_session.send(
                    on_volume_changed(message.aid, level=50, muted=False)
                )

    def _handle_lounge_terminate(self, error: Exception) -> None:
        """Log irrecoverable lounge session death (a cast will restart it via retry)."""
        self.logger.error("Lounge session terminated: %s", error)

    def _persist_screen_id(self, screen_id: str) -> None:
        """Persist the screen id so the phone can reconnect across MA restarts."""
        try:
            self.mass.config.set_raw_provider_config_value(
                self.instance_id, CONF_SCREEN_ID, screen_id
            )
        except Exception as err:
            self.logger.debug("Failed to persist screen id: %s", err)

    async def _resolve_port(self) -> int:
        """Return this instance's persistent DIAL port, allocating one if needed."""
        configured_port = self.config.get_value(CONF_PORT)
        if isinstance(configured_port, int) and self._is_port_available(configured_port):
            return configured_port
        port = await select_free_port(PORT_RANGE_START, PORT_RANGE_END)
        try:
            self.mass.config.set_raw_provider_config_value(self.instance_id, CONF_PORT, port)
        except Exception as err:
            self.logger.debug("Failed to persist DIAL port %s: %s", port, err)
        return port

    @staticmethod
    def _is_port_available(port: int) -> bool:
        """Check whether a TCP port can still be bound on all interfaces."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", port))
        except OSError:
            return False
        return True
