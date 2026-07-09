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

import hashlib
import socket
from typing import TYPE_CHECKING, cast

from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import ConfigEntryType, ProviderFeature

from music_assistant.constants import CONF_ENTRY_WARN_PREVIEW
from music_assistant.helpers.util import select_free_port
from music_assistant.models.plugin import PluginProvider
from music_assistant.providers.ytmusic_cast.dial import DialServer
from music_assistant.providers.ytmusic_cast.ssdp import DialAdvertisement, SharedSsdpResponder

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

CONF_MASS_PLAYER_ID = "mass_player_id"
CONF_CAST_NAME = "cast_name"
CONF_PORT = "port"
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
    )


class YTMusicCastProvider(PluginProvider):
    """Exposes a Music Assistant player as a YouTube Music cast target."""

    _dial_server: DialServer | None = None

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
        self.logger.info(
            "Cast target '%s' advertising for player %s (DIAL port %s)",
            self.cast_name,
            self.mass_player_id,
            port,
        )

    async def unload(self, is_removed: bool = False) -> None:
        """Tear down SSDP advertisement and DIAL endpoint."""
        await SharedSsdpResponder.get().unregister(self.device_uuid)
        if self._dial_server:
            await self._dial_server.stop()
            self._dial_server = None

    async def _handle_dial_launch(self, params: dict[str, str]) -> None:
        """
        Handle a cast launch from a sender app.

        Phase 1 skeleton: log the pairing code. The lounge session (next phase)
        will register the pairing code so the sender connects to our screen.
        """
        self.logger.info(
            "Cast launch received (pairingCode=%s theme=%s) - lounge session not yet implemented",
            params.get("pairingCode"),
            params.get("theme"),
        )

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
