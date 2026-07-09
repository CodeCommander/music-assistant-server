"""
DIAL HTTP server for the YouTube Music Cast Receiver provider.

Implements the receiver side of the DIAL protocol (the HTTP half; discovery via
SSDP lives in ssdp.py): the device description document and the YouTube
application endpoint. When a sender app "casts", it POSTs a launch request with
a pairing code to the application endpoint — that pairing code is what links the
sender to our YouTube Lounge session.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl
from xml.sax.saxutils import escape

from aiohttp import web

if TYPE_CHECKING:
    import logging

DEVICE_DESC_PATH = "/dd.xml"
APP_PATH = "/apps/YouTube"

# Called with the parsed launch params (pairingCode, theme, ...) when a sender casts
LaunchCallback = Callable[[dict[str, str]], Awaitable[None]]


class DialServer:
    """Per-instance DIAL HTTP endpoint advertising the YouTube receiver app."""

    def __init__(
        self,
        *,
        port: int,
        publish_ip: str,
        device_uuid: str,
        friendly_name: str,
        launch_callback: LaunchCallback,
        logger: logging.Logger,
    ) -> None:
        """Initialize the DIAL server (does not bind until start() is called)."""
        self.port = port
        self.publish_ip = publish_ip
        self.device_uuid = device_uuid
        self.friendly_name = friendly_name
        self.launch_callback = launch_callback
        self.logger = logger
        self.app = web.Application()
        self.app.router.add_get(DEVICE_DESC_PATH, self._handle_device_desc)
        self.app.router.add_get(APP_PATH, self._handle_app_info)
        self.app.router.add_post(APP_PATH, self._handle_app_launch)
        self.app.router.add_delete(f"{APP_PATH}/run", self._handle_app_stop)
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    @property
    def location(self) -> str:
        """Absolute URL of the device description document (for SSDP LOCATION)."""
        return f"http://{self.publish_ip}:{self.port}{DEVICE_DESC_PATH}"

    async def start(self) -> None:
        """Start serving the DIAL endpoints."""
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await self._site.start()
        self.logger.debug(
            "DIAL server for '%s' listening on port %s", self.friendly_name, self.port
        )

    async def stop(self) -> None:
        """Stop serving."""
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None

    async def _handle_device_desc(self, request: web.Request) -> web.Response:
        """Serve the UPnP device description; the Application-URL header is mandatory."""
        body = (
            '<?xml version="1.0"?>\n'
            '<root xmlns="urn:schemas-upnp-org:device-1-0">\n'
            "  <specVersion><major>1</major><minor>0</minor></specVersion>\n"
            f"  <URLBase>http://{self.publish_ip}:{self.port}</URLBase>\n"
            "  <device>\n"
            "    <deviceType>urn:dial-multiscreen-org:device:dial:1</deviceType>\n"
            f"    <friendlyName>{escape(self.friendly_name)}</friendlyName>\n"
            "    <manufacturer>Music Assistant</manufacturer>\n"
            "    <modelName>YouTube Music Cast Receiver</modelName>\n"
            f"    <UDN>uuid:{self.device_uuid}</UDN>\n"
            "  </device>\n"
            "</root>\n"
        )
        return web.Response(
            text=body,
            content_type="application/xml",
            headers={"Application-URL": f"http://{self.publish_ip}:{self.port}/apps/"},
        )

    async def _handle_app_info(self, request: web.Request) -> web.Response:
        """Report the YouTube app state (always installed and running)."""
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<service xmlns="urn:dial-multiscreen-org:schemas:dial" dialVer="1.7">\n'
            "  <name>YouTube</name>\n"
            '  <options allowStop="false"/>\n'
            "  <state>running</state>\n"
            '  <link rel="run" href="run"/>\n'
            "</service>\n"
        )
        return web.Response(text=body, content_type="application/xml")

    async def _handle_app_launch(self, request: web.Request) -> web.Response:
        """
        Handle a cast launch: the sender POSTs pairingCode (and theme etc.) here.

        The pairing code is forwarded to the lounge session so the sender can
        connect; DIAL just acknowledges with 201 + the run URL.
        """
        raw_body = await request.text()
        params = dict(parse_qsl(raw_body, keep_blank_values=True))
        self.logger.info("DIAL launch request from %s: %s", request.remote, params)
        if params.get("pairingCode"):
            await self.launch_callback(params)
        else:
            self.logger.warning("DIAL launch without pairingCode; body: %s", raw_body)
        return web.Response(
            status=201,
            headers={
                "Location": f"http://{self.publish_ip}:{self.port}{APP_PATH}/run",
            },
        )

    async def _handle_app_stop(self, request: web.Request) -> web.Response:
        """DELETE on the run URL; we advertise allowStop=false so this is a no-op."""
        self.logger.debug("DIAL stop requested (ignored, allowStop=false)")
        return web.Response(status=200)
