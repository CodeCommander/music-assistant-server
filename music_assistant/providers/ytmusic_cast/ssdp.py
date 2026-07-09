"""
Shared SSDP responder for the YouTube Music Cast Receiver provider.

DIAL discovery works over SSDP: sender apps multicast an M-SEARCH for the DIAL
service type and every advertised device unicasts a response pointing at its
device-description URL. All provider instances share ONE UDP socket on port 1900:
binding multiple sockets with SO_REUSEPORT would make the kernel load-balance
incoming datagrams across them, so any given instance would miss searches.
Instances register their advertisement here; the single responder answers once
per registered instance.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import socket
import struct
from dataclasses import dataclass
from email.utils import formatdate
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging

SSDP_ADDR = "239.255.255.250"
SSDP_PORT = 1900
DIAL_ST = "urn:dial-multiscreen-org:service:dial:1"
# STs we answer with a DIAL advertisement
ANSWERED_STS = (DIAL_ST, "ssdp:all", "upnp:rootdevice")
NOTIFY_INTERVAL = 300  # seconds between periodic ssdp:alive notifications
CACHE_MAX_AGE = 1800
SERVER_STRING = "Linux/1.0 UPnP/1.1 MusicAssistant/1.0"


@dataclass
class DialAdvertisement:
    """A single advertised DIAL device (one per provider instance)."""

    device_uuid: str
    location: str  # absolute URL of the device description XML
    friendly_name: str


class _SsdpProtocol(asyncio.DatagramProtocol):
    """Datagram protocol that answers M-SEARCH requests for registered devices."""

    def __init__(self, responder: SharedSsdpResponder) -> None:
        self.responder = responder
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        """Store the transport once the socket is up."""
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Parse an incoming SSDP datagram and schedule responses if it is a match."""
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            return
        if not text.startswith("M-SEARCH"):
            return
        headers = _parse_headers(text)
        search_target = headers.get("st", "")
        if search_target not in ANSWERED_STS:
            return
        # cap the spec-mandated random response delay at a snappy maximum
        try:
            mx = min(int(headers.get("mx", "1")), 2)
        except ValueError:
            mx = 1
        self.responder.handle_search(addr, search_target, mx)


class SharedSsdpResponder:
    """
    Module-wide singleton answering DIAL M-SEARCHes for all provider instances.

    The socket is opened when the first advertisement registers and closed when
    the last one unregisters.
    """

    _instance: SharedSsdpResponder | None = None

    def __init__(self) -> None:
        """Initialize the responder; the socket opens on first register()."""
        self._advertisements: dict[str, DialAdvertisement] = {}
        self._transport: asyncio.DatagramTransport | None = None
        self._notify_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self.logger: logging.Logger | None = None

    @classmethod
    def get(cls) -> SharedSsdpResponder:
        """Return the process-wide responder instance."""
        if cls._instance is None:
            cls._instance = SharedSsdpResponder()
        return cls._instance

    async def register(self, advertisement: DialAdvertisement, logger: logging.Logger) -> None:
        """
        Register a DIAL advertisement and start the responder if needed.

        :param advertisement: The device advertisement to answer searches with.
        :param logger: Provider logger; adopted for responder diagnostics.
        """
        async with self._lock:
            self.logger = logger
            self._advertisements[advertisement.device_uuid] = advertisement
            if self._transport is None:
                await self._start()
        self._send_notify(advertisement, alive=True)

    async def unregister(self, device_uuid: str) -> None:
        """Remove an advertisement; stops the responder when none remain."""
        async with self._lock:
            advertisement = self._advertisements.pop(device_uuid, None)
            if advertisement and self._transport:
                self._send_notify(advertisement, alive=False)
            if not self._advertisements and self._transport is not None:
                await self._stop()

    def handle_search(self, addr: tuple[str, int], search_target: str, mx: int) -> None:
        """Schedule unicast M-SEARCH responses for every registered advertisement."""
        for advertisement in list(self._advertisements.values()):
            delay = random.uniform(0, max(mx, 1))
            asyncio.get_running_loop().call_later(
                delay, self._send_search_response, addr, search_target, advertisement
            )

    async def _start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", SSDP_PORT))
        membership = socket.inet_aton(SSDP_ADDR) + struct.pack("=I", socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _SsdpProtocol(self), sock=sock
        )
        self._notify_task = loop.create_task(self._notify_loop())
        if self.logger:
            self.logger.debug("SSDP responder listening on %s:%s", SSDP_ADDR, SSDP_PORT)

    async def _stop(self) -> None:
        if self._notify_task:
            self._notify_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._notify_task
            self._notify_task = None
        if self._transport:
            self._transport.close()
            self._transport = None
        if self.logger:
            self.logger.debug("SSDP responder stopped")

    async def _notify_loop(self) -> None:
        """Periodically broadcast ssdp:alive so idle cast menus stay populated."""
        while True:
            await asyncio.sleep(NOTIFY_INTERVAL)
            for advertisement in list(self._advertisements.values()):
                self._send_notify(advertisement, alive=True)

    def _send_search_response(
        self, addr: tuple[str, int], search_target: str, advertisement: DialAdvertisement
    ) -> None:
        if self._transport is None or advertisement.device_uuid not in self._advertisements:
            return
        # ssdp:all / rootdevice searches still get the DIAL service advertisement:
        # that is the only service we exist to expose
        response_st = DIAL_ST if search_target == "ssdp:all" else search_target
        usn = f"uuid:{advertisement.device_uuid}"
        if response_st != f"uuid:{advertisement.device_uuid}":
            usn = f"{usn}::{response_st}"
        message = (
            "HTTP/1.1 200 OK\r\n"
            f"CACHE-CONTROL: max-age={CACHE_MAX_AGE}\r\n"
            f"DATE: {formatdate(usegmt=True)}\r\n"
            "EXT:\r\n"
            f"LOCATION: {advertisement.location}\r\n"
            f"SERVER: {SERVER_STRING}\r\n"
            f"ST: {response_st}\r\n"
            f"USN: {usn}\r\n"
            "BOOTID.UPNP.ORG: 1\r\n"
            "CONFIGID.UPNP.ORG: 1\r\n"
            "\r\n"
        )
        with contextlib.suppress(OSError):
            self._transport.sendto(message.encode(), addr)

    def _send_notify(self, advertisement: DialAdvertisement, *, alive: bool) -> None:
        if self._transport is None:
            return
        nts = "ssdp:alive" if alive else "ssdp:byebye"
        message = (
            "NOTIFY * HTTP/1.1\r\n"
            f"HOST: {SSDP_ADDR}:{SSDP_PORT}\r\n"
            f"CACHE-CONTROL: max-age={CACHE_MAX_AGE}\r\n"
            f"LOCATION: {advertisement.location}\r\n"
            f"NT: {DIAL_ST}\r\n"
            f"NTS: {nts}\r\n"
            f"SERVER: {SERVER_STRING}\r\n"
            f"USN: uuid:{advertisement.device_uuid}::{DIAL_ST}\r\n"
            "BOOTID.UPNP.ORG: 1\r\n"
            "CONFIGID.UPNP.ORG: 1\r\n"
            "\r\n"
        )
        with contextlib.suppress(OSError):
            self._transport.sendto(message.encode(), (SSDP_ADDR, SSDP_PORT))


def _parse_headers(text: str) -> dict[str, str]:
    """Parse SSDP request headers into a lowercase-keyed dict."""
    headers: dict[str, str] = {}
    for line in text.split("\r\n")[1:]:
        if ":" in line:
            key, _, value = line.partition(":")
            headers[key.strip().lower()] = value.strip().strip('"')
    return headers
