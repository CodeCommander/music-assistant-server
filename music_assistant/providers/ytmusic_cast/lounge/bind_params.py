"""
Query-string state machine for the YouTube Lounge bind endpoint.

Port of yt-cast-receiver's BindParams: tracks the session identifiers
(SID/gsessionid/loungeIdToken) and the order-sensitive counters (RID/AID) used
to construct the query strings for establishing sessions, sending messages and
opening the RPC long-poll.
"""

from __future__ import annotations

import json
import random
import uuid
from typing import Literal
from urllib.parse import urlencode

QueryType = Literal["initSession", "sendMessage", "rpc"]


class MissingBindDataError(Exception):
    """Raised when required session data is missing for a query string."""

    def __init__(self, missing: list[str]) -> None:
        """Initialize with the list of missing field names."""
        self.missing = missing
        super().__init__(f"Missing bind params: {', '.join(missing)}")


class BindParams:
    """Mutable bind-parameter state for one lounge session."""

    def __init__(
        self,
        *,
        theme: str,
        device_id: str,
        screen_name: str,
        screen_app: str,
        brand: str,
        model: str,
    ) -> None:
        """Initialize bind parameters for a new session."""
        self.device = "LOUNGE_SCREEN"
        self.id = device_id
        self.obfuscated_gaia_id = ""
        self.name = screen_name
        self.app = screen_app
        self.theme = theme
        self.capabilities = "dsp,mic,dpa,ntb"
        self.cst = "m"
        self.mdx_version = 2
        self.lounge_id_token: str | None = None
        self.ver = 8
        self.v = 2
        self.cver = 1
        self.device_info = {
            "brand": brand,
            "model": model,
            "year": 0,
            "os": "Windows",
            "osVersion": "10.0",
            "chipset": "",
            "clientName": "TVHTML5",
            "dialAdditionalDataSupportLevel": "unsupported",
            "mdxDialServerType": "MDX_DIAL_SERVER_TYPE_UNKNOWN",
        }
        self.sid: str | None = None
        self.gsessionid: str | None = None
        self.rid = self._generate_rid()
        self.aid = 3
        self.t = 1

    def reset(self) -> None:
        """Clear session identifiers and re-randomize counters (token refresh)."""
        self.sid = None
        self.gsessionid = None
        self.lounge_id_token = None
        self.rid = self._generate_rid()
        self.aid = 3

    def update_with_message(self, name: str, payload: object, aid: int | None) -> None:
        """Absorb session identifiers ('c'/'S') and AID watermarks from a message."""
        if name == "c" and isinstance(payload, list) and payload:
            self.sid = str(payload[0])
        elif name == "S" and isinstance(payload, str):
            self.gsessionid = payload
        if aid:
            self.aid = max(self.aid, aid)

    def to_query_string(self, query_type: QueryType, aid: int | None = None) -> str:
        """
        Construct the query string for the given action type.

        :param query_type: One of initSession / sendMessage / rpc.
        :param aid: AID of the incoming message being replied to (sendMessage only).
        """
        missing = []
        if not self.lounge_id_token:
            missing.append("loungeIdToken")
        if query_type in ("sendMessage", "rpc"):
            if not self.sid:
                missing.append("SID")
            if not self.gsessionid:
                missing.append("gsessionid")
        if missing:
            raise MissingBindDataError(missing)

        params: dict[str, object] = {
            "device": self.device,
            "id": self.id,
            "obfuscatedGaiaId": self.obfuscated_gaia_id,
            "name": self.name,
            "app": self.app,
            "theme": self.theme,
            "capabilities": self.capabilities,
            "cst": self.cst,
            "mdxVersion": self.mdx_version,
            "loungeIdToken": self.lounge_id_token,
            "VER": self.ver,
            "v": self.v,
            "zx": self._generate_zx(),
            "t": self.t,
        }
        if query_type == "initSession":
            params.update(
                {
                    "deviceInfo": json.dumps(self.device_info),
                    "RID": self.rid,
                    "CVER": self.cver,
                }
            )
            self.rid += 1
        elif query_type == "sendMessage":
            if aid and self.aid:
                self.aid = max(aid, self.aid)
            elif aid:
                self.aid = aid
            params.update(
                {
                    "deviceInfo": json.dumps(self.device_info),
                    "SID": self.sid,
                    "RID": self.rid,
                    "AID": self.aid,
                    "gsessionid": self.gsessionid,
                }
            )
            if not aid:
                self.aid += 1
            self.rid += 1
        elif query_type == "rpc":
            params.update(
                {
                    "RID": "rpc",
                    "SID": self.sid,
                    "CI": 0,
                    "AID": self.aid,
                    "gsessionid": self.gsessionid,
                    "TYPE": "xmlhttp",
                }
            )
        return urlencode(params)

    @staticmethod
    def _generate_zx() -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def _generate_rid() -> int:
        return random.randint(41000, 49999)
