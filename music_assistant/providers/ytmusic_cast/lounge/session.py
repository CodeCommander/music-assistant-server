"""
YouTube Lounge API session (screen side).

Port of yt-cast-receiver's Session + RPCConnection: registers a "screen" with
YouTube, keeps a long-poll RPC connection open to receive sender messages, and
POSTs outbound status messages. Senders (the YouTube Music app) discover the
screen via DIAL and connect to it through YouTube's cloud, so this session must
be up before a cast can complete.

Key protocol invariants (mirrored from the reference implementation):
- Outbound messages are strictly serialized: the `ofs` counter and `RID` are
  order-sensitive, so a single worker drains the send queue.
- The RPC long-poll being closed by the remote end is NORMAL; reconnect with
  the current AID. Repeated failures escalate to a full lounge-token refresh.
- On token refresh the AID sequence restarts: SID/gsessionid are cleared, RID
  re-randomized, AID reset, `ofs` reset and the AIDs of any queued outbound
  messages nulled.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aiohttp import ClientTimeout

from music_assistant.providers.ytmusic_cast.lounge.bind_params import BindParams
from music_assistant.providers.ytmusic_cast.lounge.messages import (
    LoungeMessage,
    lounge_screen_disconnected,
    parse_incoming,
)

if TYPE_CHECKING:
    import logging
    from collections.abc import Awaitable, Callable

    from aiohttp import ClientSession

YOUTUBE_BASE_URL = "https://www.youtube.com"
URL_GENERATE_SCREEN_ID = f"{YOUTUBE_BASE_URL}/api/lounge/pairing/generate_screen_id"
URL_GET_LOUNGE_TOKEN_BATCH = f"{YOUTUBE_BASE_URL}/api/lounge/pairing/get_lounge_token_batch"
URL_REGISTER_PAIRING_CODE = f"{YOUTUBE_BASE_URL}/api/lounge/pairing/register_pairing_code"
URL_BIND = f"{YOUTUBE_BASE_URL}/api/lounge/bc/bind"

TOKEN_REFRESH_FALLBACK_MS = 1123200000  # 13 days, same fallback as the reference
RPC_MAX_RETRIES = 3


class LoungeSessionError(Exception):
    """Raised when the lounge session cannot be established or maintained."""


def _form_value(value: Any) -> str:
    """Serialize a payload value the way a JS sender would (bools lowercase)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


class _SendTask:
    """One queued outbound send (a message batch plus its completion future)."""

    def __init__(self, messages: list[LoungeMessage]) -> None:
        self.messages = messages
        self.future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self.retried = False


@dataclass
class ScreenInfo:
    """Static identity of the lounge screen this session represents."""

    name: str
    device_id: str
    app: str = "ytcr"
    # 'cl' = YouTube client theme: what the YouTube Music iOS app actually
    # uses when casting (with topic=music in the DIAL launch)
    theme: str = "cl"
    brand: str = "Music Assistant"
    model: str = "Cast Receiver"


class LoungeSession:
    """One YouTube Lounge screen session bound to a single cast target."""

    def __init__(
        self,
        *,
        http_session: ClientSession,
        screen: ScreenInfo,
        get_screen_id: Callable[[], str | None],
        set_screen_id: Callable[[str], None],
        on_messages: Callable[[list[LoungeMessage]], Awaitable[None]],
        on_terminate: Callable[[Exception], None],
        logger: logging.Logger,
    ) -> None:
        """
        Initialize the session (does not connect until begin() is called).

        :param http_session: Shared aiohttp client session.
        :param screen: Static identity of this screen (name, device id, ...).
        :param get_screen_id: Returns the persisted screen id, if any.
        :param set_screen_id: Persists a newly generated screen id.
        :param on_messages: Async callback invoked with each inbound message batch.
        :param on_terminate: Callback invoked when the session dies irrecoverably.
        :param logger: Provider logger.
        """
        self._http = http_session
        self._screen = screen
        self._get_screen_id = get_screen_id
        self._set_screen_id = set_screen_id
        self._on_messages = on_messages
        self._on_terminate = on_terminate
        self.logger = logger
        self._bind_params = BindParams(
            theme=screen.theme,
            device_id=screen.device_id,
            screen_name=screen.name,
            screen_app=screen.app,
            brand=screen.brand,
            model=screen.model,
        )
        self._screen_id: str | None = None
        self._ofs = 0
        self._status = "stopped"
        self._send_queue: deque[_SendTask] = deque()
        self._send_event = asyncio.Event()
        self._send_paused = False
        self._send_worker_task: asyncio.Task[None] | None = None
        self._rpc_task: asyncio.Task[None] | None = None
        self._token_refresh_task: asyncio.Task[None] | None = None
        self._deferred_sends: dict[str, tuple[asyncio.TimerHandle, _SendTask]] = {}
        self._refresh_lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        """Whether the session is established."""
        return self._status == "running"

    async def begin(self) -> None:
        """Establish the lounge session (register screen, bind, start RPC)."""
        if self._status not in ("stopped", "refreshing"):
            return
        is_refreshing = self._status == "refreshing"
        if not is_refreshing:
            self._status = "starting"
        try:
            await self._establish()
        except Exception as err:
            if is_refreshing:
                await self.end(LoungeSessionError(f"Failed to refresh lounge session: {err}"))
                return
            self._status = "stopped"
            raise LoungeSessionError(f"Failed to establish lounge session: {err}") from err
        if not is_refreshing:
            self._status = "running"
        if self._send_worker_task is None or self._send_worker_task.done():
            self._send_worker_task = asyncio.create_task(self._send_worker())
        self.logger.debug("Lounge session established (screen_id=%s)", self._screen_id)

    async def end(self, error: Exception | None = None) -> None:
        """Tear down the session, notifying senders where possible."""
        if self._status in ("stopped", "stopping"):
            return
        self._status = "stopping"
        self._cancel_token_refresh()
        for handle, task in self._deferred_sends.values():
            handle.cancel()
            if not task.future.done():
                task.future.set_result(False)
        self._deferred_sends.clear()
        if self._rpc_task:
            self._rpc_task.cancel()
            self._rpc_task = None
        if self._send_worker_task:
            self._send_worker_task.cancel()
            self._send_worker_task = None
        for task in self._send_queue:
            if not task.future.done():
                task.future.set_result(False)
        self._send_queue.clear()
        # best-effort goodbye so senders drop the connection promptly
        with contextlib.suppress(Exception):
            await self._post_messages([lounge_screen_disconnected()])
        self._bind_params.reset()
        self._status = "stopped"
        if error:
            self._on_terminate(error)

    async def register_pairing_code(self, code: str) -> None:
        """
        Register a pairing code from a DIAL launch so the sender can connect.

        :param code: The pairingCode from the DIAL launch request body.
        """
        if not self._screen_id:
            raise LoungeSessionError("Cannot register pairing code: no screen id")
        data = {
            "access_type": "permanent",
            "app": self._screen.app,
            "pairing_code": code,
            "screen_id": self._screen_id,
            "screen_name": self._screen.name,
            "device_id": self._bind_params.id,
        }
        async with self._http.post(URL_REGISTER_PAIRING_CODE, data=data) as resp:
            if not resp.ok:
                raise LoungeSessionError(f"register_pairing_code returned {resp.status}")
        self.logger.debug("Pairing code registered")

    async def send(
        self,
        messages: LoungeMessage | list[LoungeMessage],
        defer: tuple[str, float] | None = None,
    ) -> bool:
        """
        Queue message(s) for sending; returns True once actually sent.

        :param messages: Message or batch to send (sent as one request).
        :param defer: Optional (key, seconds): delay sending, replacing any
            pending deferred send with the same key (used for debouncing).
        """
        if isinstance(messages, LoungeMessage):
            messages = [messages]
        task = _SendTask(messages)
        if defer:
            key, interval = defer
            if existing := self._deferred_sends.pop(key, None):
                handle, old_task = existing
                handle.cancel()
                if not old_task.future.done():
                    old_task.future.set_result(False)
            loop = asyncio.get_running_loop()
            handle = loop.call_later(interval, self._enqueue_deferred, key)
            self._deferred_sends[key] = (handle, task)
        else:
            self._send_queue.append(task)
            self._send_event.set()
        return await task.future

    async def _establish(self) -> None:
        """Run the establish sequence: screen id -> token -> bind -> RPC."""
        self._ofs = 0
        screen_id_from_store = False
        if not self._screen_id:
            if stored := self._get_screen_id():
                self._screen_id = stored
                screen_id_from_store = True
            else:
                self._screen_id = await self._generate_screen_id()
        try:
            token = await self._get_lounge_token()
        except Exception:
            if not screen_id_from_store:
                raise
            # stored screen id may have gone stale on YouTube's side: start fresh
            self.logger.warning("Stored screen id rejected; generating a fresh one")
            self._screen_id = await self._generate_screen_id()
            token = await self._get_lounge_token()

        self._bind_params.lounge_id_token = token["loungeToken"]
        self._schedule_token_refresh(token.get("refreshIntervalInMillis") or 0)

        init_messages = await self._init_session()
        forward: list[LoungeMessage] = []
        for message in init_messages:
            self._bind_params.update_with_message(message.name, message.payload, message.aid)
            if message.name not in ("c", "S"):
                forward.append(message)
        if forward:
            await self._on_messages(forward)
        # raises MissingBindDataError if SID/gsessionid did not arrive
        self._bind_params.to_query_string("rpc")

        if self._rpc_task:
            self._rpc_task.cancel()
        self._rpc_task = asyncio.create_task(self._rpc_loop())

    async def _generate_screen_id(self) -> str:
        async with self._http.get(URL_GENERATE_SCREEN_ID) as resp:
            if not resp.ok:
                raise LoungeSessionError(f"generate_screen_id returned {resp.status}")
            screen_id = (await resp.text()).strip()
        self.logger.debug("Generated screen id: %s", screen_id)
        self._set_screen_id(screen_id)
        return screen_id

    async def _get_lounge_token(self) -> dict[str, Any]:
        data = {"screen_ids": self._screen_id}
        async with self._http.post(URL_GET_LOUNGE_TOKEN_BATCH, data=data) as resp:
            if not resp.ok:
                raise LoungeSessionError(f"get_lounge_token_batch returned {resp.status}")
            body = await resp.json(content_type=None)
        try:
            token: dict[str, Any] = body["screens"][0]
        except (KeyError, IndexError, TypeError) as err:
            raise LoungeSessionError(f"Unexpected lounge token response: {body}") from err
        if not token.get("loungeToken"):
            raise LoungeSessionError(f"No loungeToken in response: {token}")
        return token

    async def _init_session(self) -> list[LoungeMessage]:
        url = f"{URL_BIND}?{self._bind_params.to_query_string('initSession')}"
        async with self._http.post(url, data={"count": "0"}) as resp:
            if not resp.ok:
                raise LoungeSessionError(f"init session bind returned {resp.status}")
            body = await resp.text()
        return parse_incoming(body)

    def _schedule_token_refresh(self, interval_ms: int) -> None:
        self._cancel_token_refresh()
        delay = (interval_ms or TOKEN_REFRESH_FALLBACK_MS) / 1000

        async def _refresh_later() -> None:
            await asyncio.sleep(delay)
            await self._refresh_lounge_token()

        self._token_refresh_task = asyncio.create_task(_refresh_later())

    def _cancel_token_refresh(self) -> None:
        if self._token_refresh_task:
            self._token_refresh_task.cancel()
            self._token_refresh_task = None

    async def _refresh_lounge_token(self) -> None:
        """Re-establish the session with a fresh lounge token (AID sequence resets)."""
        async with self._refresh_lock:
            if self._status not in ("running", "refreshing"):
                return
            self.logger.debug("Refreshing lounge token...")
            self._status = "refreshing"
            self._send_paused = True
            self._bind_params.reset()
            old_rpc = self._rpc_task
            self._rpc_task = None
            try:
                await self.begin()
            except Exception as err:
                await self.end(LoungeSessionError(f"Error while refreshing lounge token: {err}"))
                return
            finally:
                if old_rpc:
                    old_rpc.cancel()
            # AID sequence restarted: null AIDs of anything still queued
            for task in self._send_queue:
                for message in task.messages:
                    message.aid = None
            for _, task in self._deferred_sends.values():
                for message in task.messages:
                    message.aid = None
            self._send_paused = False
            self._send_event.set()
            self._status = "running"
            self.logger.debug("Lounge token refreshed")

    async def _rpc_loop(self) -> None:
        """Hold the RPC long-poll open, parsing message frames as they stream in."""
        retries = 0
        while self._status in ("starting", "running", "refreshing"):
            try:
                url = f"{URL_BIND}?{self._bind_params.to_query_string('rpc')}"
                # the long-poll is effectively infinite: no total timeout
                rpc_timeout = ClientTimeout(total=None, sock_connect=30)
                async with self._http.get(url, timeout=rpc_timeout) as resp:
                    if not resp.ok:
                        raise LoungeSessionError(f"RPC bind returned {resp.status}")
                    self.logger.debug("RPC long-poll connected")
                    retries = 0
                    async for line in resp.content:
                        messages = parse_incoming(line.decode("utf-8", errors="replace"))
                        if not messages:
                            continue
                        for message in messages:
                            self._bind_params.update_with_message(
                                message.name, message.payload, message.aid
                            )
                        await self._on_messages(messages)
                # remote closed the long-poll: normal, reconnect with current AID
                self.logger.debug("RPC long-poll ended; reconnecting")
            except asyncio.CancelledError:
                raise
            except Exception as err:
                retries += 1
                self.logger.debug("RPC connection error (%s/%s): %s", retries, RPC_MAX_RETRIES, err)
                if retries > RPC_MAX_RETRIES:
                    # escalate to a token refresh (in a fresh task: refresh cancels us)
                    asyncio.create_task(self._refresh_lounge_token())  # noqa: RUF006
                    return
                await asyncio.sleep(1)

    def _enqueue_deferred(self, key: str) -> None:
        if entry := self._deferred_sends.pop(key, None):
            _, task = entry
            self._send_queue.append(task)
            self._send_event.set()

    async def _send_worker(self) -> None:
        """Drain the send queue one request at a time (ofs/RID are order-sensitive)."""
        while True:
            if not self._send_queue or self._send_paused:
                self._send_event.clear()
                await self._send_event.wait()
                continue
            task = self._send_queue.popleft()
            try:
                await self._post_messages(task.messages)
            except asyncio.CancelledError:
                if not task.future.done():
                    task.future.set_result(False)
                raise
            except Exception as err:
                if task.retried:
                    self.logger.error("Send failed again after token refresh: %s", err)
                    if not task.future.done():
                        task.future.set_result(False)
                    asyncio.create_task(  # noqa: RUF006
                        self.end(LoungeSessionError(f"Send failed after refresh: {err}"))
                    )
                    return
                # retry the same task first after refreshing the token
                self.logger.debug("Send failed (%s); refreshing lounge token to retry", err)
                task.retried = True
                self._send_queue.appendleft(task)
                self._send_paused = True
                asyncio.create_task(self._refresh_lounge_token())  # noqa: RUF006
                continue
            if not task.future.done():
                task.future.set_result(True)

    async def _post_messages(self, messages: list[LoungeMessage]) -> None:
        """POST a message batch to the bind endpoint."""
        aid = messages[0].aid if messages else None
        url = f"{URL_BIND}?{self._bind_params.to_query_string('sendMessage', aid)}"
        payload: dict[str, str] = {"count": str(len(messages)), "ofs": str(self._ofs)}
        for index, message in enumerate(messages):
            prefix = f"req{index}_"
            payload[f"{prefix}_sc"] = message.name
            if isinstance(message.payload, dict):
                for key, value in message.payload.items():
                    payload[prefix + key] = _form_value(value)
        self._ofs += len(messages)
        names = " + ".join(m.name for m in messages)
        self.logger.log(5, "Sending '%s': %s", names, payload)
        async with self._http.post(url, data=payload) as resp:
            if not resp.ok:
                raise LoungeSessionError(f"sendMessage '{names}' returned {resp.status}")
