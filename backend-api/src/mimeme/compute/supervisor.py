from __future__ import annotations

import asyncio
import multiprocessing as mp
from multiprocessing.process import BaseProcess
from pathlib import Path

from mimeme.compute.child import run_child
from mimeme.compute.model import ENABLED_ROLES, RESERVED_ROLES, Readiness, Role, RoleStatus
from mimeme.compute.protocol import read_frame, write_frame


class ChildDead(Exception):
    pass


class _Child:
    def __init__(self, role: Role, socket_path: Path) -> None:
        self.role = role
        self.socket_path = socket_path
        self.process: BaseProcess | None = None
        self.lock = asyncio.Lock()
        self.error: str | None = None


class Supervisor:
    def __init__(self, socket_dir: Path, *, start_timeout_s: float = 30.0) -> None:
        self._dir = socket_dir
        self._ctx = mp.get_context("spawn")
        self._start_timeout = start_timeout_s
        self._children: dict[Role, _Child] = {
            role: _Child(role, socket_dir / f"{role}.sock") for role in ENABLED_ROLES
        }

    async def start(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        for role in ENABLED_ROLES:
            await self._spawn(role)

    async def _spawn(self, role: Role) -> None:
        child = self._children[role]
        child.socket_path.unlink(missing_ok=True)
        process = self._ctx.Process(
            target=run_child,
            args=(role, str(child.socket_path)),
            name=f"compute-{role}",
            daemon=True,
        )
        process.start()
        child.process = process
        child.error = None
        try:
            await self._await_socket(child)
        except TimeoutError:
            child.error = "child did not open its socket in time"

    async def _await_socket(self, child: _Child) -> None:
        deadline = asyncio.get_event_loop().time() + self._start_timeout
        while asyncio.get_event_loop().time() < deadline:
            if child.process is not None and not child.process.is_alive():
                raise ChildDead(f"{child.role} exited during startup")
            if child.socket_path.exists():
                return
            await asyncio.sleep(0.05)
        raise TimeoutError(child.role)

    async def call(self, role: Role, request: bytes) -> bytes:
        child = self._children.get(role)
        if child is None:
            raise ChildDead(f"role {role} is not enabled")
        async with child.lock:
            if child.process is None or not child.process.is_alive():
                raise ChildDead(f"{role} child is not running")
            try:
                reader, writer = await asyncio.open_unix_connection(str(child.socket_path))
            except (OSError, ConnectionError) as exc:
                raise ChildDead(f"{role} connect failed: {exc}") from exc
            try:
                await write_frame(writer, request)
                return await read_frame(reader)
            except (asyncio.IncompleteReadError, ConnectionError, OSError) as exc:
                raise ChildDead(f"{role} call failed: {exc}") from exc
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except (OSError, ConnectionError):
                    pass

    async def restart(self, role: Role) -> None:
        child = self._children.get(role)
        if child is None:
            return
        await self._terminate(child)
        await self._spawn(role)

    async def _terminate(self, child: _Child, *, grace_s: float = 5.0) -> None:
        process = child.process
        child.process = None
        if process is None:
            return
        if process.is_alive():
            process.terminate()
            deadline = asyncio.get_event_loop().time() + grace_s
            while process.is_alive() and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.05)
            if process.is_alive():
                process.kill()
        for _ in range(40):
            process.join(0)
            if not process.is_alive():
                break
            await asyncio.sleep(0.05)
        child.socket_path.unlink(missing_ok=True)

    def readiness(self) -> Readiness:
        roles: list[RoleStatus] = []
        ok = True
        for role in ENABLED_ROLES:
            child = self._children[role]
            alive = child.process is not None and child.process.is_alive()
            if alive and child.socket_path.exists() and child.error is None:
                roles.append(RoleStatus(role=role, state="ready"))
            elif alive:
                roles.append(RoleStatus(role=role, state="starting", detail=child.error))
            else:
                ok = False
                roles.append(RoleStatus(role=role, state="failed", detail=child.error))
        for role in RESERVED_ROLES:
            roles.append(RoleStatus(role=role, state="disabled"))
        return Readiness(ok=ok, roles=roles)

    async def close(self) -> None:
        for child in self._children.values():
            await self._terminate(child)
