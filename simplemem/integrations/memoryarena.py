"""MemoryArena adapter — run SimpleMem as a memory system inside MemoryArena.

MemoryArena (github.com/ZexueHe/MemoryArena) evaluates memory systems on
continuous, tool-using agent tasks. Its memory contract is two methods:

    add_chunk(chunk: str)            store what just happened
    wrap_user_prompt(prompt: str)    return text to inject into the agent

Both sides are strings, so this file is a signature adapter and nothing more.
It deliberately does not look inside a chunk. Deciding what a chunk contains and
what is worth keeping from it is Stage 1's job; a host that pre-digests the text
would be answering that question on the method's behalf and the benchmark would
no longer be measuring the method.

Two mappings are not one-to-one and are worth stating:

- `finalize()` runs after every chunk. MemoryArena reads memory again on the
  next round, so the dialogue buffer cannot be left waiting for a window to
  fill. A side effect is that each round becomes its own window, and the
  sliding-window overlap never engages on this benchmark.

- Retrieval stops before `AnswerGenerator`. That stage exists so the system can
  be scored on QA benchmarks, where the answer is recalled from the store. Here
  the answer is constructed instead: it has to satisfy numeric constraints
  against a database, and MemoryArena supplies the tool-using agent that does
  it. So the retrieved entries are returned as context rather than answered.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from simplemem.core.answer_generator import format_entries
from simplemem.core.models.memory_entry import MemoryEntry
from simplemem.text.system import SimpleMemSystem


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:120]


class MemoryArenaMemory:
    """SimpleMem behind MemoryArena's `add_chunk` / `wrap_user_prompt` contract.

    MemoryArena's memory server constructs systems with no arguments and only
    learns the real user id when registering the instance, so construction is
    lazy and `bind_user_id` fixes the id first.
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        db_path: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        trace_dir: Optional[str] = None,
        **system_kwargs: Any,
    ):
        self.user_id = user_id
        self._db_path = db_path
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        # Where to record what went in and what came back. Named after the user
        # id, which is only known once `bind_user_id` runs.
        self._trace_dir = Path(trace_dir) if trace_dir else None
        self._trace_path: Optional[Path] = None
        # Streaming is off: the response carries no token usage unless it is,
        # which makes the call unaccountable to a host measuring cost. Same
        # prompt, same model, same output.
        system_kwargs.setdefault("use_streaming", False)
        self._system_kwargs = system_kwargs
        self._system: Optional[SimpleMemSystem] = None

    # -- lazy construction ---------------------------------------------------

    def bind_user_id(self, user_id: str) -> None:
        if self._system is not None:
            raise RuntimeError("SimpleMem is already built; user_id cannot change")
        self.user_id = user_id
        if self._trace_dir is not None:
            self._trace_path = self._trace_dir / f"{_slug(user_id)}.trace.jsonl"

    @property
    def system(self) -> SimpleMemSystem:
        if self._system is None:
            # One LanceDB directory per user rather than one table per user in a
            # shared directory: MemoryArena runs several cases against this server
            # at once, and keeping their stores apart on disk means no two of them
            # ever touch the same LanceDB manifest.
            slug = _slug(self.user_id or "default")
            db_path = os.path.join(self._db_path, slug) if self._db_path else None
            self._system = SimpleMemSystem(
                api_key=self._api_key,
                model=self._model,
                base_url=self._base_url,
                db_path=db_path,
                table_name=slug,
                clear_db=True,       # each run starts from an empty store
                **self._system_kwargs,
            )
        return self._system

    def _trace(self, record: Dict[str, Any]) -> None:
        if self._trace_path is None:
            return
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -- MemoryArena contract ------------------------------------------------

    def add_chunk(self, chunk: str) -> Dict[str, Any]:
        before = len(self.system.get_all_memories())
        self.system.add_dialogue(speaker="user", content=chunk)
        self.system.finalize()
        entries = self.system.get_all_memories()

        result = {"entries_added": len(entries) - before, "entries_total": len(entries)}
        self._trace({
            "op": "add",
            "chunk_chars": len(chunk),
            "chunk": chunk,
            "result": result,
            "memories": [e.lossless_restatement for e in entries[before:]],
        })
        return result

    def wrap_user_prompt(self, prompt: str) -> str:
        entries: List[MemoryEntry] = self.system.hybrid_retriever.retrieve(prompt)
        wrapped = "\n".join([
            "<memory_context>",
            format_entries(entries) if entries else "None",
            "</memory_context>",
            f"User Prompt: {prompt}",
        ])
        self._trace({
            "op": "search",
            "query_chars": len(prompt),
            "query": prompt,
            "hits": [
                {"memory": e.lossless_restatement, "timestamp": e.timestamp, "topic": e.topic}
                for e in entries
            ],
            "wrapped_chars": len(wrapped),
        })
        return wrapped
