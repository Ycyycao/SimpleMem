"""MemoryArena adapter — run SimpleMem as a memory system inside MemoryArena.

MemoryArena (github.com/ZexueHe/MemoryArena) evaluates memory systems on
continuous, tool-using agent tasks. Its memory contract is two methods:

    add_chunk(chunk: str)            store what just happened
    wrap_user_prompt(prompt: str)    return text to inject into the agent

Two things have to be bridged, and both are about shape rather than substance.

**Input.** MemoryArena hands over one JSON blob per round holding a person's
request, the agent's whole ReAct trace, the plan it produced, and the
environment's judgement. SimpleMem's Stage 1 is built for timestamped dialogue
turns — it resolves pronouns against neighbouring turns and normalises relative
time against the turn's own timestamp. Feeding it one 16k-character blob
defeats both. `_to_dialogues` unpacks the blob back into the turns it was
assembled from: the request, one turn per agent step, one per day of the plan,
one for the judgement. No information is added or dropped, only unpacked.

**Output.** SimpleMem's pipeline ends in `AnswerGenerator`, which exists so the
system can be scored on QA benchmarks like LoCoMo. MemoryArena supplies its own
agent, and that agent owns the tools (flight search, restaurant search) that the
answer actually depends on — the answer is constructed, not recalled. So the
last stage is left out and retrieval output is returned as context instead.

Everything between those two edges — window compression, indexing, intent-aware
retrieval planning, reflection — runs exactly as configured.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from simplemem.core.answer_generator import format_entries
from simplemem.core.models.memory_entry import Dialogue, MemoryEntry
from simplemem.text.system import SimpleMemSystem

# Used when a group's date range cannot be recovered from the base person's
# request. Absolute values are then meaningless but ordering still holds, which
# is what Stage 1's time normalisation and the symbolic index rely on.
_FALLBACK_START = datetime(2022, 1, 1)

_DATE_RANGE = re.compile(
    r"from\s+([A-Z][a-z]+\s+\d{1,2})(?:st|nd|rd|th)?\s+to\s+"
    r"[A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(\d{4})",
)
_DAY_HEADER = re.compile(r"^Day\s+(\d+)\s*:", re.MULTILINE)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:120]


def _parse_trip_start(query: str) -> Optional[datetime]:
    """Recover the trip's first day from a request like

        "... spanning 3 days from March 16th to March 18th, 2022 ..."

    The year sits at the end of the range, so it has to be carried back onto the
    start date. Returns None when the phrasing does not match; the caller falls
    back to a fixed anchor.
    """
    match = _DATE_RANGE.search(query)
    if not match:
        return None
    try:
        return datetime.strptime(f"{match.group(1)} {match.group(2)}", "%B %d %Y")
    except ValueError:
        return None


def _split_days(plan_text: str) -> List[tuple[int, str]]:
    """Split "Day 1: ... Day 2: ..." into (day_number, body) pairs."""
    headers = list(_DAY_HEADER.finditer(plan_text))
    if not headers:
        return []
    days = []
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(plan_text)
        body = plan_text[header.end():end].strip()
        if body:
            days.append((int(header.group(1)), body))
    return days


def _step_text(step: Dict[str, Any]) -> str:
    """Flatten one ReAct step into a single readable turn."""
    parts = []
    thought = step.get("thought")
    if thought:
        parts.append(str(thought))
    for call in step.get("tool_calls") or []:
        parts.append(f"called {call.get('name')} with {json.dumps(call.get('args'), ensure_ascii=False)}")
    for result in step.get("tool_results") or []:
        parts.append(f"{result.get('name')} returned:\n{result.get('result')}")
    if step.get("final_output"):
        parts.append(str(step["final_output"]))
    return "\n".join(p for p in parts if p)


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
        # Where to record what was stored and what came back. The file is named
        # after the user id, which is only known once `bind_user_id` runs.
        self._trace_dir = Path(trace_dir) if trace_dir else None
        self._trace_path: Optional[Path] = None
        # Streaming is off by default here: the response carries no token usage
        # unless it is, which makes the call unaccountable to a host that is
        # measuring cost. It changes no prompt, model or output.
        system_kwargs.setdefault("use_streaming", False)
        self._system_kwargs = system_kwargs
        self._system: Optional[SimpleMemSystem] = None
        self._trip_start: Optional[datetime] = None
        self._next_dialogue_id = 1

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
            table = _slug(self.user_id or "default")
            self._system = SimpleMemSystem(
                api_key=self._api_key,
                model=self._model,
                base_url=self._base_url,
                db_path=self._db_path,
                table_name=table,
                clear_db=True,       # each run starts from an empty store
                **self._system_kwargs,
            )
        return self._system

    # -- trace ---------------------------------------------------------------

    def _trace(self, record: Dict[str, Any]) -> None:
        if self._trace_path is None:
            return
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -- chunk -> dialogues --------------------------------------------------

    def _day_timestamp(self, day: int) -> str:
        start = self._trip_start or _FALLBACK_START
        return (start + timedelta(days=day - 1)).strftime("%Y-%m-%dT%H:%M:%S")

    def _to_dialogues(self, chunk: str) -> List[Dialogue]:
        try:
            data = json.loads(chunk)
        except (TypeError, ValueError):
            # Not the expected envelope — keep it whole rather than guess.
            return [self._dialogue("unknown", chunk, self._day_timestamp(1))]

        name = data.get("name") or "unknown"
        query = data.get("query") or ""

        # The base person's request is the only place the trip dates appear, and
        # it is always the first chunk of a group.
        if self._trip_start is None and query:
            self._trip_start = _parse_trip_start(query)

        day_one = self._day_timestamp(1)
        dialogues: List[Dialogue] = []

        if query:
            dialogues.append(self._dialogue(name, f"{name}'s request: {query}", day_one))

        # The agent's ReAct trace is the bulk of the chunk and holds the
        # evidence behind every choice (prices, ratings, what was rejected).
        for step in data.get("scratchpad") or []:
            text = _step_text(step)
            if text:
                dialogues.append(self._dialogue("Assistant", f"While planning for {name}, {text}", day_one))

        # One turn per travel day, stamped with that day's real date, which is
        # what Stage 1 needs to produce absolute timestamps.
        plan = data.get("final_plan") or ""
        days = _split_days(plan)
        for day, body in days:
            dialogues.append(self._dialogue(
                name, f"{name}'s confirmed plan for day {day}:\n{body}", self._day_timestamp(day)))
        if plan and not days:
            dialogues.append(self._dialogue(name, f"{name}'s confirmed plan:\n{plan}", day_one))

        judgement = data.get("judgement")
        if judgement:
            dialogues.append(self._dialogue(
                "Environment",
                f"Feedback on {name}'s plan: {json.dumps(judgement, ensure_ascii=False)}"
                if not isinstance(judgement, str) else f"Feedback on {name}'s plan: {judgement}",
                self._day_timestamp(1),
            ))

        return dialogues

    def _dialogue(self, speaker: str, content: str, timestamp: str) -> Dialogue:
        dialogue = Dialogue(
            dialogue_id=self._next_dialogue_id,
            speaker=speaker,
            content=content,
            timestamp=timestamp,
        )
        self._next_dialogue_id += 1
        return dialogue

    # -- MemoryArena contract ------------------------------------------------

    def add_chunk(self, chunk: str) -> Dict[str, Any]:
        dialogues = self._to_dialogues(chunk)
        before = len(self.system.get_all_memories())
        self.system.add_dialogues(dialogues, auto_process=True)
        # MemoryArena reads memory again on the next round, so the buffer cannot
        # be left waiting for a window to fill.
        self.system.finalize()
        entries = self.system.get_all_memories()

        result = {
            "dialogues": len(dialogues),
            "entries_added": len(entries) - before,
            "entries_total": len(entries),
        }
        self._trace({
            "op": "add",
            "chunk_chars": len(chunk),
            "chunk": chunk,
            "dialogues": [
                {"speaker": d.speaker, "timestamp": d.timestamp, "content": d.content}
                for d in dialogues
            ],
            "result": result,
            "memories": [e.lossless_restatement for e in entries[before:]],
        })
        return result

    def wrap_user_prompt(self, prompt: str) -> str:
        entries: List[MemoryEntry] = self.system.hybrid_retriever.retrieve(prompt)
        body = format_entries(entries) if entries else ""
        wrapped = "\n".join([
            "<memory_context>",
            body if body else "None",
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
