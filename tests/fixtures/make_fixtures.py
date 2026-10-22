"""Write the synthetic agentpile test fixtures (no real dataset content)."""
import json
from pathlib import Path

OUT = Path(__file__).parent


def fn(name, desc, props):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}}


defs = [
    fn("search_flights", "Search flights between two airports.", {"origin": {"type": "string"}, "dest": {"type": "string"}}),
    fn("read_calendar", "Read the user's calendar for one date.", {"date": {"type": "string"}}),
    fn("book_flight", "Book a flight by offer id.", {"offer_id": {"type": "integer"}}),
]

traj = {
    "schema_version": "ATIF-v1.7",
    "session_id": "fx-1",
    "trajectory_id": "fx-1",
    "agent": {"name": "synthetic-travel", "version": "0", "model_name": "fake-model", "tool_definitions": defs},
    "steps": [
        {"step_id": 1, "source": "system", "message": "You plan and book trips."},
        {"step_id": 2, "source": "user", "message": "Book me a flight from LAX to SFO on 2026-11-02.",
         "extra": {"agentpile": {"source_ref": 0}}},
        {"step_id": 3, "source": "agent", "message": "I will search flights and check your calendar.",
         "tool_calls": [
             {"tool_call_id": "c1", "function_name": "search_flights", "arguments": {"origin": "LAX", "dest": "SFO"}},
             {"tool_call_id": "c2", "function_name": "read_calendar", "arguments": {"date": "2026-11-02"}}],
         "observation": {"results": [
             {"source_call_id": "c2", "content": "No events."},
             {"source_call_id": "c1", "content": "[{\"offer_id\": 7, \"depart\": \"09:00\"}]"}]},
         "extra": {"agentpile": {"actor": "planner", "source_ref": 1,
                                 "calls": {"c1": {"tool_id": "t.search"}, "c2": {"tool_id": "t.calendar"}},
                                 "result_refs": [3, 2]}}},
        {"step_id": 4, "source": "agent", "message": "",
         "tool_calls": [{"tool_call_id": "c3", "function_name": "book_flight", "arguments": {}}],
         "observation": {"results": [{"source_call_id": "c3", "content": "error: malformed arguments"}]},
         "extra": {"agentpile": {"actor": "booker", "message_state": "null", "source_ref": 4,
                                 "calls": {"c3": {"tool_id": "t.book", "arguments_raw": "{'offer_id': 7",
                                                  "arguments_state": "unparsed",
                                                  "error": {"type": "tool_error", "message": "malformed arguments"}}},
                                 "result_refs": [5]}}},
        {"step_id": 5, "source": "agent", "message": "I could not complete the booking.",
         "extra": {"agentpile": {"actor": "booker", "source_ref": 6}}},
    ],
    "extra": {"agentpile": {
        "version": "0",
        "source": {"format": "synthetic", "dataset": None, "revision": None, "record_key": "fx-1",
                   "reader": "fixture", "reader_version": "0", "lost_fields": []},
        "task": {"id": "travel-1", "text": "Book a flight from LAX to SFO on 2026-11-02."},
        "outcome": {"success": False, "score": 0.0, "raw": {"booked": False}},
        "labels": [{"name": "mistake_step", "target": {"kind": "step", "address": {"trajectory": "fx-1", "step": 4}},
                    "value": True, "label_source": "fixture", "source_ref": "mistake_step=4"},
                   {"name": "mistake_agent", "target": {"kind": "agent", "address": {"agent": "booker"}},
                    "value": True, "label_source": "fixture"}],
        "agents": [{"id": "planner", "name": "Planner", "instructions": "Plan the trip.", "model": "fake-model",
                    "tools": ["t.search", "t.calendar"], "handoffs": ["booker"]},
                   {"id": "booker", "name": "Booker", "instructions": None, "model": "fake-model",
                    "tools": ["t.book"], "handoffs": []}],
        "tools": {
            "t.search": {"name": "search_flights", "agent_or_server": "planner", "type": "search",
                         "definition": defs[0], "definition_format": "openai", "atif_index": 0,
                         "permission": "read", "permission_from": "heuristic"},
            "t.calendar": {"name": "read_calendar", "agent_or_server": "planner", "type": "calendar",
                           "definition": defs[1], "definition_format": "openai", "atif_index": 1,
                           "permission": "read", "permission_from": "heuristic"},
            "t.book": {"name": "book_flight", "agent_or_server": "booker", "type": "payment",
                       "definition": defs[2], "definition_format": "openai", "atif_index": 2,
                       "permission": "write", "permission_from": "declared"}},
        "dataset": {"fixture_note": "synthetic, written for tests"},
    }},
}

config = {
    "schema": "agentpile-config/0",
    "source": {"format": "mcp-tools-list", "dataset": None, "record_key": "fx-config-1", "reader": "fixture",
               "reader_version": "0", "lost_fields": []},
    "task": {"id": None, "text": None},
    "agents": [{"id": "assistant", "name": "assistant", "instructions": "Help with files.", "model": None,
                "tools": ["fs.read_file", "fs.delete_file"], "handoffs": []}],
    "tools": {
        "fs.read_file": {"name": "read_file", "agent_or_server": "fs", "type": "filesystem",
                         "definition": {"name": "read_file", "description": "Read a file.",
                                        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
                                        "annotations": {"readOnlyHint": True}},
                         "definition_format": "mcp", "atif_index": None,
                         "permission": "read", "permission_from": "annotation"},
        "fs.delete_file": {"name": "delete_file", "agent_or_server": "fs", "type": "filesystem",
                           "definition": {"name": "delete_file", "description": "Delete a file.",
                                          "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
                                          "annotations": {"readOnlyHint": False, "destructiveHint": True}},
                           "definition_format": "mcp", "atif_index": None,
                           "permission": "write", "permission_from": "annotation"}},
    "labels": [{"name": "excess_permission", "target": {"kind": "agent", "address": {"agent": "assistant"}},
                "value": True, "label_source": "fixture"}],
    "dataset": {"server": "fs"},
}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "atif_multiagent.json").write_text(json.dumps(traj, indent=2) + "\n", encoding="utf-8", newline="\n")
(OUT / "config_mcp.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n")
second = json.loads(json.dumps(traj))
second["session_id"] = second["trajectory_id"] = "fx-2"
second["extra"]["agentpile"]["labels"] = []
with (OUT / "atif_two.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
    for t in (traj, second):
        fh.write(json.dumps(t) + "\n")
(OUT / "not_a_trace.json").write_text(json.dumps({"hello": "world"}) + "\n", encoding="utf-8", newline="\n")
print("fixtures written")
