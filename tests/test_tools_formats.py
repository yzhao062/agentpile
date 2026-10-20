"""Readers for a native MCP tools/list result, an OpenAI tool list and an Anthropic tool list."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import agentpile as ap
from agentpile.formats import tools as tools_mod

FIX = Path(__file__).parent / "fixtures"
TOOLS = FIX / "tools"
CORPORA = Path(__file__).parent / "corpora" / "fixtures"
READERS = ("mcp-tools-list", "openai-tools", "anthropic-tools")


def _json(name):
    return json.loads((TOOLS / name).read_text(encoding="utf-8"))


def _claims(source):
    return [n for n in READERS if ap.readers.get(n).sniff(source)]


# -- registration and detection -----------------------------------------------------------------------

def test_the_readers_are_registered_built_ins():
    assert set(READERS) <= set(ap.readers.names())
    assert ap.readers.get("mcp-tools-list") is tools_mod.mcp
    assert ap.readers.get("openai-tools") is tools_mod.openai
    assert ap.readers.get("anthropic-tools") is tools_mod.anthropic


@pytest.mark.parametrize("name, reader", [
    ("mcp_tools_list.json", "mcp-tools-list"), ("mcp_tools_list_jsonrpc.json", "mcp-tools-list"),
    ("openai_tools.json", "openai-tools"), ("anthropic_tools.json", "anthropic-tools")])
def test_each_native_file_is_claimed_by_its_own_reader_only(name, reader):
    assert _claims(TOOLS / name) == [reader]
    run = next(ap.load(TOOLS / name))  # detection alone, no reader= argument
    assert run.kind == "config" and run.source["format"] == reader and run.source["reader"] == reader


def _existing_fixture_files():
    files = [p for p in FIX.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl") and "tools" not in p.relative_to(FIX).parts]
    files += [p for p in CORPORA.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl", ".gz")]
    return sorted(files)


# SWE-rebench keeps its tool definitions as a bare OpenAI tool list beside the trajectories
NATIVE_TOOL_FILES = {CORPORA / "swe_rebench_openhands" / "tools.json": ["openai-tools"]}


@pytest.mark.parametrize("path", _existing_fixture_files(), ids=lambda p: str(p.relative_to(Path(__file__).parent)))
def test_no_other_fixture_is_claimed(path):
    assert _claims(path) == NATIVE_TOOL_FILES.get(path, [])


def test_the_swe_rebench_tool_sidecar_is_a_native_tool_list():
    path = next(iter(NATIVE_TOOL_FILES))
    run = next(ap.load(path))
    assert run.source["format"] == "openai-tools" and [t["definition"] for t in run.tools.values()] == json.loads(path.read_text())


def test_other_shapes_are_not_claimed():
    atif = json.loads((FIX / "atif_multiagent.json").read_text())
    config = json.loads((FIX / "config_mcp.json").read_text())
    openai_list = _json("openai_tools.json")["tools"]
    shapes = [
        atif, [atif], config, [config], None, "text", 3, {}, [], [{}], {"tools": []}, {"tools": {}},
        [{"role": "user", "content": "x"}],
        {"messages": [{"role": "user", "content": "x"}], "tools": openai_list},  # a message document that lists its tools
        {"model": "m", "tools": openai_list},  # a request body is not a bare tool list
        {"resourceSpans": []},
        [{"name": "no_schema", "description": "a name alone is no evidence of any shape"}],
        [{"name": "n", "inputSchema": {}}, "not an object"],
        {"jsonrpc": "2.0", "id": 1, "result": {"content": []}},
        {"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "x"}},
    ]
    for shape in shapes:
        assert _claims(shape) == [], shape


def test_in_memory_documents_are_claimed_and_loaded():
    mcp_doc = _json("mcp_tools_list.json")
    openai_list, anthropic_list = _json("openai_tools.json")["tools"], _json("anthropic_tools.json")["tools"]
    assert _claims(mcp_doc) == ["mcp-tools-list"] and _claims(mcp_doc["tools"]) == ["mcp-tools-list"]
    assert _claims(openai_list) == ["openai-tools"] and _claims({"tools": openai_list}) == ["openai-tools"]
    assert _claims(anthropic_list) == ["anthropic-tools"] and _claims({"tools": anthropic_list}) == ["anthropic-tools"]
    assert next(ap.load(mcp_doc)).source["record_key"] == "tools-0"
    assert [r.source["format"] for r in ap.load(openai_list)] == ["openai-tools"]  # one list is one record
    assert len(next(ap.load({"tools": anthropic_list})).tools) == 3


def test_a_jsonl_file_and_a_list_of_documents_give_one_record_each(tmp_path):
    docs = [_json("mcp_tools_list.json"), _json("mcp_tools_list_jsonrpc.json")]
    lines = tmp_path / "lists.jsonl"
    lines.write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    runs = list(ap.load(lines))
    assert [r.source["record_key"] for r in runs] == ["lists-0", "lists-1"] and [len(r.tools) for r in runs] == [5, 1]
    many = tmp_path / "many.json"
    anthropic = _json("anthropic_tools.json")["tools"]
    many.write_text(json.dumps([{"tools": anthropic}, {"tools": anthropic[:1]}]))
    runs = list(ap.load(many, reader="anthropic-tools"))
    assert [len(r.tools) for r in runs] == [3, 1]


def test_a_bare_list_file_is_detected_by_its_shape(tmp_path):
    for name, reader in (("mcp_tools_list.json", "mcp-tools-list"), ("openai_tools.json", "openai-tools"),
                         ("anthropic_tools.json", "anthropic-tools")):
        bare = tmp_path / f"bare-{name}"
        bare.write_text(json.dumps(_json(name)["tools"]))
        assert _claims(bare) == [reader]
        assert next(ap.load(bare)).record == next(ap.load(bare, reader=reader)).record
        run = next(ap.load(bare))
        assert run.source["format"] == reader and run.source["record_key"] == f"bare-{name.split('.')[0]}"
        assert [t["definition"] for t in run.tools.values()] == _json(name)["tools"]
    # a JSONL line is one document, so a bare list on a line is claimed
    line = tmp_path / "lines.jsonl"
    line.write_text(json.dumps(_json("anthropic_tools.json")["tools"]) + "\n")
    assert _claims(line) == ["anthropic-tools"] and len(next(ap.load(line)).tools) == 3


# -- MCP --------------------------------------------------------------------------------------------

def test_mcp_tools_keep_their_full_definitions_and_permissions_from_annotations():
    raw = _json("mcp_tools_list.json")
    run = next(ap.load(TOOLS / "mcp_tools_list.json"))
    assert ap.validate(run) == [] and run.kind == "config" and run.steps == [] and list(run.events()) == []
    assert list(run.tools) == [t["name"] for t in raw["tools"]]
    for tid, definition in zip(run.tools, raw["tools"]):
        entry = run.tools[tid]
        assert entry["definition"] == definition and entry["definition"] is not definition  # a copy, verbatim
        assert (entry["name"], entry["definition_format"], entry["atif_index"]) == (definition["name"], "mcp", None)
        assert (entry["agent_or_server"], entry["type"]) == (None, None)  # nothing the source does not say
    perms = {tid: (e["permission"], e["permission_from"]) for tid, e in run.tools.items()}
    assert perms == {"read_file": ("read", "annotation"), "delete_file": ("write", "annotation"),
                     "fetch_url": ("unknown", "unknown"),  # annotations that state neither hint
                     "echo": ("unknown", "unknown"),  # no annotations
                     "contradiction": ("unknown", "unknown")}  # read-only and destructive together
    assert run.tools["read_file"]["definition"]["outputSchema"]["type"] == "object"
    assert run.dataset == {"tool_list_members": {"nextCursor": "page-2"}}
    assert run.source["record_key"] == "mcp_tools_list" and run.source["file"] == "mcp_tools_list.json"
    assert run.agents == [] and run.labels == []


def test_mcp_jsonrpc_response_keeps_its_envelope():
    run = next(ap.load(TOOLS / "mcp_tools_list_jsonrpc.json"))
    assert ap.validate(run) == [] and list(run.tools) == ["list_dir"]
    assert run.tools["list_dir"]["permission"] == "read"
    assert run.dataset == {"jsonrpc": {"jsonrpc": "2.0", "id": 7}}


def test_the_mcp_permission_follows_the_hints_the_source_states():
    def perm(annotations):
        tool = {"name": "t", "inputSchema": {}}
        if annotations is not None:
            tool["annotations"] = annotations
        run = next(ap.load({"tools": [tool]}))
        return run.tools["t"]["permission"], run.tools["t"]["permission_from"]

    assert perm({"readOnlyHint": True}) == ("read", "annotation")
    assert perm({"readOnlyHint": False}) == ("write", "annotation")
    assert perm({"destructiveHint": True}) == ("write", "annotation")
    assert perm({"destructiveHint": False}) == ("unknown", "unknown")  # not read-only is not stated
    assert perm({"readOnlyHint": "yes"}) == ("unknown", "unknown")  # only a JSON boolean counts
    assert perm({"readOnlyHint": True, "destructiveHint": True}) == ("unknown", "unknown")
    assert perm({}) == ("unknown", "unknown") and perm(None) == ("unknown", "unknown")
    assert perm("read-only") == ("unknown", "unknown")


def test_a_bare_mcp_tool_list_and_the_fixture_config_shape():
    run = next(ap.load(_json("mcp_tools_list.json")["tools"]))
    assert len(run.tools) == 5 and run.dataset == {}
    reference = next(ap.load(FIX / "config_mcp.json"))
    assert set(run.record) - {"dataset"} == set(reference.record) - {"dataset"}  # the same top-level members
    assert set(run.tools["read_file"]) == set(reference.tools["fs.read_file"])


# -- OpenAI ------------------------------------------------------------------------------------------

def test_openai_tools_keep_chat_and_flat_forms_and_repeated_names():
    raw = _json("openai_tools.json")["tools"]
    run = next(ap.load(TOOLS / "openai_tools.json"))
    assert ap.validate(run) == []
    assert list(run.tools) == ["get_weather", "get_time", "get_weather#2", "tool#3"]
    assert [run.tools[t]["definition"] for t in run.tools] == raw
    assert [(e["name"], e["type"]) for e in run.tools.values()] == [
        ("get_weather", "function"), ("get_time", "function"), ("get_weather", "function"), ("", "web_search")]
    assert {e["definition_format"] for e in run.tools.values()} == {"openai"}
    assert {(e["permission"], e["permission_from"]) for e in run.tools.values()} == {("unknown", "unknown")}
    assert run.tools["get_weather"]["definition"]["function"]["strict"] is True  # vendor members survive


def test_the_legacy_functions_form_is_an_openai_list():
    legacy = [{"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}]
    assert _claims(legacy) == ["openai-tools"]
    run = next(ap.load(legacy))
    assert list(run.tools) == ["f"] and run.tools["f"]["type"] is None and run.tools["f"]["definition"] == legacy[0]


# -- Anthropic ---------------------------------------------------------------------------------------

def test_anthropic_tools_keep_custom_and_server_tools_whole():
    raw = _json("anthropic_tools.json")["tools"]
    run = next(ap.load(TOOLS / "anthropic_tools.json"))
    assert ap.validate(run) == []
    assert list(run.tools) == ["get_stock_price", "get_news", "bash"]
    assert [run.tools[t]["definition"] for t in run.tools] == raw
    assert [e["type"] for e in run.tools.values()] == [None, "custom", "bash_20250124"]
    assert run.tools["get_stock_price"]["definition"]["cache_control"] == {"type": "ephemeral"}
    assert {(e["definition_format"], e["permission"], e["permission_from"]) for e in run.tools.values()} == {
        ("anthropic", "unknown", "unknown")}


# -- explicit reading, errors and round trips ---------------------------------------------------------

def test_an_explicit_reader_is_more_lenient_than_detection():
    nameless = [{"description": "no name"}, {"name": "ok"}]
    assert _claims(nameless) == []
    run = next(ap.load(nameless, reader="anthropic-tools"))
    assert list(run.tools) == ["tool#0", "ok"] and run.tools["tool#0"]["name"] == ""
    assert next(ap.load({"tools": []}, reader="mcp-tools-list")).tools == {}
    body = {"model": "m", "tools": _json("openai_tools.json")["tools"]}
    run = next(ap.load(body, reader="openai-tools"))
    assert len(run.tools) == 4 and run.dataset == {"tool_list_members": {"model": "m"}}


def test_a_document_without_a_tool_list_is_refused():
    with pytest.raises(ValueError, match="not a mcp-tools-list document"):
        list(ap.load({"content": []}, reader="mcp-tools-list"))
    with pytest.raises(ValueError, match="tool definition 1 is not a JSON object"):
        list(ap.load([{"name": "a"}, "b"], reader="anthropic-tools"))
    with pytest.raises(TypeError, match="takes no options"):
        list(ap.load(TOOLS / "mcp_tools_list.json", typo_option=1))


def test_config_records_round_trip_through_json(tmp_path):
    for name in ("mcp_tools_list.json", "mcp_tools_list_jsonrpc.json", "openai_tools.json", "anthropic_tools.json"):
        run = next(ap.load(TOOLS / name))
        out = tmp_path / f"out-{name}"
        ap.writers.get("json").write(run, out)
        assert next(ap.load(out)) == run  # read back as an agentpile-config record
        assert ap.Run(json.loads(run.to_json())) == run


def test_the_readers_need_only_the_standard_library_and_python_39_syntax():
    script = (
        "import sys\n"
        "import agentpile as ap\n"
        f"for name in ('mcp_tools_list.json', 'openai_tools.json', 'anthropic_tools.json'):\n"
        f"    run = next(ap.load(r'{TOOLS}' + '/' + name))\n"
        "    assert ap.validate(run) == []\n"
        "third = sorted(m for m in ('yaml', 'numpy', 'pandas', 'pyarrow', 'pydantic', 'opentelemetry', 'huggingface_hub')\n"
        "               if m in sys.modules)\n"
        "print(','.join(third))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""
