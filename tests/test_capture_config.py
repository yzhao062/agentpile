"""``config_from`` for both frameworks: instructions, model, tools with full input schemas, handoffs.

Reading a configuration needs only the framework package, not the OpenTelemetry SDK, so these tests skip per
framework and run without the capture extras.
"""

import os

import pytest

import agentpile as ap

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


# -- Pydantic AI ----------------------------------------------------------------------

@pytest.fixture
def pai():
    pytest.importorskip("pydantic_ai")
    from pydantic import BaseModel
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.toolsets import FunctionToolset

    class Verdict(BaseModel):
        answer: str

    class Kit:
        pass

    kit = Kit()
    kit.Agent, kit.Verdict, kit.FunctionToolset = Agent, Verdict, FunctionToolset
    kit.model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("x")]))
    return kit


def test_pydantic_ai_instructions_model_and_tool_schemas(pai):
    agent = pai.Agent(pai.model, name="planner", instructions="Plan trips.", system_prompt="You are a travel planner.",
                      output_type=pai.Verdict)

    @agent.tool_plain(strict=True)
    def add(x: int, y: int = 2) -> int:
        """Add two numbers.

        Args:
            x: the first number
            y: the second number
        """
        return x + y

    @agent.instructions
    def dynamic() -> str:
        return "today"

    run = ap.config_from(agent)
    assert run.kind == "config" and run.id == "planner"
    assert ap.validate(run) == []
    (entry,) = run.agents
    assert entry["name"] == "planner"
    assert entry["instructions"] == "You are a travel planner.\n\nPlan trips."
    assert entry["model"]
    tools = run.tools
    add_tool = tools["planner.add"]
    fn = add_tool["definition"]["function"]
    assert (add_tool["type"], add_tool["definition_format"], add_tool["agent_or_server"]) == ("function", "openai", "planner")
    assert fn["name"] == "add" and fn["description"].startswith("Add two numbers.")
    assert fn["strict"] is True
    assert fn["parameters"]["properties"]["x"] == {"description": "the first number", "type": "integer"}
    assert fn["parameters"]["required"] == ["x"]
    final = tools["planner.final_result"]  # the structured output is a tool the model calls
    assert final["type"] == "output" and final["definition"]["function"]["parameters"]["required"] == ["answer"]
    assert set(entry["tools"]) == {"planner.add", "planner.final_result"}
    assert any("dynamic instruction function" in item for item in run.source["lost_fields"])
    assert run.dataset["framework"] == "pydantic-ai"


def test_pydantic_ai_names_what_it_cannot_read_and_does_not_invent(pai):
    toolset = pai.FunctionToolset()

    @toolset.tool_plain
    def ping() -> str:
        return "pong"

    run = ap.config_from(pai.Agent(pai.model, toolsets=[toolset.prefixed("p_")]))
    (entry,) = run.agents
    assert entry["id"] == "agent" and entry["instructions"] is None and entry["tools"] == []
    assert any("toolsets[" in item for item in run.source["lost_fields"])
    assert all(t["permission"] == "unknown" and t["permission_from"] == "unknown" for t in run.tools.values())


def test_pydantic_ai_several_agents_keep_ids_unique(pai):
    run = ap.config_from([pai.Agent(pai.model, name="same"), pai.Agent(pai.model, name="same")])
    assert [x["id"] for x in run.agents] == ["same", "same#2"]
    assert ap.validate(run) == []


def test_config_record_round_trips_through_json_and_the_config_reader(pai, tmp_path):
    agent = pai.Agent(pai.model, name="planner", instructions="Plan trips.")

    @agent.tool_plain
    def ping() -> str:
        return "pong"

    run = ap.config_from(agent)
    path = tmp_path / "config.json"
    run.to_json(path)
    (again,) = ap.load(path)
    assert again == run


# -- OpenAI Agents SDK -----------------------------------------------------------------

@pytest.fixture
def sdk():
    pytest.importorskip("agents.tracing")
    import agents

    return agents


def make_team(sdk):
    @sdk.function_tool
    def get_weather(city: str) -> str:
        """Return the weather for a city."""
        return f"sunny in {city}"

    billing = sdk.Agent(name="billing", instructions="Handle billing.", handoff_description="Billing questions.",
                        model="gpt-5-mini")
    triage = sdk.Agent(name="triage", instructions="Route requests.", tools=[get_weather], handoffs=[billing],
                       model="gpt-5")
    return triage, billing


def test_openai_agents_tools_schemas_instructions_and_a_handoff(sdk):
    triage, billing = make_team(sdk)
    run = ap.config_from(triage)
    assert run.kind == "config" and run.id == "triage"
    assert ap.validate(run) == []
    first, second = run.agents
    assert (first["id"], first["instructions"], first["model"]) == ("triage", "Route requests.", "gpt-5")
    assert second["instructions"] == "Handle billing."  # the agent a handoff reaches is part of the record
    assert first["handoffs"] == ["billing"]
    assert first["tools"] == ["triage.get_weather", "triage.transfer_to_billing"]

    weather = run.tools["triage.get_weather"]
    fn = weather["definition"]["function"]
    assert (weather["type"], weather["definition_format"]) == ("function", "openai")
    assert fn["name"] == "get_weather" and fn["description"] == "Return the weather for a city."
    assert fn["parameters"]["properties"]["city"]["type"] == "string" and fn["parameters"]["required"] == ["city"]
    assert fn["strict"] is True

    transfer = run.tools["triage.transfer_to_billing"]
    assert transfer["type"] == "handoff" and transfer["agent_or_server"] == "triage"
    assert transfer["definition"]["function"]["name"] == "transfer_to_billing"
    assert "Billing questions." in transfer["definition"]["function"]["description"]
    assert run.dataset["agents"]["billing"]["handoff_description"] == "Billing questions."
    assert run.dataset["framework"] == "openai-agents"


def test_openai_agents_handoff_object_with_its_own_name_and_a_cycle(sdk):
    a = sdk.Agent(name="a", instructions="A.")
    b = sdk.Agent(name="b", instructions="B.", handoffs=[a])
    a.handoffs = [sdk.handoff(b, tool_name_override="escalate", tool_description_override="Escalate to b.")]
    run = ap.config_from(a)
    assert [x["id"] for x in run.agents] == ["a", "b"]  # a cycle is followed once
    assert run.agents[0]["handoffs"] == ["b"] and run.agents[1]["handoffs"] == ["a"]
    assert run.tools["a.escalate"]["definition"]["function"]["description"] == "Escalate to b."


def test_openai_agents_names_what_it_cannot_read(sdk):
    run = ap.config_from(sdk.Agent(name="dyn", instructions=lambda ctx, agent: "computed"))
    assert run.agents[0]["instructions"] is None
    assert any("dynamic instruction function" in item for item in run.source["lost_fields"])
    assert all(t["permission"] == "unknown" for t in run.tools.values())


def test_openai_agents_distinct_agents_with_one_name_keep_distinct_ids(sdk):
    run = ap.config_from([sdk.Agent(name="same"), sdk.Agent(name="same")])
    assert [x["id"] for x in run.agents] == ["same", "same#2"]


def test_agents_of_two_frameworks_make_one_record(pai, sdk):
    run = ap.config_from([pai.Agent(pai.model, name="p"), sdk.Agent(name="s")])
    assert sorted(a["id"] for a in run.agents) == ["p", "s"]
    assert run.source["format"] == "pydantic-ai+openai-agents"
    assert ap.validate(run) == []
