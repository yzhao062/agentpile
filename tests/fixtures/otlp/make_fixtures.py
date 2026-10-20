"""Write the OTLP JSON test fixtures from scripted runs of two agent frameworks.

The runs use fake models only (no network, no API key, no dataset content). The
spans are captured with a local tracer provider, encoded by the official OTLP
encoder, and written as OTLP/JSON (hex ids, integer enums, int64 as strings).

Needs the framework packages, which the tests do not. Install them into a
scratch directory and point FIXTURE_LIBS at it (os.pathsep-separated):

    pip install --target LIBS pydantic-ai-slim==2.58.0 openai-agents==0.24.0 \
        openinference-instrumentation-openai-agents==2.5.3 opentelemetry-sdk \
        opentelemetry-proto opentelemetry-exporter-otlp-proto-common==1.45.1
    FIXTURE_LIBS=LIBS python make_fixtures.py

Python 3.11 or later is required by the frameworks. Ids and times differ on every
run, so the saved files are the fixtures; the script documents how they came about.
"""
import base64
import hashlib
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path

OUT = Path(__file__).parent
for lib in filter(None, os.environ.get("FIXTURE_LIBS", "").split(os.pathsep)):
    sys.path.insert(0, lib)
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "0"
os.environ.setdefault("OPENAI_API_KEY", "sk-not-used")  # the SDK client is never called

from google.protobuf.json_format import MessageToDict  # noqa: E402
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans  # noqa: E402
from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402


def local_provider(service):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def to_otlp_json(spans):
    """ExportTraceServiceRequest as OTLP/JSON: hex ids and integer enums, as the OTLP spec requires."""
    doc = MessageToDict(encode_spans(spans), use_integers_for_enums=True)

    def hexify(obj):
        for key in ("traceId", "spanId", "parentSpanId"):
            if obj.get(key):
                obj[key] = base64.b64decode(obj[key]).hex()

    for rs in doc.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for span in ss.get("spans", []):
                hexify(span)
                for link in span.get("links", []):
                    hexify(link)
    return doc


# -- Pydantic AI 2.58.0 ---------------------------------------------------------

def pydantic_ai_kit(service):
    from opentelemetry.sdk.metrics import MeterProvider
    from pydantic_ai import Agent, InstrumentationSettings
    from pydantic_ai.capabilities import Instrumentation
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    provider, exporter = local_provider(service)
    settings = InstrumentationSettings(tracer_provider=provider, meter_provider=MeterProvider(), version=5)

    def make_agent(name, instructions, script, model_name):
        return Agent(FunctionModel(script, model_name=model_name), capabilities=[Instrumentation(settings=settings)],
                     name=name, instructions=instructions)

    return make_agent, exporter, ModelResponse, TextPart, ToolCallPart


def pydantic_ai_spans():
    make_agent, exporter, ModelResponse, TextPart, ToolCallPart = pydantic_ai_kit("fixture-pydantic-ai")
    turns = {"n": 0}

    def script(messages, info):
        turns["n"] += 1
        if turns["n"] == 1:  # one model turn that calls both tools at once
            return ModelResponse(parts=[
                TextPart("Let me check both."),
                ToolCallPart("get_weather", {"city": "Paris"}, tool_call_id="call_weather_1"),
                ToolCallPart("get_time", {"zone": "UTC"}, tool_call_id="call_time_1")])
        if turns["n"] == 2:
            return ModelResponse(parts=[TextPart("It is sunny in Paris and 12:00 UTC.")])
        return ModelResponse(parts=[TextPart("Anytime.")])

    agent = make_agent("weather_agent", "You answer questions about weather and time.", script, "fake-fn")

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Weather for a city."""
        return f"sunny in {city}"

    @agent.tool_plain
    def get_time(zone: str) -> dict:
        """Current time in a zone."""
        return {"zone": zone, "time": "12:00"}

    agent.run_sync("What is the weather in Paris and what time is it?")
    agent.run_sync("Thanks.")  # a second trace: no tools
    return list(exporter.get_finished_spans())


def pydantic_ai_delegation_spans():
    """A tool that runs a second agent: its invoke_agent span nests under the first agent's execute_tool span."""
    from pydantic_ai import RunContext

    make_agent, exporter, ModelResponse, TextPart, ToolCallPart = pydantic_ai_kit("fixture-pydantic-ai-delegation")
    helper = make_agent("helper_agent", "Help.", lambda m, i: ModelResponse(parts=[TextPart("42")]), "fake-helper")
    turns = {"n": 0}

    def boss_script(messages, info):
        turns["n"] += 1
        if turns["n"] == 1:
            return ModelResponse(parts=[ToolCallPart("ask_helper", {"question": "answer?"}, tool_call_id="call_ask_1")])
        return ModelResponse(parts=[TextPart("The answer is 42.")])

    boss = make_agent("boss_agent", "Delegate.", boss_script, "fake-boss")

    @boss.tool
    async def ask_helper(ctx: RunContext[None], question: str) -> str:
        """Ask the helper agent."""
        result = await helper.run(question, usage=ctx.usage)
        return result.output

    boss.run_sync("What is the answer?")
    return list(exporter.get_finished_spans())


# -- OpenAI Agents SDK 0.24.0 through openinference-instrumentation-openai-agents 2.5.3 ---------

def openai_agents_kit():
    from types import SimpleNamespace

    from agents import Agent, Runner, function_tool
    from agents.items import ModelResponse
    from agents.models.interface import Model
    from agents.models.openai_responses import Converter
    from agents.tracing import response_span, set_trace_processors
    from agents.usage import Usage
    from openai.types.responses import (FunctionTool, Response, ResponseFunctionToolCall, ResponseOutputMessage,
                                        ResponseOutputText, ResponseUsage)
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
    from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

    class ScriptedModel(Model):
        """A fake model that records spans the way OpenAIResponsesModel does: one response span per call."""

        def __init__(self, script):
            self.script = list(script)
            self.n = 0

        async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs,
                               tracing, previous_response_id=None, conversation_id=None, prompt=None):
            out = self.script[self.n]
            self.n += 1
            declared = [FunctionTool(**t) for t in Converter.convert_tools(tools, handoffs).tools
                        if t.get("type") == "function"]
            usage = ResponseUsage(input_tokens=30 * self.n, output_tokens=10, total_tokens=30 * self.n + 10,
                                  input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
                                  output_tokens_details=OutputTokensDetails(reasoning_tokens=0))
            with response_span(disabled=tracing.is_disabled()) as sr:
                response = Response(id=f"resp_{id(self)}_{self.n}", created_at=0.0, model="fake-model",
                                    object="response", output=out, parallel_tool_calls=False, tool_choice="auto",
                                    tools=declared, instructions=system_instructions, usage=usage)
                sr.span_data.response = response
                sr.span_data.input = input
            return ModelResponse(output=out, usage=Usage(), response_id=response.id)

        def stream_response(self, *args, **kwargs):
            raise NotImplementedError

    def call(name, args, call_id):
        return ResponseFunctionToolCall(type="function_call", name=name, arguments=json.dumps(args),
                                        call_id=call_id, id="fc_" + call_id)

    def message(text, message_id):
        return ResponseOutputMessage(id=message_id, role="assistant", status="completed", type="message",
                                     content=[ResponseOutputText(type="output_text", text=text, annotations=[])])

    def traced(service, run):
        """Run ``run()`` with the instrumentor on a local provider; the SDK's processor list is cleared before and after."""
        provider, exporter = local_provider(service)
        # The instrumentor replaces the SDK's process-wide processor list unless told not to; clear the
        # default exporter first and leave nothing behind (see tested-versions.md of stage 0).
        set_trace_processors([])
        instrumentor = OpenAIAgentsInstrumentor()
        instrumentor.instrument(tracer_provider=provider, exclusive_processor=False)
        try:
            run()
        finally:
            instrumentor.uninstrument()
            set_trace_processors([])
        return list(exporter.get_finished_spans())

    return SimpleNamespace(Agent=Agent, Runner=Runner, function_tool=function_tool, ScriptedModel=ScriptedModel,
                           call=call, message=message, traced=traced)


def openai_agents_spans():
    k = openai_agents_kit()

    @k.function_tool
    def lookup_order(order_id: str) -> str:
        """Look up the status of an order."""
        return f"order {order_id}: shipped"

    refunds = k.Agent(name="Refunds", instructions="Handle refunds.",
                      model=k.ScriptedModel([[k.message("Refund started for order A1.", "msg_3")]]))
    triage = k.Agent(name="Triage", instructions="Route the customer.", tools=[lookup_order], handoffs=[refunds],
                     model=k.ScriptedModel([[k.message("Checking.", "msg_1"), k.call("lookup_order", {"order_id": "A1"}, "call_lookup_1")],
                                            [k.call("transfer_to_refunds", {}, "call_handoff_1")]]))
    return k.traced("fixture-openai-agents", lambda: k.Runner.run_sync(triage, "Where is order A1? I want a refund."))


def openai_agents_as_tool_spans():
    """An agent used as a tool of another agent: the inner agent span nests under the outer function span."""
    k = openai_agents_kit()
    helper = k.Agent(name="Helper", instructions="Help.", model=k.ScriptedModel([[k.message("42", "msg_h")]]))
    boss = k.Agent(name="Boss", instructions="Delegate.",
                   tools=[helper.as_tool(tool_name="ask_helper", tool_description="Ask the helper")],
                   model=k.ScriptedModel([[k.call("ask_helper", {"input": "answer?"}, "call_ask_1")],
                                          [k.message("The answer is 42.", "msg_b")]]))
    return k.traced("fixture-openai-agents-as-tool", lambda: k.Runner.run_sync(boss, "What is the answer?"))


def write_json(path, doc):
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path, spans, per_line):
    """The OTel file exporter writes one ExportTraceServiceRequest per line; a batch holds a few spans."""
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for i in range(0, len(spans), per_line):
            fh.write(json.dumps(to_otlp_json(spans[i:i + per_line])) + "\n")


def main():
    outputs = {
        "pydantic_ai_2.58.0.json": ("json", pydantic_ai_spans()),
        "pydantic_ai_2.58.0_delegation.json": ("json", pydantic_ai_delegation_spans()),
        "openai_agents_0.24.0_oi_2.5.3.jsonl": ("jsonl", openai_agents_spans()),
        "openai_agents_0.24.0_oi_2.5.3_as_tool.json": ("json", openai_agents_as_tool_spans()),
    }
    for name, (how, spans) in outputs.items():
        if how == "json":
            write_json(OUT / name, to_otlp_json(spans))
        else:
            write_jsonl(OUT / name, spans, per_line=4)
    versions = {n: md.version(n) for n in ["pydantic-ai-slim", "openai-agents",
                                           "openinference-instrumentation-openai-agents", "opentelemetry-sdk",
                                           "opentelemetry-proto", "opentelemetry-exporter-otlp-proto-common"]}
    versions["python"] = sys.version.split()[0]
    files = {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in outputs}
    write_json(OUT / "versions.json", {"generated_with": versions, "sha256": files,
                                       "note": "fake models only; no network, no API key, no dataset content"})
    print("wrote", {name: len(spans) for name, (_, spans) in outputs.items()})


if __name__ == "__main__":
    main()
