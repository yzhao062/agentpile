# OTel GenAI Value Schemas

These six JSON schemas are copied unchanged from [open-telemetry/semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai) at commit `9ffc7e00c639be7712e544e3e54331e885dccf84`, from `model/gen-ai/`. That repository is licensed under Apache-2.0.

`tests/test_otlp_conventions.py` validates the GenAI message, tool-call and tool-definition attributes that the OTLP writer generates against them. The writer follows this revision; moving to a newer one means replacing these files and rerunning that test file.

To refresh a file:

```bash
curl -sfL https://raw.githubusercontent.com/open-telemetry/semantic-conventions-genai/<commit>/model/gen-ai/<name>.json -o <name>.json
```
