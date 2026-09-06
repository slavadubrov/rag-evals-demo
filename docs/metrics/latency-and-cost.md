# Latency and token usage

`Tracer.stage()` measures elapsed time, including failed operations. `quantiles()`
uses linear interpolation for p50/p95/p99. Retrieval reports include retrieval
latency; generation rows and adapter call logs retain their own durations.

The OpenAI adapter records prompt/completion token usage and the resolved model.
`cost_usd: null` means unknown cost. This demo does not apply a guessed pricing table,
measure streaming TTFT, or infer cache hit rates. Small fixture timing samples are
not service-level evidence. Set provider billing controls separately if needed.

`LLM_TIMEOUT`, `LLM_MAX_TOKENS` and `LLM_MAX_CALLS` apply per LLM adapter instance;
SDK retries are disabled. A failed request still consumes a call-budget slot.
