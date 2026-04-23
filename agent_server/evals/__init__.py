"""Dev evaluation harness for CoMarketer.

LLM-as-judge scoring over five per-request fields:

- ``intent``                  — supervisor intent classification
- ``rewritten_question``      — question handed to Genie
- ``genie_sql``               — SQL produced by Genie
- ``genie_summary``           — analyst-facing narrative
- ``genie_trace_id``          — correlation handle (not judged; logged)

Entry point: ``python -m evals.run_dev_eval --mode {live|replay}``.
"""
