"""
rag package.

Logging note
------------
This package logs internal events (e.g. "Agent hit max_iterations
without finishing") via the standard `logging` module, at module-level
loggers such as `rag.agent.agent`. It intentionally does NOT call
`logging.basicConfig()` or attach a `StreamHandler` itself — that's the
calling application's decision to make (where to send logs, what
format, what level).

If your application's console/chat loop does something like:

    logging.basicConfig(level=logging.WARNING)  # or similar
    answer = agent.run(question)
    print("Assistant:", answer.answer)

...then a warning logged during `agent.run()` (e.g. the max_iterations
fallback notice) is written to the same stream right before the
`print()` call, which can visually read as if it were *part of* the
assistant's reply. It isn't — `context.answer` never contains log
text. To keep the two separate, send this package's logs somewhere
other than the same stream you print user-facing replies to, e.g.:

    logging.getLogger("rag").addHandler(logging.FileHandler("rag.log"))
    logging.getLogger("rag").propagate = False

A NullHandler is attached below (standard practice for libraries) so
nothing is printed by default if the host application hasn't
configured logging at all.
"""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
