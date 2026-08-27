# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Render an observation-derived value into a log line without forging a record.

Every provider in this package emits operator diagnostics that quote values read
straight out of a live observation: the observation dict's own key names, a
camera key, a language instruction, a joint-state object. A value carrying a
carriage return or a line feed splits the emitted record in two the moment it
reaches a line-oriented consumer - a ``FileHandler``, ``journalctl``, a shipper
that frames on ``\n`` - and the second half arrives looking like a record this
process never wrote, free to carry its own timestamp, level and logger name.
That is the ``py/log-injection`` class, and it is what :func:`sanitize_log_value`
closes.

Two shapes that look like they close it do not, both measured rather than
assumed (see issue #2853 for the full eight-alert landscape):

* **Keeping the value in the** ``args`` **tuple with a literal** ``%s`` **format
  string.** Better logging practice on its own merits, because a payload
  containing ``%s`` then cannot re-enter interpolation - but the line break
  survives interpolation either way, so the record still splits.
* **Rendering through** ``%r``. This one does neutralize the break
  (``"\n" in ("%r" % "a\nb")`` is ``False``), which is why the three sinks that
  already did it were never the risk - but it states nothing about the other
  five, and it is not visible as a decision at the sinks that lack it.

So the escape is applied here, at one named function, and the call sites read
``%s`` with the value wrapped. What the escape does *not* do is drop content:
each break becomes its own two-character visible escape, which is what
:func:`repr` would have shown, so an operator reading the line still sees every
byte the observation offered and a key whose name genuinely contains a newline
is diagnosable rather than silently reflowed.
"""

# The two characters that can start a new record. A ``\r\n`` pair needs no entry
# of its own: escaping each half independently already renders it as ``\r\n``,
# in order, which is what a reader has to see to know which pair was on the wire.
_LINE_BREAKS: tuple[tuple[str, str], ...] = (
    ("\r", "\\r"),
    ("\n", "\\n"),
)


def sanitize_log_value(value: object) -> str:
    r"""Return ``value`` rendered for a log line with its line breaks escaped.

    The only characters this touches are ``\r\n``, ``\r`` and ``\n``, each
    replaced by its visible two-character escape. Everything else - commas,
    brackets, quotes, the punctuation a joint-state repr or a key list is made
    of - is passed through, because these messages are read by a human
    diagnosing a binding and a broader filter would corrupt the diagnosis.

    Args:
        value: Any object bound for an operator-facing log message. A non-string
            is rendered with :func:`str` first; pass ``repr(value)`` explicitly
            at a site that wants the quoted form (the escape then applies to
            what ``%r`` would have emitted).

    Returns:
        The rendered value, guaranteed to contain no ``\r`` or ``\n``.
    """
    text = value if isinstance(value, str) else str(value)
    for raw, escaped in _LINE_BREAKS:
        text = text.replace(raw, escaped)
    return text
