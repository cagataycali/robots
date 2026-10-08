"""mkdocs hook: a page description is HTML-escaped before the theme prints it.

mkdocs runs Jinja without autoescape and Material writes ``page.meta.description``
raw into ``<meta name="description" content="...">`` (and this site's
``partials/content.html`` into the lede). A description holding a ``"`` or a ``<``
therefore broke the document: on ``learn/simulation/urdf.md`` the ``"`` after
``Robot(`` closed the attribute, ``<name>`` was read as a tag, and the rest of the
sentence (``") compiles a robot_descriptions URDF ... refuses.">``) rendered as text
at the top of the page, also after an instant-navigation hop from any other page.

Escaping once here, before any template sees the value, is correct for both
places: a ``content="..."`` attribute and element text both decode entities. The
comparison the lede makes against the rendered ``<h1>`` (``meta.description not in
parts[0]``) also becomes escaped-against-escaped, which is the right one.
"""

from __future__ import annotations

import html


def on_page_context(context, page, config, nav):  # noqa: ANN001 - mkdocs signature
    """mkdocs hook entry point: escape the description for the templates."""
    meta = page.meta or {}
    description = meta.get("description")
    if isinstance(description, str) and description and not meta.get("_description_escaped"):
        meta["description"] = html.escape(description, quote=True)
        meta["_description_escaped"] = True
    return context
