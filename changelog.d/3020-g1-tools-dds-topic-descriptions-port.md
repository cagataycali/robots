### Added: `g1_list_dds_topic_descriptions` and `g1_dds_topic_description_admits` verb pair

Ports the third position of the neon bundle's ``TOPIC_CATALOG`` value tuple
(``cagataycali/neon-the-g1/tools/_dds_engine.py``) - the plain-text description
column naming what each of the twenty-two G1 DDS topics decodes at the wire
(e.g. ``"IMU, joints, motors (~1kHz)"`` for ``rt/lowstate`` or
``"Low-level motor cmd (🚨)"`` for ``rt/lowcmd``) - into
``strands_robots.tools.g1``. Twin of the already-shipped
:mod:`~strands_robots.tools.g1.g1_dds_topic_idl_types` (positions one and two
of the same tuple) and :mod:`~strands_robots.tools.g1.g1_dds_topic_categories`
(position four); together the three lookups now name every column of the neon
catalog decidably, so a caller planning a bus-side read or write can resolve
the intent of a topic before a future driver-side wrapper for the neon
``g1_dds_snapshot`` verb dispatches. The import pulls no ``unitree_sdk2py``
submodule (SDK-load-hygiene rule, refs strands-labs/robots#358). Refs
strands-labs/robots#358.
