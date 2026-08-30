Ported the ``g1_list_speak_actions_envelope`` and ``g1_speak_action_admits``
lookup tools from the neon bundle: they name the five action strings the
``g1_speak`` verb (``cagataycali/neon-the-g1/tools/g1_speak.py``) admits on
its ``action`` argument today (``start`` / ``stop`` / ``status`` / ``say``
/ ``debug``), with the module-local shape-refusal text a future driver-
side wrapper would quote on an off-set name.
