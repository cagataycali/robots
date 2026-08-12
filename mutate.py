import ast, pathlib, re, subprocess, sys, json
SRC = pathlib.Path("strands_robots/mesh/transport/iot_transport.py")
ORIG = SRC.read_text()
NEW = "tests/mesh/test_iot_client_teardown_failures.py"
OLD_ARM = ["tests/mesh/test_iot_reconnect_client_lifecycle.py",
           "tests/mesh/test_iot_connect_timeout_domain.py",
           "tests/mesh/test_iot_transport_session.py",
           "tests/mesh/test_transport.py"]

def fn_range(src, name):
    t = ast.parse(src)
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n.lineno, n.end_lineno
    raise AssertionError(name)

TIMEOUT_WRAP = '''                try:
                    self._client.stop()
                except Exception as stop_exc:
                    # Same contract as the construction-failure path above: the
                    # connect() has already failed and we return False
                    # regardless, so a stop() error here must not replace that
                    # report with a raise out of a method documented to return
                    # bool. Log at debug and move on.
                    logger.debug("IoT client stop after connect timeout: %s", stop_exc)
'''
CLOSE_HANDLER = '''            except Exception as exc:
                # The two connect()-side teardowns log this; close() is the
                # public one, and it is the only path whose visible report is a
                # success, so a silent swallow leaves "session closed" as the
                # sole record of a client that did not stop. Warn rather than
                # debug: the reference is dropped below either way, so nothing
                # can reach that client afterwards to retry.
                logger.warning(
                    "IoT MQTT client stop() failed during close (thing=%s): %s; "
                    "its IO thread and socket may still be open",
                    self._thing_name,
                    exc,
                )
'''

MUTATIONS = [
    ("M1 timeout stop() unwrapped again (delete the guard)", "connect",
     TIMEOUT_WRAP, "                self._client.stop()\n"),
    ("M2 timeout tolerates but drops the record (structural-blind)", "connect",
     '                    logger.debug("IoT client stop after connect timeout: %s", stop_exc)\n',
     "                    pass\n"),
    ("M3 close() back to a bare swallow", "close",
     CLOSE_HANDLER, "            except Exception:\n                pass\n"),
    ("M4 close() records at debug, not warning", "close",
     '                logger.warning(\n', '                logger.debug(\n'),
    ("M5 close() warning omits the thing name", "close",
     "                    self._thing_name,\n                    exc,\n",
     "                    exc,\n"),
]

def run(paths):
    r = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q", "--no-cov",
                        "-p", "no:randomly", "--tb=no"], capture_output=True, text=True)
    f = re.search(r"(\d+) failed", r.stdout); p = re.search(r"(\d+) passed", r.stdout)
    return (int(f.group(1)) if f else 0), (int(p.group(1)) if p else 0)

rows = []
try:
    base_new, base_old = run([NEW])[1], run(OLD_ARM)[1]
    print(f"unmutated: new={base_new} passed | pre-existing={base_old} passed\n")
    for label, fname, old, new in MUTATIONS:
        lo, hi = fn_range(ORIG, fname)
        region = "".join(ORIG.splitlines(keepends=True)[lo-1:hi])
        in_fn, in_file = region.count(old), ORIG.count(old)
        assert in_fn == 1, f"{label}: in_fn={in_fn}"
        print(f"  anchor scoping {label[:26]:28s} in_fn={in_fn} in_file={in_file}")
        mutated = ORIG.replace(old, new, 1)
        assert mutated != ORIG
        ast.parse(mutated)
        SRC.write_text(mutated)
        nf, np_ = run([NEW]); of, op = run(OLD_ARM)
        rows.append((label, nf, of))
        SRC.write_text(ORIG)
    print()
    print(f"{'mutation':52s} {'new file':>10s} {'pre-existing':>14s}")
    for label, nf, of in rows:
        print(f"{label:52s} {nf:5d} failed {of:8d} failed" + ("   <- BLIND" if of == 0 else ""))
    print(f"\ncaught by the new module: {sum(1 for _,nf,_ in rows if nf)} of {len(rows)}")
    print(f"caught by the pre-existing suite: {sum(1 for _,_,of in rows if of)} of {len(rows)}")
    json.dump({"rows": rows, "base_new": base_new, "base_old": base_old},
              open(f"/tmp/mut-{sys.argv[1]}.json", "w"), indent=2)
finally:
    SRC.write_text(ORIG)
    assert SRC.read_text() == ORIG, "restore failed"
    print("\nsource restored byte-identically")
