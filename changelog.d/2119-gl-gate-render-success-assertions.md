### Quality: gate every MuJoCo render-success assertion on the shared GL probe

Three test modules asserted `render(...)["status"] == "success"` inline, which
conflates the property under test with a host graphics capability: `render`
reports an error on a headless host without EGL/OSMesa, so on such a host the
suite failed with a bare `'error' != 'success'` naming neither GL nor the
contract. The render assertion is now split into its own `requires_gl` case in
each module, so the assertions that need no GL context keep running everywhere,
and a new guard keeps that the convention for any module added later.
