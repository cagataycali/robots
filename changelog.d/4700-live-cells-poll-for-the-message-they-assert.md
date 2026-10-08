### Tests: twenty-four live cells wait for the message they assert, not a fixed sleep or a full join budget

Seven Zenoh ACL cells slept 1.3 s each around one `put`; they now bracket it
with a put the ACL always admits and end when that one lands. Ten wedged
camera-recorder cells waited out a 0.5 s join that cannot succeed; they now wait
0.1 s. Seven remote-policy cells slept 0.5 s after releasing a parked reply;
they now wait for the parked call to return. Same planted defects caught; the
three files take 18 s instead of 36 s.
