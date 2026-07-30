# Held reply for PyMammotion PR #181

**Context for the operator (not part of the comment):** on 2026-07-28 Mikey landed
`68e0095` — "fix bug in ble comms thanks @Chorty" — which hand-applies all three
`clear_notification()` calls this PR adds, at the same three sites with the same
semantics. **#181 is superseded by his own commit.** The honest reply is to say so
and close it, rather than defend a duplicate diff. His review question gets
answered in passing. Only #180 (teardown) still has no upstream home.

Suggested comment:

---

Thanks — and I see you've already landed the same fix in `68e0095`, covering all
three sites (sequence gap, checksum failure, exception handler). That's the
substance of this PR, so I'll close it as superseded rather than have you review a
duplicate.

To answer the question anyway, since the thread anchor drifted (GitHub marks it
outdated and pins it to diff position 1, which lands on `if response is None:` — a
line this PR doesn't touch): the only actual removal was in `get_json_string`,
where a manual copy loop became `dict(hash_map)`:

```python
-            jSONObject2: dict[str, int] = {}
-            for key, value in hash_map.items():
-                jSONObject2[key] = value
+            jSONObject2 = dict(hash_map)
```

Same behaviour — a shallow copy, so the caller's dict isn't aliased into the
serialised payload — just written directly. It was unrelated tidying while I was in
the file, and not something that belonged in a fix PR.

The other `-` lines weren't removals: the `except Exception` lines reappear
unchanged with a `# noqa: BLE001` comment, and `type` became `package_type` in
`getTypeValue`/`getPostBytes` to stop shadowing the builtin (every call site in the
package passes those positionally, so no API change).

One piece here might still be worth taking: the three regression tests in
`tests/unit/bluetooth/test_ble_message.py`, which cover the sequence-gap,
checksum-failure, and exception paths independently. Happy to send those as a
test-only PR if useful.

The fix still without an upstream home is #180 — BLE teardown made
failure-atomic. That one releases the client on the post-connect-setup and
write-failure paths, which is what exhausts an ESPHome proxy's connection slots and
produces `out of connection slots` followed by a 120 s cooldown.

---

**Operator note:** `#180`/`#181` shorthand inside a *commit message* pushed to a
fork can create a cross-reference event on the upstream PR — that's why the fork's
merge commit says "PR 180" instead. It does not apply to a comment you post
yourself, so the references above are fine.
