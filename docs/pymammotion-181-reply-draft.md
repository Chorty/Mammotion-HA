Thanks for looking at this. The comment anchor drifted (GitHub reports the thread
as outdated and pins it to diff position 1), so line 355 lands on
`if response is None:` / `return -1`, which this PR does not touch. Let me cover
every `-` line in the diff so I answer the right one.

The only actual code removal is in `get_json_string`:

```python
-            jSONObject2: dict[str, int] = {}
-            for key, value in hash_map.items():
-                jSONObject2[key] = value
+            jSONObject2 = dict(hash_map)
```

Same behaviour — a shallow copy so the caller's dict is not aliased into the
serialised payload — just written directly. It was drive-by tidying while I was
in the file; happy to drop it and keep the explicit loop if you would rather the
diff stay strictly on the reassembly fix.

The remaining `-` lines are not removals:

- `except Exception ...` lines reappear unchanged with a `# noqa: BLE001`
  comment explaining why the broad catch is intentional.
- `type: int` -> `package_type: int` in `getTypeValue`/`getPostBytes` avoids
  shadowing the builtin. Every call site in the package passes these
  positionally, so there is no API change.

Nothing was removed from the `parseNotification` parsing path. The fix is purely
additive: `clear_notification()` on a read-sequence gap, on a checksum failure,
and in the exception handler, so a partial message cannot prefix stale bytes onto
the next completed report.

If you meant a different line, point me at it and I will explain or restore it.
