# Final Face Pipeline Validation

## Test 14 validated configuration

The supplied demo-video tests validated the following integration settings:

```text
YOLO confidence       = 0.35
SFace threshold       = 0.45
Identity margin       = 0.05
Minimum face          = 16x20
Upscaling             = disabled
Complete track        = yes
Strict votes          = >= 5
Strict vote ratio     = >= 70%
Strict average score  = >= 0.60
Strict average margin = >= 0.05
Evidence dominance    = >= 3x competing evidence
```

## Validation outcome on supplied demo

```text
Track 2  -> AUTH-002
Track 5  -> UNVERIFIED
Track 9  -> AUTH-002
Track 10 -> UNVERIFIED under strict validation
Track 11 -> AUTH-002
```

Small-face support was important: faces below 40x40 generated substantial qualified evidence in the tested demo, while 2x/3x upscaling did not improve the native 1x result.

## Semantics

`AUTHORIZED` means the track satisfied the strict evidence rules for the demo.

`UNVERIFIED` means the available evidence did not meet those rules. It is deliberately not treated as proof that a person is unauthorized.

`AMBIGUOUS` means competing identity evidence remains too close under the configured evidence-dominance check.

`FACE HIDDEN` is a separate explicit demo rule: a person whose face remains unavailable for the configured hidden-face interval is marked for the scenario's hidden-face alert.
