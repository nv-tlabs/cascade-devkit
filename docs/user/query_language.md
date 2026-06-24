# Query Language Specification

The DSL evaluated by `cascade_av.query` against an `AnnotationBundle`
(per-clip) or a `CascadeDataset` (corpus). Examples, grammar, and
semantics.

This document is the **spec**. If a question about syntax or semantics
isn't answered here, it's a defect.

The fastest way to get a feel for the language is to read §1 below.
The formal grammar and semantics start in §2; reach for them when the
examples leave a question open.

---

## 1. Worked examples

### 1.1 Basic patterns

```
# any pedestrian in the clip
agent.type = ped

# pedestrian on a crosswalk while ego decelerates
agent.type = ped and env.type = crosswalk and ego.action = decel

# vehicle in front of ego, signaling left
agent(type = vehicle, pos = front, signaling = signal_left)

# ego decelerates because of a pedestrian
ego.action = decel because_of agent.type = ped

# yellow light precedes ego stopping (within 3s)
light.color = yellow then(3) ego.action = stop

# during a red light, ego never stops
within light.color = red: not ego.action = stop

# multi-lane road, ego changes lane
env(type = road, lanes >= 2) and ego.action in (change_lane_left, change_lane_right)

# action with a flag (suffix encoded in the type string — see §4.10)
agent(type = ped, action.type = "oxd:Walk (jaywalk)")

# light with a flag (the "defining yellow" of the scenarios doc)
light(color = yellow, ego_in_on_yellow = true)
```

### 1.2 Scenarios from `scenarios_and_queries_reference.md`

**Scenario 1** — Ego passes through crosswalk while pedestrian present.
```
agent.type = ped
  and env.type = crosswalk
  and ego.action = drive
```

**Scenario 8** — Pedestrian crosses mid-block (jaywalking).
```
agent(type = ped, action.type in ("oxd:Walk (jaywalk)", "oxd:Walk (jaywalk, erratic)", "oxd:Run (jaywalk)", "oxd:Run (jaywalk, erratic)"))
  and ego.action in (stop, yield, decel)
```

**Scenario 14** — Ego stops at red traffic light.
```
light.color = red
  and ego.action = stop
  and env.type = intersection
```

**Scenario 20** — Ego in intersection when light turns yellow, proceeds.
```
light(color = yellow, ego_in_on_yellow = true)
  and ego.action in (drive, enter, creep)
```

**Scenario 76** — Vehicle stopped in front of ego, ego nudges.
```
agent(type = vehicle, pos = front, action(type in (stop, not_move)))
  and ego.action in (nudge, change_lane_left, change_lane_right)
```

### 1.3 Statistics queries

```
# count clips with any pedestrian
dataset.count("agent.type = ped")

# group clips by ego action
dataset.group_by("ego.action in (stop, yield, decel, drive, turn)", key="ego.action.type")

# histogram of agent types
dataset.histogram("agent.type = vehicle or agent.type = vru", key="agent.type")
```

---

## 2. Lexical structure

```
IDENT      := [a-z_] [a-z0-9_]*           # case-folded to lowercase at parse
QUALIFIED  := IDENT ( ':' IDENT )?         # e.g. oxd:Pedestrian — escape hatch
NUMBER     := DIGIT+ ( '.' DIGIT+ )?
DIGIT      := [0-9]
BOOL       := 'true' | 'false'
COMMENT    := '#' ... end of line         # ignored
WHITESPACE := ' ' | '\t' | '\n' | '\r'    # ignored between tokens
```

**Identifiers are case-insensitive.** `Ped`, `PED`, `ped` all parse the
same.

**Reserved words** (never usable as identifiers):
```
and  or  not  in  while  then  because_of  influenced_by  within  true  false
```

**Comments** start with `#` and run to end of line. Allowed anywhere
whitespace is allowed.

---

## 3. Grammar (EBNF)

```
query           := expression

expression      := within_expr
                 | or_expr

within_expr     := 'within' predicate ':' expression

or_expr         := and_expr ( 'or' and_expr )*

and_expr        := temporal_expr ( 'and' temporal_expr )*

temporal_expr   := unary_expr ( temporal_op unary_expr )*

temporal_op     := 'while'
                 | 'then' ( '(' NUMBER 's'? ')' )?
                 | 'because_of'
                 | 'influenced_by'

unary_expr      := 'not' unary_expr
                 | primary

primary         := entity_clause
                 | entity_ref
                 | attribute_predicate
                 | '(' expression ')'

entity_ref      := ('ego' | 'agent')              # only as influenced_by LHS

entity_clause   := IDENT '(' expression ')'

attribute_predicate
                := attribute_path comparison value

attribute_path  := IDENT ( '.' IDENT )*

comparison      := '=' | '!=' | '>' | '>=' | '<' | '<=' | 'in'

value           := atom | value_set | attr_ref
value_set       := '(' atom ( ',' atom )* ')'
atom            := IDENT | QUALIFIED | NUMBER | BOOL

attr_ref        := attribute_path                # same-entity only; see §4.2
```

`attr_ref` is only valid as the RHS of a non-`in` comparison. It denotes
a reference to another scalar attribute **on the same entity** as the
LHS — see §4.2 for the disambiguation rules (a bare ident on the RHS is
parsed as an `attr_ref` only when it names a known scalar attribute on
the LHS entity; otherwise it falls through to a literal value atom).

### Precedence (lowest to highest)

| Level | Construct |
|---|---|
| 1 | `within W: E` |
| 2 | `or` |
| 3 | `and` |
| 4 | `while`, `then(K)`, `because_of`, `influenced_by` |
| 5 | `not` |
| 6 | entity clause, entity ref, attribute predicate, parentheses |

`while`, `then`, `because_of`, `influenced_by` are **left-associative**
(same level — chainable left-to-right).

---

## 4. Semantics

### 4.1 Match tuples

Every expression evaluates to a `MatchSet`: a set of
`(clip_id, entity_ref, interval)` tuples.

- `clip_id` identifies the matching clip.
- `entity_ref` is the schema entity the predicate matched (an `Agent`, an
  `AgentAction`, a `LightState`, …).
- `interval` is the entity's parseable lifetime. When the entity has no
  meaningful interval (e.g. `clip` itself, or an `Agent` missing
  visibility timestamps), the interval is **the whole clip** — so
  interval algebra (`while`, `then`) stays total.

### 4.2 Attribute predicates

`<entity>.<attr> <cmp> <value>` — match every entity of type `<entity>`
whose `<attr>` satisfies the comparison.

```
agent.type = ped              # every Agent with type ped
env.lanes >= 2                # every Environment with num_lanes >= 2
ego.action in (stop, yield)   # every EgoAction with type stop or yield
light.color = red             # every LightState with color = red
```

- `=` / `!=` use string equality after alias resolution (`ped` →
  `oxd:Pedestrian`).
- `>`, `>=`, `<`, `<=` are numeric.
- `in (a, b, c)` is set membership; the set is OR of the values.
- For **list-valued schema fields** (e.g. `Condition.type: list[str]`),
  `=` matches **existentially** (X is in the list).
- The implicit AND-and-existential-quantifier rule: a clip matches iff
  *at least one* entity satisfies the predicate.

> **Coverage — environments and containment are optional.** Not every
> clip carries environments, conditions, or containment; these layers
> default to empty and their density depends on the annotation scope.
> Ego-centric batches in particular may have few or no environments and
> little or no containment (the ego frequently has none, and agent
> containment may omit `env_id`). Predicates like `env.type = …`,
> `cond.type = …`, `agent.contained_in …` / `ego.contained_in …`, or the
> containment flags below simply won't match on clips where the layer is
> absent — that's expected coverage, not a query error.

> **Containment — `agent.contained_in` / `ego.contained_in`.** The
> containment attribute is also exposed under the name `contained_in`
> in addition to its canonical name `in` (`agent.in` / `ego.in`).
> The DSL parser tokenises bare `in` as the IN set-membership operator
> in every position, which makes the bare-name form unreachable from
> a query string — use `contained_in` instead:
>
> ```
> agent(type = vehicle, contained_in = roundabout)
> ego.contained_in in (road, intersection)
> ```

> **Ego-relative pose — `pos` vs `pos_any`.**
> `agent.pos` and `agent.dir` are sampled at the agent's visibility-window
> **midpoint** (single deterministic value per agent).  For agents whose
> relative pose changes across intervals (e.g. a vehicle that approaches
> in front of ego and ends up perpendicular as it crosses the
> intersection), use the per-interval counterparts `agent.pos_any` and
> `agent.dir_any`.  Both are `kind="list"` attributes that match
> existentially across every `EgoRelativePose` interval the agent
> carries:
>
> ```
> agent.pos_any = front                                  # any pose interval has pos=front
> agent(type = vehicle, dir_any in (perpendicular_lr, perpendicular_rl))
> ```

> **Same-entity attribute-to-attribute comparisons.** The right-hand
> side of a non-`in` comparison may be another scalar attribute **on
> the same entity** as the left-hand side. Use the entity-clause
> shorthand or the qualified `<entity>.<attr>` form — both compile to
> the same AST:
>
> ```
> env(out_lanes > lanes)            # forks: more out-lanes than in-lanes
> env(out_lanes < lanes)            # merges: fewer out-lanes than in-lanes
> env.out_lanes != env.lanes        # qualified form, outside the clause
> ```
>
> Only the six scalar comparators (`=`, `!=`, `>`, `>=`, `<`, `<=`) are
> supported on this RHS shape; `in` continues to take a parenthesized
> value set. Cross-entity references (`env.lanes = agent.amount`),
> sub-entity descent on the RHS (`agent(action.illegal != action.illegal)`),
> arithmetic, and list-valued attributes are rejected at parse time.
>
> Disambiguation: a bare ident on the RHS is treated as an attribute
> reference only when it names a known *scalar* attribute on the LHS
> entity. Otherwise it falls through to a literal value atom — so
> existing queries like `agent.type = ped` still parse as a string
> literal because `ped` is not an attribute.
>
> **None handling.** When either side resolves to `None` for the
> candidate entity, the comparison treats the two sides as **equal**.
> "Equal" satisfies `=`, `>=`, and `<=` but not `!=`, `>`, or `<`.
> Both sides `None` are also treated as equal. The truth table for
> `lanes <op> out_lanes`:
>
> | `lanes` | `out_lanes` | `=` | `!=` | `>` | `>=` | `<` | `<=` |
> |---|---|---|---|---|---|---|---|
> | 3 | None | True | False | False | True | False | True |
> | None | 2 | True | False | False | True | False | True |
> | None | None | True | False | False | True | False | True |
> | 3 | 4 | False | True | False | False | True | True |
> | 3 | 2 | False | True | True | True | False | False |
>
> This rule applies only to attr-to-attr comparisons; literal RHS
> predicates keep the existing behaviour (any `None` on the LHS → no
> match).

> **Agent group size — `agent.amount`.** Matches the agent-count
> annotation: lone agent vs row/group vs density tiers. The aliases
> are `single`, `row` (or `group`), `light_traffic`, `medium_traffic`,
> `heavy_traffic`, plus two parents: `multiple` (everything but
> `single`) and `traffic` (the three density tiers):
>
> ```
> agent.amount = single
> agent(type = ped, amount = group)
> agent.amount in (light_traffic, medium_traffic)
> ```
>
> The three density tiers `light_traffic` / `medium_traffic` /
> `heavy_traffic` (and the `traffic` parent over them) are
> **deprecated upstream** — the annotator no longer emits them — but
> stay resolvable for older bundles.

> **Signaling details on `agent.prop` / `ego.prop`.** Properties of
> type `signal` carry a `signaling_details` sub-object that names
> the source modality (flashing light vs hand gesture vs sign) and
> a `not_facing_ego` boolean. Both surface as predicates on the
> property sub-entity:
>
> ```
> agent.prop.source = flashing_light
> agent.prop.source in (flashing_light, holding_sign)
> agent.prop.not_facing_ego = false
> agent(prop.source = flashing_light)
> ```
>
> For non-signal properties both attributes resolve to `None` and
> the predicate falls through, so they compose with the broader
> property filter without an explicit type guard.

> **Signaling intent — `agent.signaling`.** The list-valued
> `agent.signaling` attribute collects the *intent* of every Signal
> property the agent carries (the modelled meaning of the gesture /
> indicator), matched existentially. Intent aliases live in the
> `signaling_intent` family:
>
> | Alias | Resolves to |
> |---|---|
> | `proceed` | `Proceed` |
> | `stop` | `Stop` |
> | `caution` | `Caution` |
> | `slow` | `Slow down` |
> | `follow` | `Follow` |
> | `danger` | `Danger` |
> | `unclear` | `Unclear` |
> | `signal_left` | `Left Indicator` |
> | `signal_right` | `Right Indicator` |
> | `turn` | `Turn` — **deprecated upstream**, still resolvable |
> | `other` | `Other` |
>
> ```
> agent(type = vehicle, pos = front, signaling = signal_left)
> agent.signaling in (signal_left, signal_right)
> ```
>
> `signal_left` / `signal_right` were added when the annotator split
> the old combined `Turn` intent into explicit Left/Right indicators;
> `turn` (→ `"Turn"`) is kept resolvable for older bundles but is
> deprecated upstream — prefer the directional aliases.

> **Containment flags — `lane_edge`, `near_lane`, `illegal_lane`.**
> Boolean predicates over the entity's containment records. Match
> existentially: true iff *any* containment carries the flag. Exposed
> on both `agent` and `obj`:
>
> ```
> agent.illegal_lane = true                       # illegally parked / stopped
> obj.lane_edge = true                            # cones / barriers on a lane boundary
> agent(type = car, illegal_lane = true)
> ```
>
> `lane_edge` is true when `Containment.edge` ∈ `{"left", "right"}`;
> `near_lane` and `illegal_lane` mirror `Containment.near_flag` and
> `Containment.illegal_flag` directly. Per §4.11, all three must
> compare against an explicit boolean.

> **Action ID-link lists — `action.link_to`, `action.action_target`.**
> Two list-valued, ID-keyed attributes that complement the existing
> `because_of` *operator*. `because_of` still walks
> `Action.because_of`; these expose the sibling ID lists for
> membership tests:
>
> ```
> agent.action.link_to = signal-id            # Signal action targeting a specific signal-head
> ego.action.action_target in (a, b)
> ```
>
> Hyphenated UUIDs in the corpus must be quoted (`"abc-def"`); the
> lexer treats bare `-` as an error outside string literals.
> (The `EgoVehicle.influenced_by` field is exposed by the
> `influenced_by` *operator* rather than as a membership-keyed
> attribute — see the relational operators section.)

> **Clip description — `clip.brief_description`.** The annotator's
> one-line summary of the clip; 100 % populated in current corpora.
> Equality and IN-set only (no regex):
>
> ```
> clip.brief_description = "ego stops at a red light"
> clip.brief_description in ("…", "…")
> ```

> **`"Other"` literals are aliased.** The four families
> `agent_type`, `action_type`, `env_type`, and `signaling_source`
> now resolve `other → "Other"` directly, so users no longer have
> to type the schema literal in quotes. The generic-vehicle literal
> `"Vehicle"` (annotator's fallback when an agent is a vehicle that
> doesn't match any specific subtype) is folded into the `vehicle`
> parent alias and also accessible as the leaf `generic_vehicle`.

> **Newer environment and object aliases.** The annotator (internal
> 0.5/0.6.x) added a few vocab values the DevKit's 2.0.0 spec now
> covers in place. New `env_type` aliases:
>
> | Alias | Resolves to |
> |---|---|
> | `speed_bump` | `fst:SpeedBump` |
> | `light_rail_lane` | `fst:LightRailLane` |
>
> New `obj` (traffic-object) type alias:
>
> | Alias | Resolves to |
> |---|---|
> | `boom_gate` | `Boom gate` |
>
> ```
> env.type = speed_bump
> obj.type = boom_gate
> ```

> **Condition → environment lookup — `cond.env_type` /
> `cond.env_lanes` / `cond.env_one_way` / `cond.env_id`.** Each
> `Condition` carries an `env_id` pointing at one of the bundle's
> `Environment` records. The DSL has no cross-entity join operator
> today, but the four most-useful environment fields are reachable
> as *denormalized* attributes on `cond` — the common "is this
> condition on a road / highway / roundabout?" question works
> without dropping into Python:
>
> ```
> cond.type = construction and cond.env_type = road      # construction on the road
> cond.env_one_way = true                                # any cond on a one-way env
> cond.env_lanes > 2                                     # any cond on a 3+ lane env
> ```
>
> `cond.env_type` reuses the `env_type` alias family (`road`,
> `crossroad`, `intersection`, …). `env_lanes` is `int | None`,
> `env_one_way` is `bool | None`, `env_id` is the raw string
> passthrough for callers that want to compose with other ID-keyed
> attributes via string equality. A condition whose `env_id` is
> empty or doesn't resolve to any Environment falls through to
> `None` for the three derived fields, so predicates compose
> without an explicit guard.

### 4.3 Entity clauses (same-entity grouping)

`<entity>(<expr>)` — match every entity of type `<entity>` such that
`<expr>` holds **on that same entity**. The inner expression sees
attribute paths rooted at the entity (no `<entity>.` prefix needed —
though it's accepted as a no-op).

```
agent(type = vehicle and pos = front)
# every Agent with both type=vehicle and pos=front

agent(type = ped, action.type = "oxd:Walk (jaywalk)")
# every Agent of type ped that has at least one action of that exact type
# (see §4.10 for the parenthesized-suffix convention)

light(color = yellow, ego_in_on_yellow = true)
# every LightState with color=yellow AND ego_in_on_yellow=true

env(type = lane_fork, out_lanes > lanes)
# every Environment that is a lane fork and adds at least one lane
# downstream — uses the same-entity attr-to-attr RHS from §4.2
```

Inside an entity clause, `and`/`or`/`not` compose constraints on the
*same* entity. Outside, they compose constraints across *the clip*.

Inside an entity clause the comma `,` is sugar for `and` (one
constraint per clause, joined left-associatively). The two forms
`agent(type=vehicle and pos=front)` and `agent(type=vehicle, pos=front)`
are equivalent; the comma form matches the worked examples in §1.

### 4.4 Boolean operators

- `not P` — within the current temporal window (default = whole clip),
  return the clips/intervals where P has *no* matches. Result is a
  whole-clip match tuple for each clip where P is empty.
- `A and B` — both A and B return non-empty for the clip; match tuples
  are the union of A's and B's tuples for that clip.
- `A or B` — clips where either A or B returns non-empty; match tuples
  are the union.

### 4.5 Temporal operators

- `A while B` — pairs `(mA, mB)` with `mA.interval ∩ mB.interval ≠ ∅`.
  Each yields a match tuple `(clip_id, [mA.entity, mB.entity], iA ∩ iB)`.
- `A then(K) B` — pairs where `mB.interval.start ≥ mA.interval.start`
  AND
  (`mA.interval.overlaps(mB.interval)` OR
   `0 ≤ mB.interval.start - mA.interval.end ≤ K`).
  Default `K = 0` (must touch or overlap).
- **Chainable**: `A then(2) B then(3) C` parses left-to-right as
  `(A then(2) B) then(3) C`.

### 4.6 Relational operator — `because_of`

- `A because_of B` — pairs where A's matched entity is an action and B's
  matched entity is referenced in `action.because_of`. Both halves are
  emitted as match tuples for the clip:
    `(clip, mA.entity, mA.interval)` and
    `(clip, mB.entity, mB.interval)`.
  Uses only the `because_of` edge.

### 4.7 Relational operator — `influenced_by`

- `S influenced_by P` — match every clip where the LHS subject `S` (the
  ego vehicle or an agent) has at least one `Influence` window
  populated with an `influencers` list that contains an entity
  satisfying the RHS predicate `P`.

The LHS is intentionally narrow: it must be one of

- the bare entity reference `ego` or `agent`, **or**
- an entity clause rooted at `ego` or `agent`
  (e.g. `agent(type = vehicle)`).

Any other LHS shape — clip-level predicates, attribute predicates,
sub-entity descent, light / object predicates — raises
`QueryParseError` with a message starting `influenced_by LHS must be …`.
Walks the schema's `EgoVehicle.influenced_by` / `Agent.influenced_by`
edge, which carries a per-window list of influencer IDs that the
annotator believed modulated the subject's behaviour.

```
ego influenced_by obj.type = stop_sign            # ego yielded at a stop sign
ego influenced_by light.color = red               # ego acted on a red light
ego influenced_by obj.type in (yield_sign, stop_sign)
ego influenced_by (obj.type = stop_sign or light.color = red)
agent(type = vehicle) influenced_by light.color = red
ego influenced_by obj.type = yield_sign while agent.type = ped
```

Resolution rules:

- The RHS is evaluated against each `influencer_id` in the window via
  the bundle's `IdIndex`. Dangling IDs are silently skipped — the
  corpus carries some.
- `signal_head` and `traffic_light` influencer IDs expand to their
  `LightStates`, but only states whose own interval **overlaps the
  Influence window** are considered — so a head cycling
  Green → Yellow → Red across the clip matches `light.color = red`
  for an Influence window over the red phase and `light.color = green`
  for one over the green phase, never both for the same window. The
  expanded states are stamped with the same `_owner_signal_head` /
  `_owner_traffic_light` back-refs used by `light.affects_ego`.
- The **match interval is the `Influence` window**, *not* the
  influencer's own lifetime. This is what makes `within W: S
  influenced_by P` and `S influenced_by P while Q` compose
  correctly — the influencer is usually a static sign or fixed light
  whose own interval spans the whole clip; the annotator's `Influence`
  window is the period that actually mattered.
- One match per `Influence` window per subject — the first satisfying
  influencer wins. Multi-influencer windows do not multiply matches.

Owner of each match is the subject (`EgoVehicle` for `ego`, the
matched `Agent` for `agent`).

### 4.8 Window scoping

- `within W: E` — restricts the temporal window of E to the union of
  intervals from W's match set. All interval-aware operators inside E
  see this window. In particular, `not P` inside `within W:` means "P
  has no matches within W."

```
within light.color = red: not ego.action = stop
# clips where, during any red-light interval, ego never stops
```

### 4.9 Value aliasing

The DSL accepts short aliases (`ped`) and full literals (`oxd:Pedestrian`)
interchangeably. Aliases live in `cascade_av.query.constants`, with
hierarchical parents — e.g. `vehicle` is the parent set of
`{car, truck, bus, motorcycle, …}`; `vru` is the parent set of
`{ped, cyclist}`. Using a parent expands to the union.

```
agent.type = vehicle
# expands to agent.type in (car, truck, bus, motorcycle, …)
```

### 4.10 Action type suffixes (corpus convention)

Action types in schema `2.0.0` encode optional flags as **parenthesized
suffixes** baked into the `action_type` string — the suffix is the
single source of truth. The pre-2.0.0 side-channel flag fields
(`jaywalk_flag`, `erratic_flag`, `ego_lane_flag`, `turn_protected`,
`change_where`, `nudge_magnitude`, `is_aggressive_or_cut_in`,
`maneuver_aborted_flag`) **are gone** — `AgentAction` keeps only
`illegal_flag` and `signaling_details`; `EgoAction` keeps only
`illegal_flag`. Querying any of the dropped attributes raises
`NameError` ("unknown attribute on agent.action / ego.action").

The full set of suffixed values the corpus emits:

| Base verb | Suffixed forms |
|---|---|
| `oxd:Walk` | `(jaywalk)`, `(erratic)`, `(jaywalk, erratic)` |
| `oxd:Run` | `(jaywalk)`, `(erratic)`, `(jaywalk, erratic)` |
| `fst:Nudge` | `(in lane)`, `(out of lane: not into ego lane)`, `(out of lane: into ego lane)` |
| `oxd:ChangeLane` | `(left)`, `(right)` |
| `oxd:Overtake` | `(using ego lane)`, `(not using ego lane)` |
| `oxd:MakeALeftTurn` | `(protected)`, `(unprotected)` |
| `oxd:MakeARightTurn` | `(protected)`, `(unprotected)` |
| `fst:MakeAUTurn` | `(protected)`, `(unprotected)` |

DSL rules — suffix matching is the **only** path (no flag attribute is
checked on either side):

- **Base-verb aliases are parents.** `agent.action.type = walk` matches
  every action whose type **starts with** `oxd:Walk` — bare base form
  *and* every `(…)` variant. Same for `run`, `nudge`, `change_lane`,
  `turn`, `overtake`, etc. Use this when you want "any walk, however
  it's flagged."
- **Direction sub-actions resolve to the specific suffixed form.**
  `change_lane_left` → `oxd:ChangeLane (left)`,
  `change_lane_right` → `oxd:ChangeLane (right)`,
  `turn_left` → `oxd:MakeALeftTurn`,
  `turn_right` → `oxd:MakeARightTurn`, etc. `change_lane` and `turn`
  remain base-verb parents that include every suffix variant.
- **For an exact suffixed form, write the full literal in quotes.**
  String literals (double-quoted) reach the engine as the matched
  `action_type` string verbatim:

  ```
  # exactly oxd:Walk (jaywalk) — no plain "oxd:Walk", no other suffixes
  agent.action.type = "oxd:Walk (jaywalk)"

  # any walk that carries the jaywalk flag (with or without erratic)
  agent.action.type in ("oxd:Walk (jaywalk)", "oxd:Walk (jaywalk, erratic)")

  # protected vs. unprotected left turn
  ego.action.type = "oxd:MakeALeftTurn (unprotected)"
  ```

- **Pre-2.0.0 syntax (`action(jaywalk = true)`) no longer parses** —
  the engine reports `unknown attribute jaywalk on agent.action`.
  Rewrite as a value-set predicate over `action.type`, or use a
  base-verb alias if you don't care which flag is present:

  ```
  # before (schema 1.x):  agent.action(jaywalk = true)
  # after (schema 2.0.0):
  agent.action.type in ("oxd:Walk (jaywalk)", "oxd:Walk (jaywalk, erratic)", "oxd:Run (jaywalk)", "oxd:Run (jaywalk, erratic)")
  ```

> **Standalone `Jaywalk` verb vs. the `(jaywalk)` suffix.** Upstream
> (annotator internal 0.5/0.6.x) split jaywalking out into a **standalone
> base verb** `"Jaywalk"` — a distinct `action_type`, not a suffix on
> `oxd:Walk` / `oxd:Run`. It has its own alias `jaywalk_action`:
>
> ```
> agent(type = ped, action.type = jaywalk_action)   # standalone "Jaywalk"
> ```
>
> This is keyed `jaywalk_action` (not `jaywalk`) so it doesn't collide
> with the suffix-flag token `jaywalk` used by the base-verb prefix
> match. The older combined forms — `"oxd:Walk (jaywalk)"`,
> `"oxd:Walk (jaywalk, erratic)"`, `"oxd:Run (jaywalk)"`,
> `"oxd:Run (jaywalk, erratic)"` (and their `(erratic)`-only siblings)
> — **remain matchable** via the suffix path above, but are
> **deprecated upstream**: producers now emit the standalone `Jaywalk`
> verb plus an `Erratic` property (§4.10) instead. The Scenario 8 query
> in §1.2 still resolves, but new queries should prefer
> `action.type = jaywalk_action`.

> **Deprecated `not_move`.** The ego-action alias `not_move`
> (→ `oxd:NotMove`, used in Scenario 76 of §1.2) is **deprecated
> upstream** — the annotator now emits `oxd:Stop` instead. It stays
> resolvable for older bundles; prefer `stop` for new queries.

> **Aggression and cut-in are not actions.** The dropped
> `is_aggressive_or_cut_in` flag is replaced by an `AgentProperty` whose
> `property_type = "Aggressive"`. A "cut-in" is decomposed as a
> `oxd:ChangeLane (left)` or `oxd:ChangeLane (right)` action **with a
> simultaneous `Aggressive` property on the same agent.** Query the
> property directly, not the action:
>
> ```
> # aggressive cut-in to the left
> agent(prop.type = aggressive) while agent.action.type = "oxd:ChangeLane (left)"
> ```
>
> See §4.11 for the property sub-entity.

### 4.11 Property sub-entity (`agent.prop`, `ego.prop`)

Agents and the ego carry a `properties` list of `AgentProperty` entries
— each is a typed flag spanning a time window. The DSL surfaces them
as a sub-entity:

```
agent.prop.type = aggressive          # any agent with an Aggressive property
ego.prop.type = signal                # ego had a Signal property
agent(type = ped, prop.type = erratic) # a pedestrian agent flagged Erratic
```

Property-type aliases live in the `agent_property_type` /
`ego_property_type` families: `signal`, `slow`, `fast`, `aggressive`,
`erratic`, `emergency`, `on_duty`, `double_parked`, `outside_camera`,
`stopped` (→ `"Stopped"`, on both `agent` and `ego`), `other`.
Properties carry their own start/end timestamps, so
`while` / `then` / `because_of` over `agent.prop` works the same as
over actions.

### 4.12 Booleans must be explicit

Boolean attributes always carry an explicit `=true` / `=false`:

```
light(color = yellow, ego_in_on_yellow = true)    # correct
light(color = yellow, ego_in_on_yellow)           # SYNTAX ERROR
```

### 4.13 Clip-level attributes

`clip` is a virtual entity for clip-level metadata:

```
clip.eventful = true
clip.eventful_reason = ego_adapts
clip.duration >= 10
clip.id = "00f2c7d3..."
```

A `clip` predicate matches the bundle itself; the entity_ref is the
bundle, the interval is the whole clip.

> **`clip.eventful_reason`.** A categorical rationale for why a clip
> was flagged `eventful` (added upstream in annotator internal 0.6.x).
> Aliases live in the `eventful_reason` family, with four values:
>
> | Alias | Meaning |
> |---|---|
> | `ego_adapts` | ego adapts its behaviour to the scene |
> | `special_env` | a special/unusual environment drove the event |
> | `agent_adapts` | another agent adapts, prompting the event |
> | `other` | none of the above |
>
> ```
> clip(eventful = true, eventful_reason = special_env)
> clip.eventful_reason in (ego_adapts, agent_adapts)
> ```
>
> A sibling free-text field `eventful_reason_other` carries the
> annotator's note when `eventful_reason = other`; it is not exposed
> as a queryable attribute (no equality/regex predicate), only
> read off the parsed bundle.

---

## 5. Result types

The query API surfaces four small, related types. You will hit all
four within the first half-hour of using the DevKit.

| Type | Where it lives | What it is |
|---|---|---|
| `Match` | `cascade_av.query` | One hit: `(clip_id, entity, interval)`. `interval` is the entity's lifetime clipped to the operator's window. |
| `MatchSet` | `cascade_av.query` | The full result of `ds.find(...)`. A tuple of `Match`es plus a weakref back to the producing dataset. |
| `Sequence` | `cascade_av.dataset` | The per-clip handle returned by `ds.get_sequence(clip_id)`: parsed annotation + camera / sensor accessors + `.visualize()`. |
| `ContextWindow` | `cascade_av.query.context` | A snapshot of every entity in a clip whose annotated time range overlaps a given interval — agents, ego actions, environments, conditions, light states, traffic objects. Built by `ds.context_for(match)`. |

### 5.1 What a `MatchSet` carries

```python
matches = ds.find("ego.action = decel because_of agent.type = ped")

len(matches)              # number of Match tuples
list(matches.clips())     # unique clip_ids (preserves first-seen order)
matches.matches           # the underlying tuple[Match, ...]
matches.matches[0].entity # the schema object that matched (Agent, EgoAction, …)
matches.matches[0].interval  # cascade_av.query.time.Interval (start, end in seconds)
matches.intervals()       # list[Interval] for every Match with an interval
matches.entities()        # list of every entity object
matches.sequences()       # iterator of (Match, Sequence) pairs — resolves clips
                          #   on the producing dataset (lazy, dedup-by-clip-id)
matches.visualize()       # carousel of mini-players (one per match)
bool(matches)             # True iff there is at least one match
```

`matches.sequences()` and `matches.visualize()` require the
`MatchSet` to have been produced by a dataset-aware entry point
(`ds.find(...)`, `find_on_dataset`, `find_on_bundle(..., dataset=ds)`).
Match sets built from a bundle without a dataset reference raise
`RuntimeError` on these methods.

### 5.2 Aggregation helpers on `CascadeDataset`

`find()` returns every match. When you only need counts, three
aggregations are wired on the dataset:

| Call | Returns | Notes |
|---|---|---|
| `ds.count(query)` | `int` | Number of clips with at least one match. |
| `ds.group_by(query, key=...)` | `dict[str, int]` | Clip count grouped by `key` (an attribute path the matched entities expose). |
| `ds.histogram(query, key=...)` | `dict[str, int]` | Match-count distribution over `key` values (counts every match, not just clips). |

`key` is the same dotted path you'd use on the right-hand side of an
attribute predicate (e.g. `"agent.type"`, `"ego.action.type"`).

---

## 6. Storage in markdown

Queries live in **fenced code blocks**, one query per block, language
tag `query` for syntax highlighting:

```query
agent(type = ped, action.type in ("oxd:Walk (jaywalk)", "oxd:Walk (jaywalk, erratic)", "oxd:Run (jaywalk)", "oxd:Run (jaywalk, erratic)"))
  and ego.action in (stop, yield, decel)
```

Whitespace (including line breaks) is insignificant. Comments with `#`
are allowed inside the block.

For documentation that lists many scenarios (like
`scenarios_and_queries_reference.md`), each scenario gets one code block.

---

## 7. Error model

The parser must produce **structured errors**:

- `LexError(pos, char)` — unknown character.
- `ParseError(pos, expected, found)` — grammar violation.
- `NameError(pos, name, kind)` — unknown entity / attribute / value
  alias. The error includes "did you mean X?" hints from a fuzzy match
  over the registry.
- `TypeError(pos, attr, expected_type, got_value)` — wrong value type
  for a typed attribute (e.g. `env.lanes = "red"`).

All errors carry `(line, column)` and a snippet of the offending text.
