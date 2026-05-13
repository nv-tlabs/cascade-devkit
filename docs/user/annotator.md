# Annotator user guide

The annotator is a local web app for editing CASCADE dataset annotation
JSON bundles in a browser. You launch it from the terminal, it opens
`http://127.0.0.1:8765/` in your browser, and from there you click,
drag, and key-stroke your way through annotating clips. No login, no
database — your clips and JSONs live on local disk and the tool reads
and writes them in place.

This document walks through the UI end-to-end. For installation and
CLI flags, see [`tools/annotator/README.md`](../../tools/annotator/README.md).

> Conventions: `Cmd/Ctrl+S` means `Cmd+S` on macOS, `Ctrl+S` elsewhere.
> "Click" is a left-click; "right-click" creates new things; "Shift+click"
> and "Ctrl+click" draw links between things.

---

## 1. Layout

The window is a three-column layout with a top bar across the centre:

```
+--------------+-----------------------------------+-------------+
| Sidebar      | Lock bar                          | Right panel |
| (clip list)  |-----------------------------------|             |
|              | Video player                      |   Save      |
|              |-----------------------------------|             |
|              | (drag the splitter)               |   Status    |
|              |-----------------------------------|             |
|              | Timeline toolbar                  |   Selected  |
|              |-----------------------------------|   segment   |
|              | Track labels |    Timeline canvas |   editor    |
|              |              |    (segments,      |             |
|              |              |     arrows,        |   Coverage  |
|              |              |     playhead)      |             |
|              |              |                    |   Export /  |
|              |              |                    |   Load /    |
|              |              |                    |   Clear     |
+--------------+--------------+--------------------+-------------+
```

- **Sidebar** (left): the list of clips, with a search box and
  "annotated" / "unlabelled" badges.
- **Lock bar** (top centre): shows whether editing is locked, the file
  path you're editing, an "Unsaved changes" pill when dirty, and a
  theme toggle.
- **Video player** (centre upper): the clip with playback controls and
  per-keypoint overlay; also hosts the visualization toggles for
  keypoints and the five arrow types.
- **Timeline** (centre lower): every annotation as a horizontal bar on
  a group of tracks (Environments, TrafficLights, Objects, Agents, Ego).
  Drag the splitter between the video and the timeline to resize.
- **Right panel**: the **Save** button, status pill, the editor for the
  currently selected segment, coverage bars, the causality summary, and
  the Export / Load / Clear JSON actions.

---

## 2. Getting started — annotating your first clip

Assumes you've followed the install in `tools/annotator/README.md` and
the server is running.

### 2.1 Pick a clip

In the left sidebar, click a clip. Two kinds of clips show up:

- **annotated** (green badge): a JSON already exists on disk.
- **unlabelled** (amber badge): only a video file. Selecting it loads a
  minimum-valid empty bundle; your first save creates the JSON.

The search box at the top filters by clip ID.

### 2.2 Unlock to edit

Every clip starts **locked** — to prevent accidental edits the first
thing you'll do for a new clip is click **Unlock to edit** in the top
lock bar. A small confirmation row appears across the lock bar:

> *Unlocking allows you to modify or delete annotation entries.
> Continue?* &nbsp;`Cancel` &nbsp;`Unlock anyway`

Click **Unlock anyway**. The lock-bar label flips to "Editing enabled"
in green, and edits will now take effect. If the server was launched
with `--read-only`, the unlock button is hidden — you can browse but
not edit.

### 2.3 Mark the clip as eventful or nominal

In the right panel, **Relevance** is the first section after the Save
button. Two buttons: **Yes** / **No**.

- **Yes** — the clip shows something worth annotating. The timeline
  becomes editable.
- **No** — the clip is nominal driving with nothing to annotate. The
  timeline collapses to a "Nominal driving — no annotation needed"
  message. You can flip this back at any time.

### 2.4 Write a brief description

Click the placeholder under **Description** ("Click to add…") in the
right panel, type a short summary of what's happening in the clip,
and click anywhere else to commit it.

### 2.5 Create entities on the timeline

The timeline is grouped into rows by entity type. To create something,
**right-click an empty stretch** of the appropriate track. Examples:

- Right-click on the **Env Track 1** row, in empty space → a new
  environment segment appears, typed `oxd:Road` by default.
- Right-click on an empty **Light Track** row → a new traffic light.
- Right-click on an empty **Object Track** row → a new traffic object.
- Right-click on an empty **Agent Track** row → a new agent.
- Right-click on an empty **Ego Actions** row → a new ego action.

The new segment is selected automatically and its editor opens in the
right panel.

### 2.6 Add subtrack rows inside an entity

Each entity has subtrack rows underneath its main bar — for example,
an environment has a `conditions` row; an agent has `containment`,
`pose`, `influences`, `actions`, `properties` rows. To create a child
on a subtrack, **right-click the subtrack row inside the parent's time
range**. The new child gets clamped to the parent's interval.

See [§4](#4-tracks-reference) for the full track / subtrack reference.

### 2.7 Fill in the editor

Whatever you select on the timeline opens its editor in the right
panel. Pick the type from the dropdown, set the time range with the
**Start (s)** / **End (s)** inputs (or drag the segment edges in the
timeline), and toggle flags like *Illegal*, *One way*, *Jaywalk*, etc.

A **Missing fields** warning chip appears at the top of the editor
when required fields are empty.

### 2.8 Save

Click the big **Save** button at the top of the right panel, or press
**`Cmd/Ctrl+S`**. The lock bar's amber "Unsaved changes" pill
disappears, and on the **first** save for this clip in this server
session the original JSON is copied to `<file>.json.bak`.

### 2.9 Switch clips

Click another clip in the sidebar. If you have unsaved changes a
dialog asks **Save / Discard / Cancel**. Save is disabled if the
bundle is still locked or the server is read-only — unlock first.

---

## 3. Mouse, keyboard, and arrow basics

### 3.1 Mouse — timeline

| Action | Effect |
|---|---|
| **Left-click** a segment | Select it; its editor opens in the right panel. |
| **Left-drag** the middle of a segment | Move it in time (or to a sibling lane, where supported). |
| **Left-drag** a segment edge | Resize that end. The cursor turns into a horizontal-resize indicator. |
| **Right-click** empty space on a track or subtrack row | Create a new segment of that track/subtrack type at the click. |
| **Right-click** a keypoint diamond | Delete that keypoint. |
| **Shift+click** a target segment | Draw a **link** from the selected segment to the target (see [§5](#5-creating-links-and-arrows)). |
| **Ctrl+click** a target segment | Add the target to the selected action's **because_of** list. |
| **Left-click** on the time ruler | Scrub the playhead. |
| **Wheel** | Horizontal scroll along time. |
| **Ctrl/Cmd + wheel** | Zoom the timeline horizontally. |

### 3.2 Mouse — video

| Action | Effect |
|---|---|
| **Left-click** a keypoint on the frame | Select the owning entity (and scrub to the keypoint's time). |
| **Left-click** empty video (entity selected, no keypoints yet) | Place the first keypoint for the selected entity. |
| **Left-click** + drag a keypoint | Move it. |
| **Right-click** a keypoint | Delete it. |
| **Wheel** | Zoom into the video frame. |
| **Drag** when zoomed | Pan. |

### 3.3 Keyboard shortcuts

| Key(s) | Where | Effect |
|---|---|---|
| `Space` | anywhere (not in a text field) | Play / pause video |
| `←` / `→` | anywhere (not in a text field) | Step one frame backward / forward |
| `Cmd/Ctrl + S` | anywhere | Save the current bundle |
| `Cmd/Ctrl + Z` | timeline focused | Undo the last edit (up to 10 steps) |
| `Delete` / `Backspace` | with a timeline segment selected | Delete the selected segment |
| `B` | anywhere (not in a text field) | Toggle keypoint overlays on / off |

The lock bar's theme toggle and the timeline's zoom buttons have no
keyboard shortcut — click them. The lock toggle is also click-only and
guarded by a confirm dialog.

### 3.4 The undo model

`Cmd/Ctrl+Z` walks back through up to ten prior bundle snapshots. Each
edit takes a snapshot before applying. If undo lands you back exactly
on the last-saved state, the "Unsaved changes" pill clears
automatically.

Undo never crosses a clip switch (the stack resets on selection).

---

## 4. Tracks reference

The timeline has five **groups**, each a coloured strip down the left
edge. Click a group header to collapse it. Plus / minus icons next to a
group header add or remove tracks within that group.

| Group | Default colour | What it holds |
|---|---|---|
| Environments | green | Road segments, intersections, crosswalks, etc. |
| TrafficLights | red | Traffic light heads + their state sequences |
| Objects | amber | Stop signs, cones, debris, barriers, etc. |
| Agents | purple | Vehicles, pedestrians, cyclists, animals, officers |
| Ego | blue | The recording vehicle's actions, containment, influences, properties |

Each group has multiple **tracks** (rows). Tracks are numbered:
`Env Track 1`, `Env Track 2`, … You add more with the `+` button next
to the group header, and remove with the `-` button next to a specific
track. Adding tracks is a UI concept only — entities are assigned to a
track automatically based on overlap; you only need extra tracks if
your entities would visually collide on a single row.

### 4.1 Per-group subtracks

Each entity is drawn as a tall stack of rows: a **main bar** on top,
plus per-entity **subtracks** underneath. The subtrack labels appear
on the right side of the track-label panel.

**Environments**

```
[ Environment main bar ]
[ conditions          ]    one or more lanes — weather, lighting, ...
```

**Traffic Lights** — the most complex group:

```
[ Traffic-light main bar          ]
[ TL control (physical containment) ]    where the light is mounted
[ signal head 1                   ]    one head per direction / aspect
[   containment                   ]    which environment(s) it governs
[   states (lanes 1..N)           ]    red / yellow / green over time
[ signal head 2 ...               ]
```

A traffic light can have multiple **signal heads** (each representing
one face of the device); each head has its own state sequence and
environment-controls.

**Objects**

```
[ Object main bar          ]
[ containment (lanes 1..N) ]    which environment the object sits in
[ states                   ]    e.g. open/closed, static/moving
```

**Agents**

```
[ Agent main bar            ]
[ containment (lanes 1..N)  ]
[ pose                      ]    ego-relative position over time
[ influences (lanes 1..N)   ]    what is causing the agent's behaviour
[ actions                   ]    drive / stop / walk / jaywalk / ...
[ properties (lanes 1..N)   ]    e.g. signalling lights, sirens
```

**Ego**

```
[ Ego main bar — not interactive ]
[ containment (lanes 1..N)       ]
[ influences (lanes 1..N)        ]
[ actions                        ]
[ properties (lanes 1..N)        ]
```

The Ego main bar is read-only. Add ego-level annotations by
right-clicking the subtrack rows, or click "+ Add Containment / Add
Influence / Add Property" on the Ego entry in the right panel when the
ego bar is selected.

### 4.2 Creating subtrack segments

Right-click an empty stretch *in the subtrack row* (inside the
parent's time range). The new child is placed at the click and clamped
to the parent's interval. If the row already has segments at that
time, the tool searches for the nearest gap.

### 4.3 Adding / removing tracks and rows

- **`+`** next to a group header → add a new track in that group.
- **`-`** next to a specific track → remove that track (only if empty
  or after reassigning entities).
- Some subtracks gain extra **lanes** automatically when you create
  segments that overlap in time inside the same row — the row grows
  to accommodate them.

### 4.4 Collapse a group

Click the chevron next to a group label (or anywhere on the group
header) to collapse all tracks in that group to a single thin row.
Click again to expand.

---

## 5. Creating links and arrows

The annotator surfaces five kinds of inter-entity relationships as
arrows overlaid on the timeline. You both **create** and **show / hide**
these relationships from the UI.

### 5.1 The five arrow types

Toggle them on or off with the `B L C I T` buttons in the bottom-right
of the video player's controls. Each is a single coloured letter; the
button is dimmed when the arrow type is hidden.

The "family key" column matches the kwarg keys the DevKit's `arrows={}`
filter accepts — so the same five names are used in the UI, in the
visualization API, and on disk.

| Button | Family key | Meaning |
|---|---|---|
| **B** | `because_of` | Action A happens because of entity B (causality). |
| **L** | `link_to` | Action A is directed at agent / ego B. |
| **C** | `containment` | Containment row points at the environment that contains it. |
| **I** | `influence` | Ego or agent influence row points at its influencers. |
| **T** | `action_target` | Action whose target is a specific entity (e.g. "follow this agent"). |

Hiding a type only hides the arrows; the underlying data is unchanged.

### 5.2 Creating a *because of* link (Ctrl+click)

> *Action A happens **because of** entity B.*

1. Select the **action** segment (an ego action or an agent action).
2. **Ctrl+click** on the target — an agent action, an ego action, a
   traffic-light state, a traffic light (parent bar), a traffic object
   (parent bar), or an environment / property.
3. The target's ID appears as a red chip under **Because of (causality)**
   in the action's editor.

Remove a link by clicking the `×` on the chip. You can also add or
remove targets by hand from the **Because of** dropdown in the right
panel — Ctrl+click is just the timeline shortcut.

### 5.3 Creating a *link to* arrow (Shift+click)

> *Action / signalling property A is directed at agent / ego B.*

1. Select an **action** or a **property** segment (ego or agent).
2. **Shift+click** on the target's agent main bar (or the ego row).
3. The target's ID is added to the segment's `link_to` (or, for
   signalling properties, `signaling_details.link_to`).

This is also how you say "the ego is following *this* car" or "this
pedestrian is signalling *that* driver".

### 5.4 Creating a *contained in* relationship (Shift+click on a containment row)

> *Containment row C lives inside environment E.*

1. Select a **containment** segment on any entity (Ego / Agent / Object
   / Traffic Light).
2. **Shift+click** on the target **environment**'s main bar.
3. The containment row's `env_id` is set to the environment's ID and
   the arrow draws.

You can also pick the environment from the **Contained in Environment**
dropdown in the right panel — Shift+click is the timeline shortcut.

### 5.5 Creating an *influenced by* relationship (Shift+click on an influence row)

> *Ego / agent action is influenced by entity I.*

1. Select an **influence** subtrack segment on the ego or an agent.
2. **Shift+click** on the influencing entity's main bar (an agent, a
   traffic object, or a traffic light).
3. The entity ID is appended to the row's `influencers` list.

### 5.6 Creating an *action target* relationship

Set the action's type to something that uses an action target (e.g.
`oxd:FollowRoadUser`); the right panel grows an **Action target**
picker. Pick the target there. The arrow then draws as type **T**.

### 5.7 Removing arrows

Open the action / property's right-panel editor and click the `×` on
the chip you want to remove. Toggling the visualisation off in the
video toolbar does **not** delete the relationship.

---

## 6. Editing segments in the timeline

### 6.1 Move and resize

- **Drag the middle** of a segment to move it. Segments can't overlap
  inside the same lane; the tool clamps movement to the available gap.
- **Drag the left or right edge** to resize that end. The cursor turns
  into a column-resize indicator when you're over an edge.
- **Type into the Start / End inputs** in the right panel for exact
  values to one decimal place.

### 6.2 Snap and constraints

- Subtrack segments (conditions, containment, states, actions,
  influences, properties, poses) are **clamped to the parent's
  interval** — you can't drag a condition out of its environment.
- Segments on the same lane can't overlap; new placements snap to the
  nearest gap.
- Signal-head sub-bars **may** overlap (one head can be on while
  another is changing state).

### 6.3 Delete

- Select the segment, press **Delete** or **Backspace** — or click
  **Delete** in the right panel.
- Deleting a parent entity (an agent, a traffic light, etc.) also
  removes its children **and** scrubs any references to its ID from
  `because_of` / `link_to` / `influencers` elsewhere in the bundle.

### 6.4 Mark a segment as illegal

For action and containment segments, an **Mark as Illegal** toggle is
exposed in the right panel — useful for recording that the ego or an
agent did something against the rules (running a red, jaywalking, etc.).

---

## 7. Keypoints — pinning entities to pixels

A **keypoint** marks where an entity sits on the video frame at a
specific time. Keypoints give the bundle a sparse 2D trajectory for
each agent, traffic object, environment, and signal head.

### 7.1 Place and move

1. **Show keypoints** if hidden: click the pin icon in the video
   controls, or press **`B`**.
2. **Select** the entity in the timeline.
3. **Click** on the video at the time you want — a keypoint is
   created at the current playhead, at the click coordinates.
4. **Drag** an existing keypoint to move it. Drag with no movement
   simply selects.
5. **Right-click** a keypoint (on either the video or its diamond in
   the timeline) to delete it.

Between annotated keypoints the player draws **interpolated** markers
as visual hints — these are not real keypoints; clicking them does
nothing.

### 7.2 Where keypoint diamonds appear

In the timeline, keypoint diamonds appear on each entity's main bar
(or, for traffic lights, on each signal-head sub-bar).

### 7.3 Hide keypoints

Press **`B`** or click the pin icon. The overlay disappears and the
diamonds vanish from the timeline. The data is unchanged.

---

## 8. Visualisation controls (video toolbar)

The video player's bottom strip carries every visualisation switch:

```
[▶/⏸] [« step ‹ › step »]  [0.5x | 1x | 2x | 3x]      00:03 / 00:12      [📍 keypoints]   Arrows  [B] [L] [C] [I] [T]
 play  jump  frame  frame jump   playback rate                                                      because  link  contained  influenced  target
       1s    back   fwd   1s
```

Left side:

- **Play / Pause** (or `Space`).
- **Jump back / forward 1 second** (the chevron buttons).
- **Step one frame** (the `‹` / `›` buttons, or `←` / `→`).
- **Playback rate**: `0.5x` / `1x` / `2x` / `3x`.

Centre:

- The **time read-out** `mm:ss / mm:ss`.

Right side:

- **Keypoints toggle** (📍 pin icon, or `B`) — show/hide all keypoints
  in the video and in the timeline.
- **Arrows** — five letter buttons toggle each arrow type. See
  [§5.1](#51-the-five-arrow-types).

Zoom into the video frame with the scroll wheel; when zoomed in you
get a `% zoom` chip and a `Reset zoom` button in the top-right corner
of the video.

---

## 9. Lock and save model in detail

### 9.1 Default-locked

Every clip starts **locked**. Switching clips re-locks. This is by
design — accidental edits are the failure mode the lock guards
against.

- Click **Unlock to edit** in the lock bar; confirm in the inline
  warning row.
- Click **Lock** to re-lock without switching clips.
- If the server was launched with `--read-only`, the unlock button is
  hidden. Edits are blocked everywhere; only navigation and the read
  side of the UI work.

### 9.2 Saves are explicit

There is no autosave. Saves only happen when:

- You click the **Save** button at the top of the right panel; or
- You press **`Cmd/Ctrl+S`**; or
- You confirm **Save** in the unsaved-changes dialog when switching
  clips.

A save is only possible when the bundle is dirty, unlocked, and the
server is writable. The Save button greys out otherwise.

### 9.3 The `.bak` invariant

The **first** save for a clip in a server session writes the original
on-disk content to `<file>.json.bak` before applying your changes. The
backup is stamped once per server session — subsequent saves of the
same clip do not re-stamp it, so the backup always reflects the state
*before this editing session*.

If your edits go bad and you want the original back: stop the server,
copy `<file>.json.bak` over `<file>.json`, and start again.

### 9.4 Atomic writes

Disk writes use `os.replace`, so a crash mid-save never produces a
half-written JSON. Either the old file or the new file is present —
never a truncation.

### 9.5 Unsaved-changes guard

- **Switching clips** while dirty pops a *Save / Discard / Cancel*
  dialog. Save is disabled while locked; you can still Discard.
- **Closing the tab / refreshing** while dirty fires the browser's
  native "Are you sure you want to leave?" prompt.
- **Undo** that lands back at the saved state clears the dirty flag
  automatically.

---

## 10. Status, coverage, and the bottom actions

### 10.1 Status pill

Top of the right panel, next to the Save button:

| Status | Meaning |
|---|---|
| `annotating` | Work in progress. |
| `approved` | Reviewed and approved. |
| `needs_revision` | Reviewer asked for changes (shown as "revision requested"). |
| `disapproved` | Reviewer rejected. |
| `pending` | Default before any explicit status. |

The pill is read-only — it reflects the bundle's `status` field as
stored on disk.

### 10.2 Coverage bars

Two progress bars at the bottom of the right panel show how much of
the clip is covered by Ego Actions and Ego Containment. Useful sanity
check: a fully-annotated clip should be ≈ 100% on both.

### 10.3 Causality summary

A vertical list of every `because_of` edge in the bundle, with the
source and target entity labels. Click an entry to select the source
segment.

### 10.4 Export / Load / Clear

At the very bottom of the right panel:

- **Export JSON** — downloads the current `annotation` slice as
  `<clip_id>_sil_av.json` to your browser's downloads folder. Useful
  for sharing a diff or producing a snapshot outside the on-disk file.
- **Load JSON** — pick a `.json` file. The annotation contents replace
  the in-memory bundle (you still need to **Save** to write to disk).
  Handles both wrapped bundles and raw annotation slices.
- **Clear All Annotations** — wipes everything for this clip. Confirms
  first; then the bundle saves immediately as an empty annotation.

---

## 11. The fresh-annotation flow

If you point the CLI at a directory of *videos* with no JSONs:

1. Each video appears as **unlabelled** in the sidebar.
2. Selecting it loads an empty bundle (schema `2.0.0`,
   `status="annotating"`).
3. Annotate as normal. On the **first save**, the tool creates
   `<dir>/<clip_id>.json`. The sidebar badge flips from amber
   ("unlabelled") to green ("annotated").

---

## 12. Themes

The lock bar's rightmost button is the theme toggle. Click cycles
through **System** (follow OS) → **Light** → **Dark** → **System**.
Your choice persists across reloads (stored in browser local storage).

---

## 13. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Video shows "Video unavailable" | `ffmpeg` / `ffprobe` isn't on `PATH`. Install it; reload. |
| Save button greyed out | Bundle is clean, locked, or the server is read-only. Check the lock bar. |
| `Cmd/Ctrl+S` does nothing | Focus is in a text input or there's nothing to save. |
| Right-click on a track did nothing | The click was on an existing main segment; right-click in **empty** space on the row. |
| Shift+click drew no arrow | The selected segment isn't a valid source for that link type, or the target wasn't a valid target. See §5. |
| Lost annotations to a typo | Use `Cmd/Ctrl+Z` (up to 10 steps), or check `<clip>.json.bak` in the data directory. |
| Switched clip and the unlock button reset | By design — every clip switch re-locks. |

---

## 14. See also

- [`tools/annotator/README.md`](../../tools/annotator/README.md) — install,
  CLI flags, transcode pipeline, architecture.
- [`docs/user/query_language.md`](./query_language.md) — the query DSL,
  for searching the corpus once you've annotated it.
