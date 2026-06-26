// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from 'vitest'

import { checkSegmentCompleteness } from './completeness'
import {
  addSegmentToAnnotation,
  addSignalHeadToLight,
  annotationToSegments,
} from './timeline-utils'
import type { SilAvAnnotation, TimelineSegment } from './types'

function emptyAnnotation(): SilAvAnnotation {
  return {
    brief_description: '',
    environments: [],
    conditions: [],
    traffic_objects: [],
    traffic_lights: [],
    ego_vehicle: { actions: [] },
    agents: [],
  }
}

describe('optional environment and containment annotation', () => {
  it('does not create placeholder containment rows for new objects, agents, or lights', () => {
    let ann = emptyAnnotation()

    ann = addSegmentToAnnotation(ann, 'obj_0', 'Dirt', 0, 5)
    expect(ann.traffic_objects[0].containment).toEqual([])

    ann = addSegmentToAnnotation(ann, 'agent_0', '', 0, 5)
    expect(ann.agents[0].containment).toEqual([])

    ann = addSegmentToAnnotation(ann, 'light_0', '', 0, 5)
    expect(ann.traffic_lights[0].containment).toEqual([])
    expect(ann.traffic_lights[0].signal_heads[0].env_controlled).toEqual([])
  })

  it('does not create a placeholder condition when adding an environment', () => {
    const ann = addSegmentToAnnotation(emptyAnnotation(), 'env_0', '', 0, 5)

    expect(ann.environments).toHaveLength(1)
    expect(ann.conditions).toEqual([])
  })

  it('does not create placeholder environment control for added signal heads', () => {
    let ann = addSegmentToAnnotation(emptyAnnotation(), 'light_0', '', 0, 5)
    ann = addSignalHeadToLight(ann, 0, 1, 4)

    expect(ann.traffic_lights[0].signal_heads[1].env_controlled).toEqual([])
  })

  it('does not flag otherwise complete objects that have no containment', () => {
    const ann: SilAvAnnotation = {
      ...emptyAnnotation(),
      traffic_objects: [{
        id: 'Object1',
        type: 'Dirt',
        visibility_start_timestamp: '0:0.0',
        visibility_end_timestamp: '0:5.0',
        state_sequence: [{ start_timestamp: '0:0.0', end_timestamp: '0:5.0' }],
        keypoints: [{ timestamp: '0:0.0', x: 10, y: 10 }],
        containment: [],
      }],
    }
    const seg = annotationToSegments(ann).find(s => s.id.startsWith('obj_'))!

    expect(checkSegmentCompleteness(seg, ann).issues).not.toContain('Missing: containment (need >= 1)')
  })

  it('allows explicit containment rows without environment or lane assignment', () => {
    const seg: TimelineSegment = {
      id: 'obj_cont_0_0',
      trackId: 'obj_0',
      label: 'Containment',
      t0: 0,
      t1: 5,
      meta: {
        id: 'ObjectContainment1',
        _isObjContSubtrack: true,
        env_id: '',
        lane_number: 'none',
      },
    }

    expect(checkSegmentCompleteness(seg, emptyAnnotation()).issues).toEqual([])
  })
})
