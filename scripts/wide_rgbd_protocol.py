"""Data-only identity rules for the one fixed wide RGB-D capture revision."""
import copy
import json

CAPTURE_OVERRIDES = dict(window_radius=10, pyramid_levels=4)
PARENT_TO_WIDE = dict(spatialrust_rgbd_pnp_cpu='spatialrust_wide_rgbd_pnp_cpu',
    spatialrust_adaptive_keyframe_rgbd_pnp_cpu='spatialrust_wide_adaptive_rgbd_pnp_cpu')


def exact_json(value):
    """Reject bool/int and int/float aliases as well as value changes."""
    return json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False)


def validate_capture_transition(parent,plan):
    expected = PARENT_TO_WIDE.get(parent.get('estimator'))
    if expected is None or plan.get('estimator') != expected:
        raise ValueError('capture revision requires its exact frontend parent identity')
    settings = parent['controls'].get('rgbd_odometry',{})
    if type(settings.get('window_radius')) is not int or settings['window_radius'] != 5 or type(settings.get('pyramid_levels')) is not int or settings['pyramid_levels'] != 3:
        raise ValueError('capture parent must retain the original radius/level setting')
    controls = copy.deepcopy(parent['controls'])
    controls['rgbd_odometry'].update(CAPTURE_OVERRIDES)
    if exact_json(plan.get('controls')) != exact_json(controls) or any(
            exact_json(plan.get(key)) != exact_json(parent[key])
            for key in ('sequence','calibration','limits','acceptance')):
        raise ValueError('only window radius and pyramid levels may change; all admission limits stay fixed')
