"""Independent presentation masks; caller data and execution stay unchanged.

STATUS: active — feedback
"""
from dataclasses import asdict, dataclass, is_dataclass
import json


@dataclass(frozen=True)
class FeedbackView:
    images: bool = True
    trajectory: bool = True
    progress: bool = True
    structured_goals: bool = True


def apply_view(context, images, view):
    if view == FeedbackView():
        return context, images

    def clean(value):
        if is_dataclass(value):
            value = asdict(value)
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()
                    if not (not view.progress and (k.startswith('progress') or k == 'learned_progress'))
                    and not (not view.trajectory and k == 'trajectory')}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    result = clean(context)
    selected = images if view.images else ()
    if not view.images:
        result['image_order'] = []
    if not view.structured_goals:
        # Preserve task semantics in text and routing IDs for valid requests.
        # Remove duplicate machine-readable target descriptions/coordinates.
        result['targets'] = [{'id': t['id']} for t in result['targets']]
        # GoalToolAdapter embeds the JSON catalog in task text too.
        prefix, marker, suffix = result['task'].partition(' Goals: ')
        if marker:
            goals, end = json.JSONDecoder().raw_decode(suffix)
            result['task'] = prefix + ' Goals: ' + '; '.join(
                f"{g['object_name']} ({g['object_id']}) to {g['destination_id']}"
                for g in goals) + suffix[end:]
    return result, selected
