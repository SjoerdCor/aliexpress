"""Build the JSON view model for the browser sociogram."""

# The view builder keeps the projection in one small, readable pass over the canonical data.
# pylint: disable=too-many-locals

from dataclasses import dataclass


@dataclass(frozen=True)
class SociogramNode:
    """A student node for the browser sociogram."""

    id: str
    label: str
    received_preference_score: float
    size: float


@dataclass(frozen=True)
class PreferenceEdge:
    """One original, directed student-to-student preference."""

    id: str
    source: str
    target: str
    weight: float
    kind: str
    line_width: float = 2.0


@dataclass(frozen=True)
class LayoutRelation:
    """One undirected, derived relation used only to position two learners."""

    source: str
    target: str
    kind: str
    strength: float
    ideal_distance: float


@dataclass(frozen=True)
class SociogramView:
    """Flat, JSON-serialisable data needed to render the browser sociogram."""

    nodes: list[SociogramNode]
    preferences: list[PreferenceEdge]
    layout_relations: list[LayoutRelation]


def build_sociogram_view(preference_data) -> SociogramView:
    """Build student nodes and visible preference arrows from canonical preference data.

    Destination-group preferences are intentionally left out: only preferences whose
    target is another known student belong in the sociogram. Every retained input row
    becomes its own directed edge, including a negative preference.
    """
    student_keys = list(preference_data.students_info)
    student_set = set(student_keys)
    student_display = preference_data.student_display
    unique_name = preference_data.unique_name

    incoming_scores = {student: 0.0 for student in student_keys}
    edges = []
    pair_preferences = {}
    for position, (index, row) in enumerate(preference_data.preferences.iterrows()):
        source = index[0]
        target = row["Waarde"]
        if source not in student_set or target not in student_set:
            continue
        weight = float(row["Gewicht"])
        incoming_scores[target] += _clip_received_preference(weight)
        edges.append(
            PreferenceEdge(
                id=f"preference-{position}",
                source=source,
                target=target,
                weight=weight,
                kind="negative" if weight < 0 else "positive",
                line_width=_edge_line_width(weight),
            )
        )
        if source != target:
            pair = _ordered_pair(source, target, student_keys)
            pair_preferences.setdefault(pair, []).append((source, target, weight))

    nodes = [
        SociogramNode(
            id=student,
            label=unique_name.get(student, student_display.get(student, student)),
            received_preference_score=score,
            size=_node_size(score),
        )
        for student, score in incoming_scores.items()
    ]
    layout_relations = [
        _build_layout_relation(source, target, preferences)
        for (source, target), preferences in pair_preferences.items()
    ]
    return SociogramView(
        nodes=nodes, preferences=edges, layout_relations=layout_relations
    )


def _ordered_pair(source: str, target: str, student_keys: list[str]) -> tuple[str, str]:
    """Return a stable pair orientation based on the canonical learner order."""
    order = {student: index for index, student in enumerate(student_keys)}
    if order[source] <= order[target]:
        return source, target
    return target, source


def _build_layout_relation(source, target, preferences) -> LayoutRelation:
    """Combine the at-most-two directed preferences into one relation.

    The preference validator guarantees that a learner can mention another learner at
    most once, so the pair contains at most one weight per direction.
    """
    weights_by_source = {
        preference_source: weight for preference_source, _, weight in preferences
    }
    negative_weights = [
        abs(weight) for weight in weights_by_source.values() if weight < 0
    ]

    if negative_weights:
        kind = "negative"
        strength = max(negative_weights)
    elif source in weights_by_source and target in weights_by_source:
        kind = "mutual_positive"
        strength = (weights_by_source[source] + weights_by_source[target]) / 2
    else:
        kind = "positive"
        strength = next(iter(weights_by_source.values()))

    return LayoutRelation(
        source=source,
        target=target,
        kind=kind,
        strength=float(strength),
        ideal_distance=_ideal_distance(kind, strength),
    )


def _ideal_distance(kind: str, strength: float) -> float:
    """Map relation strength into a non-overlapping, bounded distance band."""
    bounded_strength = _bounded_strength(strength)
    bands = {
        "mutual_positive": (60.0, 90.0),
        "positive": (110.0, 145.0),
        "negative": (220.0, 290.0),
    }
    closest, furthest = bands[kind]
    if kind == "negative":
        return closest + (furthest - closest) * bounded_strength
    return furthest - (furthest - closest) * bounded_strength


def _bounded_strength(strength: float) -> float:
    """Compress non-negative strength to [0, 1): x/(x+1) grows but approaches 1."""
    return strength / (strength + 1.0)


def _node_size(received_preference_score: float) -> float:
    """Map a received-preference score to a restrained Cytoscape node diameter."""
    bounded_score = max(-2.0, min(10.0, received_preference_score))
    return 36.0 + 28.0 * (bounded_score + 2.0) / 12.0


def _edge_line_width(weight: float) -> float:
    """Map absolute preference strength to a bounded, non-linear line width."""
    return 2.0 + 6.0 * _bounded_strength(abs(weight))


def _clip_received_preference(weight: float) -> float:
    """Limit one contribution to the received score without changing the edge weight."""
    return max(-2.0, min(2.0, weight))
