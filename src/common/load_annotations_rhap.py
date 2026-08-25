import re
from pathlib import Path


def list_tiers(textgrid_path):
    tiers = _parse_textgrid(textgrid_path)
    return {name: len(intervals) for name, intervals in tiers.items()}


def _parse_textgrid(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")

    tiers = {}
    tier_blocks = re.split(r"item \[\d+\]:", text)[1:]

    for block in tier_blocks:
        name_match = re.search(r'name = "(.*?)"', block)
        if not name_match:
            continue
        tier_name = name_match.group(1)

        intervals = []
        for interval_match in re.finditer(
            r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "(.*?)"',
            block,
        ):
            xmin, xmax, txt = interval_match.groups()
            intervals.append((float(xmin), float(xmax), txt.strip()))

        points = []
        for point_match in re.finditer(
            r'number = ([\d.]+)\s*\n\s*mark = "(.*?)"', block
        ):
            time_val, mark = point_match.groups()
            points.append((float(time_val), mark.strip()))

        tiers[tier_name] = intervals if intervals else [
            (t, t, m) for t, m in points
        ]

    return tiers


PERIOD_LABEL_PATTERN = re.compile(r"\$L(\d+)")


def _is_period_label(text):
    return bool(PERIOD_LABEL_PATTERN.search(text))


def label_stats(textgrid_path, tier_name="periode"):
    tiers = _parse_textgrid(textgrid_path)
    intervals = tiers.get(tier_name, [])

    period_count = sum(1 for _, _, txt in intervals if _is_period_label(txt))
    ignored = [txt for _, _, txt in intervals if not _is_period_label(txt)]

    return {
        "total_intervals": len(intervals),
        "period_intervals": period_count,
        "ignored_intervals": len(ignored),
        "ignored_labels_seen": sorted(set(ignored)),
    }


def list_period_labels(textgrid_path, tier_name="periode"):
    """Utilitaire de diagnostic : liste les valeurs de texte distinctes
    présentes dans une tier.
    """
    tiers = _parse_textgrid(textgrid_path)
    intervals = tiers.get(tier_name, [])
    return sorted(set(txt for _, _, txt in intervals))


SILENCE_LABELS = {"_", "__"}


def get_ground_truth_rhapsodie_pause_midpoint(
    textgrid_path,
    tier_name,
    duration=None,
    include_utterance_end=True,
    min_pause_duration=0.0,
    silence_labels=SILENCE_LABELS,
):
    
    tiers = _parse_textgrid(textgrid_path)
    if tier_name not in tiers:
        raise KeyError(
            f"Tier '{tier_name}' absente de {textgrid_path}. "
            f"Tiers disponibles : {list(tiers.keys())}"
        )

    intervals = tiers[tier_name]
    boundaries = []

    for xmin, xmax, txt in intervals:
        if txt not in silence_labels:
            continue
        if (xmax - xmin) < min_pause_duration:
            continue
        boundaries.append((xmin + xmax) / 2)

    if include_utterance_end:
        boundaries.append(duration if duration is not None else intervals[-1][1])

    if duration is not None:
        boundaries = [b for b in boundaries if b <= duration]

    return sorted(set(boundaries))


def get_ground_truth_rhapsodie(textgrid_path,tier_name,duration=None,
    include_utterance_end=True,
    period_levels=None,
):
    
    tiers = _parse_textgrid(textgrid_path)
    if tier_name not in tiers:
        raise KeyError(
            f"Tier '{tier_name}' absente de {textgrid_path}. "
            f"Tiers disponibles : {list(tiers.keys())}"
        )

    intervals = tiers[tier_name]
    boundaries = []

    for xmin, xmax, txt in intervals:
        match = PERIOD_LABEL_PATTERN.search(txt)
        if not match:
            continue
        if period_levels is not None and int(match.group(1)) not in period_levels:
            continue
        boundaries.append(xmax)

    if not include_utterance_end and intervals:
        last_xmax = intervals[-1][1]
        boundaries = [b for b in boundaries if b < last_xmax]

    if duration is not None:
        boundaries = [b for b in boundaries if b <= duration]

    return sorted(set(boundaries))