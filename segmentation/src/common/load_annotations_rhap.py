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


# Motif observé empiriquement sur le corpus (fichiers TextGrids-fev2013) :
# les intervalles "periode" valides portent un label contenant "$L" suivi
# d'un chiffre (1 à 5 observés), parfois décoré de '*', '-' ou '_' en
# préfixe/suffixe (ex: "$L1", "*$L1*", "-$L2-", "$L3_"). Les autres valeurs
# rencontrées ("_", "__", "?", "$" seul) ne sont PAS des unités prosodiques :
# "_"/"__" marquent du silence, "?" et "$" seul marquent une décision
# d'annotation non résolue -- on les exclut plutôt que de les traiter comme
PERIOD_LABEL_PATTERN = re.compile(r"\$L(\d+)")


def _is_period_label(text):
    return bool(PERIOD_LABEL_PATTERN.search(text))


def label_stats(textgrid_path, tier_name="periode"):
    """Utilitaire de diagnostic/QA : compte, pour un fichier donné, combien
    d'intervalles sont reconnus comme unité prosodique valide (motif $L\\d+)
    vs. ignorés (silence, marqueurs d'incertitude "?"/"$" seul, autre).
    Utile pour repérer les fichiers atypiques avant de lancer l'extraction
    sur tout le corpus.
    """
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


def get_ground_truth_rhapsodie(
    textgrid_path,
    tier_name,
    duration=None,
    include_utterance_end=True,
    period_levels=None,
):
    """
    Extrait les frontières prosodiques de référence pour un fichier Rhapsodie.

    Interface volontairement symétrique à `generate_labels.get_ground_truth`.

    Convention observée sur la tier "periode" : les intervalles portant un
    label contenant le motif "$L<chiffre>" (avec décorations éventuelles)
    correspondent à une unité prosodique ; les autres (silence "_"/"__",
    marqueurs d'incertitude "?"/"$" seul) sont ignorés. Une frontière
    prosodique se trouve à la fin (xmax) de chaque intervalle "période".

    Args:
        textgrid_path : chemin vers le fichier .TextGrid
        tier_name : nom de la tier à utiliser (ex: "periode")
        duration : durée totale de l'audio en secondes (pour filtrer les
            frontières hors bornes) ; si None, pas de filtrage
        include_utterance_end : conserve ou non la frontière de fin de fichier
        period_levels : si fourni (ex: {1, 2}), ne garde que les frontières
            des intervalles dont le chiffre après "$L" est dans cet ensemble
            -- utile UNIQUEMENT si "$Lx" s'avère être un code de décision
            hiérarchisé plutôt qu'un identifiant de locuteur (à vérifier).

    Returns:
        Liste triée de frontières en secondes.
    """
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
