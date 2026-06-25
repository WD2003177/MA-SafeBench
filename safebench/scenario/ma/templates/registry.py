from __future__ import annotations

from typing import Dict

from safebench.scenario.ma.templates.base import MAScenarioTemplate


_TEMPLATES: Dict[str, MAScenarioTemplate] = {}


def register_template(template: MAScenarioTemplate) -> None:
    template.spec_dict()
    _TEMPLATES[template.template_id] = template


def get_template(template_id: str = "cut_in") -> MAScenarioTemplate:
    key = template_id or "cut_in"
    if key not in _TEMPLATES:
        from safebench.scenario.ma.templates.cut_in import CutInTemplate

        register_template(CutInTemplate())
    if key not in _TEMPLATES:
        raise KeyError("Unknown MA scenario template: %s" % key)
    return _TEMPLATES[key]
