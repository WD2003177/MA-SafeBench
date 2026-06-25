from __future__ import annotations

from safebench.scenario.ma.runtime import MATemplateRuntimeScenario


class MultiAgentCutInLeadingVehicle(MATemplateRuntimeScenario):
    """Backward-compatible SafeBench entrypoint for the cut-in MA template."""

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(MultiAgentCutInLeadingVehicle, self).__init__(
            "MultiAgentCutInLeadingVehicle",
            "cut_in",
            world,
            ego_vehicle,
            config,
            timeout=timeout,
        )
