class RefinementStrategy:
    def build_base_recipe(self, active_phase_names):
        recipe = []
        recipe.append((
            "background",
            {
                "hist": {"Background": {"no. coeffs": 6, "refine": True}},
                "phase_hap_all": {},
                "phase_all": {},
            },
        ))
        recipe.append((
            "scale",
            {
                "hist": {},
                "phase_hap_all": {"Scale": True},
                "phase_all": {},
            },
        ))
        recipe.append((
            "zero",
            {
                "hist": {"Instrument Parameters": ["Zero"]},
                "phase_hap_all": {},
                "phase_all": {},
            },
        ))
        if active_phase_names:
            recipe.append((
                "target_cell",
                {
                    "hist": {},
                    "phase_hap_all": {},
                    "phase_by_name": {active_phase_names[0]: {"Cell": True}},
                },
            ))
        return recipe

    def build_le_bail_recipe(self, active_phase_names, le_bail_phases):
        """Recipe for Le Bail (Pawley) refinement — no atom refinement.
        
        le_bail_phases: dict of {phase_name: True} for phases using Le Bail.
        Le Bail phases get their cell + profile refined; intensities come
        from data extraction, not structure-factor computation.
        """
        recipe = self.build_base_recipe(active_phase_names)
        # After cell, add profile (U) for Le Bail phases
        recipe.append((
            "profile_u",
            {
                "hist": {"Instrument Parameters": ["U"]},
                "phase_hap_all": {},
                "phase_all": {},
            },
        ))
        return recipe

    def build_microstructure_recipe(self, active_phase_names, target_phase_name=None):
        names = list(active_phase_names or [])
        target = target_phase_name or (names[0] if names else None)
        if not target:
            return []
        return [
            (
                "target_size",
                {
                    "hist": {},
                    "phase_hap_all": {},
                    "phase_by_name_hap": {
                        target: {
                            "Size": {
                                "type": "isotropic",
                                "value": 1.0,
                                "refine": True,
                                "LGmix": {"value": 1.0, "refine": False},
                            }
                        }
                    },
                    "phase_all": {},
                },
            ),
            (
                "target_mustrain",
                {
                    "hist": {},
                    "phase_hap_all": {},
                    "phase_by_name_hap": {
                        target: {
                            "Mustrain": {
                                "type": "isotropic",
                                "refine": True,
                                "LGmix": {"value": 1.0, "refine": False},
                            }
                        }
                    },
                    "phase_all": {},
                },
            ),
        ]

    def build_recipe(self, active_phase_names, le_bail_phases=None):
        if le_bail_phases:
            return self.build_le_bail_recipe(active_phase_names, le_bail_phases)
        return self.build_base_recipe(active_phase_names)
