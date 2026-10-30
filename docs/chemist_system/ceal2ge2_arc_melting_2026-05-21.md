# CeAl₂Ge₂ Arc Melting Synthesis Protocol
Date: 2026-05-21
Status: PROVISIONAL — based on similar-system inference, not experimental validation

## Material Classification
- **Category**: Intermetallic (ThCr₂Si₂-type, space group I4/mmm)
- **Ground state**: CeAlGe (1:1:1) and Ce₂Al₃Ge₄ are thermodynamically stable
- CeAl₂Ge₂: E_above_hull = 12.4 meV/atom (0K, MP DFT) — marginally unstable at 0K
- **Key challenge**: CeAl₂Ge₂ is metastable at 0K; requires kinetic stabilization at high T

## Similar-System Evidence
- Ce-Ag-Ge @ 850°C: Phase relations characterized (Journal of Phase Equilibria)
- Ce-Pt-Ge @ 600°C: Isothermal section available
- BaPd₂As₂ (ThCr₂Si₂-type): self-flux single crystal growth @ 700-900°C
- RIr₂Si₂ (R=Pr,Nd): arc melting + annealing @ 800-900°C for 7-10 days

## Recommended Protocol
```json
{
  "material": "CeAl2Ge2",
  "method": "arc_melting + annealing",
  "precursors": ["Ce (99.9%)", "Al (99.99%)", "Ge (99.999%)"],
  "molar_ratio": "Ce:Al:Ge = 1.0:2.0:2.0",
  "arc_melting": {
    "atmosphere": "UHP Ar (< 1 ppm O2)",
    "current_A": "200-300",
    "melting_sequence": "Ce first → Al second → Ge last",
    "flips": 3,
    "al_excess_wt%": "3-5 (Al volatile at arc T)"
  },
  "annealing": {
    "temp_C": "800-900",
    "duration_h": "168-240 (7-10 days)",
    "atmosphere": "Ar or high vacuum",
    "crucible": "Ta tube or quartz ampule"
  },
  "characterization": {
    "PXRD": "20-90° 2θ, step 0.01°, 2s/step. Expected (002) ~28.5°, (110) ~40.2°, (112) ~56.8°",
    "SEM-EDS": "Verify Ce₁Al₂Ge₂ ±2 at%",
    "competing_phases": "Ce₂Al₃Ge₄ (XRD @ ~30.1°, 41.5°), CeAlGe"
  },
  "confidence": 0.5,
  "ca_status": "NOT APPLICABLE — intermetallic requires elemental precursors"
}
```

## Why rxn-ca CA Does Not Apply
- rxn-ca CA is designed for oxide solid-state diffusion (ErTa₇O₁₉ type)
- CeAl₂Ge₂ is a metallic intermetallic — no oxide pathway exists
- Elemental precursors (Ce, Al, Ge) cannot be processed by rxn-ca's ReactionLibrary framework

## MP Phase Diagram Data (Ce-Al-Ge)
- CeAl₂Ge₂: E_hull = 12.4 meV/atom (marginally unstable)
- Ce₂Al₃Ge₄: E_hull = 0 (stable)
- CeAlGe: E_hull = 0 (stable, 1:1:1 composition)

## No-CALPHAD Fallback Summary
- Ce-Al-Ge ternary system: NO public thermodynamic parameters (tdb file)
- pycalphad cannot model this system without a database
- Recommendation: Use experimental TTT curve measurement to establish processing window
