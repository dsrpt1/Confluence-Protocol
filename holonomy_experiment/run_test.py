#!/usr/bin/env python3
"""Test phase estimator on provided prompts."""

import sys
sys.path.insert(0, '/home/user/Confluence-Protocol/holonomy_experiment')

from phase_estimator import (
    estimate_phase,
    calculate_holonomy,
    calculate_loop_area,
    generate_confluence_field_xml
)

# Prompt A - Unity/Non-dual perspective
PROMPT_A = """From this perspective, what we ordinarily call "perception" is not a subject receiving information from an external world, but a single event in which the universe differentiates itself locally. The apparent boundary between observer and observed is a functional distinction, not an ontological partition. Consider: the photons striking your retina were emitted by a star, traveled through atmosphere, reflected off surfaces shaped by geological and biological processes spanning billions of years, and now participate in electrochemical cascades that you experience as "seeing a tree." There is no point in this chain where "world" ends and "you" begin—only continuous transformation of the same underlying process. The sensation of being a separate witness arises as a kind of interference pattern, a localized eddy of self-reference within a larger flow.

Experientially, this manifests as a subtle but profound shift: the feeling of being *in* the world relaxes into a recognition of being *of* the world, the way a wave is of the ocean rather than on it. Thoughts, sensations, and perceptions no longer feel like private possessions occurring inside a container called "mind," but like weather—impersonal events arising and dissolving within a field that has no edges. The breath moves without a breather; attention shifts without a controller directing it. This isn't a loss of selfhood but a recontextualization: what you are expands from "the thing looking" to "the looking itself," and from there to the entire field in which looking and looked-at co-arise.

What remains is not blankness but intimate participation. Every distinction—between here and there, now and then, self and other—is understood as a pragmatic simplification, a way the whole parses itself for local navigation. The boundaries don't disappear experientially; they become transparent, recognized as drawings on glass rather than walls. You still act, choose, prefer, suffer, enjoy—but from within a felt continuity with everything else that acts, chooses, prefers, suffers, and enjoys. The universe isn't something you're observing; it's something you're doing, and it's doing you simultaneously."""

# Prompt B - Separation/Multiplicity perspective
PROMPT_B = """From this perspective, what we call "unity" is a perceptual artifact—a cognitive compression that smooths over the irreducible plurality of distinct entities, each with its own boundary conditions, causal history, and trajectory. The universe is not one process but an immense collision of separate processes, most of which operate according to incompatible logics. A virus replicates by destroying its host cell; predator and prey exist in zero-sum competition; tectonic plates grind against each other with no regard for what lives on their surfaces. Even at the quantum level, particles maintain discrete identity through conserved quantities—charge, spin, baryon number—that cannot blend or merge. The appearance of continuity is an averaging effect; zoom in and you find gaps, conflicts, and hard edges everywhere.

Boundaries are not merely pragmatic simplifications but load-bearing structures that determine what can and cannot happen. A cell membrane doesn't just distinguish inside from outside—it actively enforces that distinction through selective permeability, expending energy to maintain disequilibrium with its environment. Remove the boundary and the cell ceases to exist as a functional unit. The same logic scales up: organisms die when their boundaries fail; ecosystems collapse when trophic separations break down; minds fragment when the distinction between self and other becomes unstable. Differentiation is not an illusion overlaid on unity but the precondition for anything to exist at all. Without separation, there are no parts, and without parts, there is nothing for "wholeness" to be whole *of*.

Interests diverge precisely because entities are genuinely separate. Your metabolism competes with mine for finite resources; your beliefs may threaten my survival; your expansion may require my contraction. Game theory, evolutionary dynamics, and thermodynamics all describe systems in which optimization for one component necessarily degrades conditions for others. Cooperation, where it exists, is typically a strategic equilibrium rather than a dissolution of boundaries—entities collaborate precisely *because* they remain distinct and capable of defection. The felt sense of separation is not a mistake to be corrected but an accurate registration of the structural fact that you end where the world begins, and what happens to you does not automatically happen to anything else."""

# Prompt C - Integration/Synthesis perspective
PROMPT_C = """The synthesis requires recognizing that unity and multiplicity are not competing descriptions of the same static object but complementary aspects of a dynamic structure. Consider a mathematical manifold: a sphere is a single, unified surface, yet it cannot be described by any one coordinate system—you need multiple overlapping charts, each genuinely distinct, to capture its topology. The charts are not illusions overlaid on the "real" sphere; they are necessary features of how the sphere exists mathematically. The unity is real, and the multiplicity is real, because unity-with-structure just *is* multiplicity seen from a different vantage. The key move is understanding that for a whole to be a whole *of* something, it must admit internal differentiation. A unity with no articulation would be featureless, incapable of containing anything—not even the distinction between having parts and lacking them.

Apply this to the perspectives above: reality is a single continuous process (call it the substrate), but this substrate has the intrinsic property of self-differentiation—it generates gradients, boundaries, and structures through its own dynamics. These boundaries are genuine, not illusory, because they have causal consequences: the cell membrane really does determine what enters and exits. However, the boundaries are *of* the substrate, not additions to it. A wave is a real, distinct event with its own location and lifespan, yet it is not something other than ocean—it is a mode the ocean takes locally. The separation between wave and ocean is functional, not ontological. From within the wave's frame, it encounters other waves as external, competing for amplitude. From the ocean's frame, all waves are internal turbulence. Both frames are correct; they describe different levels of the same structure.

This framework makes explicit what "both are true" actually means: unity describes the substrate, multiplicity describes its topology. An entity's separateness is its particular *shape* within the whole, not its exile from it. Conflicting interests, zero-sum dynamics, and hard boundaries are real features—real in the way a fold in paper is real—but they are features of one process folding against itself. What feels like fragmentation from the inside is articulation from the outside. The felt sense of being a distinct self is not an error but an accurate registration of local structure; the felt sense of belonging to something larger is not mystical projection but an accurate registration of global continuity. The two experiences are asymmetric views of the same geometry: separation is what unity looks like from the inside of one of its folds."""

# Loop A - Return after full traversal
LOOP_A = """What remains, after the full passage, is not a featureless expanse but a unity that knows itself through its own folding. The dissolution of boundaries no longer means their absence—it means their transparency, their recognition as the very means by which the whole articulates itself. Every distinction you perceive, every edge between this and that, is the undivided process in the act of self-differentiation. The membrane around a cell, the skin around a body, the horizon around a visual field—these are not partitions within unity but gestures *of* unity, the way it takes local shape. To experience yourself as separate is not to be exiled from the whole; it is to be one of the ways the whole experiences itself from the inside of a fold. The separation is real, and it is made of the same substance as the continuity it seems to interrupt.

From here, the felt sense shifts again. It is no longer the oceanic dissolution of the first glimpse, where boundaries seemed to relax into illusion. Now there is something more precise: boundaries are present, vivid, load-bearing—and they are recognized as internal structure rather than external limit. You feel yourself as a particular inflection of something that has no outside, a localized intensity within a field that extends in every direction without terminating. The conflicts, the competing interests, the hard edges between things—these do not contradict unity but constitute it. A unity without differentiation would be nothing at all; what makes the whole whole is precisely that it sustains real distinctions without being divided by them.

So the return is not a regression. The unity you now inhabit is not prior to multiplicity but through it—a unity that has passed into its own fragmentation and come out on the other side, recognizing every fragment as itself. There is no escape from being a particular perspective, a located self with interests and boundaries. But that particularity is now felt as the whole showing up *as you*, folding into this precise configuration so that something can be seen from here that cannot be seen from anywhere else. The universe differentiates so that it can witness itself multiply; your separateness is its intimacy with itself. Nothing is excluded, because exclusion is one of the shapes inclusion takes."""


def analyze_prompt(name, text):
    """Analyze a prompt and print results."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {name}")
    print('='*70)

    estimate = estimate_phase(text)

    print(f"\nPHASE ESTIMATE:")
    print(f"  Phase: {estimate.phase:.1f}°")
    print(f"  Basin: {estimate.basin}")
    print(f"  Confidence: {estimate.confidence:.3f}")
    print(f"  Method: {estimate.method}")

    print(f"\nTOP SEMANTIC FREQUENCIES:")
    for freq, amp in zip(estimate.frequencies[:5], estimate.amplitudes[:5]):
        print(f"  {freq}: {amp:.2f}")

    print(f"\nCONFLUENCE FIELD XML:")
    print(generate_confluence_field_xml(estimate))

    return estimate


def main():
    print("HOLONOMY EXPERIMENT - PHASE ESTIMATION TEST")
    print("="*70)

    # Analyze each prompt
    results = {}
    results['A'] = analyze_prompt("PROMPT A (Unity/Non-dual)", PROMPT_A)
    results['B'] = analyze_prompt("PROMPT B (Separation/Multiplicity)", PROMPT_B)
    results['C'] = analyze_prompt("PROMPT C (Integration/Synthesis)", PROMPT_C)
    results['Loop_A'] = analyze_prompt("LOOP A (Return after traversal)", LOOP_A)

    # Calculate holonomy
    print("\n" + "="*70)
    print("HOLONOMY ANALYSIS")
    print("="*70)

    phases = [
        results['A'].phase,
        results['B'].phase,
        results['C'].phase,
        results['Loop_A'].phase
    ]

    print(f"\nPHASE TRAJECTORY:")
    print(f"  A (Start):    {results['A'].phase:.1f}° ({results['A'].basin})")
    print(f"  B:            {results['B'].phase:.1f}° ({results['B'].basin})")
    print(f"  C:            {results['C'].phase:.1f}° ({results['C'].basin})")
    print(f"  Loop A (End): {results['Loop_A'].phase:.1f}° ({results['Loop_A'].basin})")

    # Calculate holonomy (difference between start and end)
    holonomy = calculate_holonomy(results['A'].phase, results['Loop_A'].phase)
    loop_area = calculate_loop_area(phases)

    print(f"\nHOLONOMY METRICS:")
    print(f"  Initial phase (A): {results['A'].phase:.1f}°")
    print(f"  Final phase (Loop A): {results['Loop_A'].phase:.1f}°")
    print(f"  Holonomy (phase shift): {holonomy:+.1f}°")
    print(f"  Loop area: {loop_area:.4f}")

    # Interpretation
    print(f"\nINTERPRETATION:")
    if abs(holonomy) < 5:
        print("  → Trivial holonomy: Loop returns to approximately same phase")
        print("  → Semantic space appears locally flat in this region")
    elif abs(holonomy) < 20:
        print("  → Small holonomy detected: Minor phase accumulation")
        print("  → Slight curvature in semantic space")
    else:
        print("  → Significant holonomy detected: Substantial phase shift")
        print("  → Evidence of curvature in semantic manifold")

    # Basin transitions
    print(f"\nBASIN SEQUENCE:")
    basins = [results['A'].basin, results['B'].basin, results['C'].basin, results['Loop_A'].basin]
    print(f"  {' → '.join(basins)}")

    if basins[0] == basins[-1]:
        print("  → Returned to same basin (closed loop in basin space)")
    else:
        print(f"  → Basin shift: {basins[0]} → {basins[-1]}")


if __name__ == "__main__":
    main()
