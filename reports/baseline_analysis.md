# Baseline Analysis

The current baseline establishes that the task can be executed as a complete engineering pipeline with deterministic splits, a fixed configuration, a train entrypoint, evaluation, and local inference. Even before any later improvements, this is useful because it turns the project into a reproducible reference point rather than a collection of ad hoc experiments.

At this stage, the baseline mainly tells us whether the dataset, label space, and local workflow are coherent enough for end-to-end detection. If the baseline produces stable artifacts and repeatable evaluation outputs, that already confirms that the task definition is operational and that future changes can be compared against a fixed starting point.

Several limitations are visible immediately. The task depends heavily on small and visually similar objects, especially traffic light variants and state-specific labels. Road scenes also contain strong scale variation, occlusion, clutter, and class imbalance, which means that not all mistakes have the same operational meaning. The baseline should therefore be interpreted as a system sanity check and a first measurement, not as a final statement of readiness.

Some open questions remain on the data and task definition side. It is still important to understand how consistent the label boundaries are between visually similar traffic-light classes, how much domain variation exists across lighting and weather conditions, and whether the current image distribution creates hidden train/test similarities. These are task questions about the dataset and system scope, not yet questions about changing the solution.
