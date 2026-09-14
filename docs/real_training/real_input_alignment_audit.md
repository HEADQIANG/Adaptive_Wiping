# Real Input Alignment Audit

> 目录已分类迁移：当前主流程见 [本类操作入口](README.md)。代码块使用新运行目录，历史结果引用归档；新配置、源码与旧实验不可混作原地续训。

Read-only offline diagnostics for the saved exploration and eight demonstrations.
No robot/serial SDK is imported, no tare is performed, and no original recording,
model, calibration or training input is changed. This is not retraining.

## Run

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wp/miniconda3/envs/clean/bin/python -m unittest discover \
  -s tests -p 'test_real_input_alignment.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/wp/miniconda3/envs/clean/bin/python -m scripts.real_training.tools.audit_real_input_alignment \
  --output runs/real_training/real_input_alignment_audit_v1
```

Output must not exist. Use a new directory for repeats. The script verifies raw
and prepared hashes through the existing loaders and all collection source hashes,
then saves manifest/status/report JSON and force-domain/XY plots. Original source
files are hashed before and after. No held-out simulation data are used for bounds;
force normalization and reference distribution use only simulation training rows.

## Interpretation

Force norms are rotation-invariant and independent of wrench origin. A norm
mismatch cannot be fixed by changing signs/axes or torque lever arms alone.
Start-subtracted plots are counterfactuals, not calibrated contact force: that
operation also subtracts gravity and any initial contact. A force divided by9.81
is only a mass equivalent, not a measured tool mass or a diagnosis of unit error.
The existing manufacturer conversion is unchanged. Loaded trajectories cannot
independently identify gravity, bias and contact; a contact-free orientation-rich
calibration record would be needed to separate them.

XY comparisons use the same component RMSE definition as downstream training.
The 25-time absolute leave-one-out mean reproduces the saved baseline. Start
position is sampled causally at time0, not fitted from future targets. Whole-path
centroid and arc-length alignment use held-out trajectory information and are
explicitly labeled oracle/shape diagnostics, not predictive validation scores.
Onset is the first displacement sustained over five 100Hz grid samples, with
1/2/5mm thresholds. This is motion onset, not contact onset. Onset-aligned curves
use future recorded onset and a common, non-extrapolated duration; compare them
only with the reported same-duration unaligned diagnostic.

Recorded robot orientation is SDK orientation, not sensor orientation. It bounds
rigidly mounted sensor orientation *changes* but does not identify the missing
absolute sensor-to-SDK rotation. The exploration completion log records command
completion, not acceptance of the measured path. Historical KWR75 labels remain
unchanged; the user's subsequently identified sensor is KWR52-TiS, S/N1F06303.

## Completed Results

`archive/real_training/real_input_alignment_audit_v1` completed. Four new synthetic tests pass,
and both generated figures were inspected. Original inputs and collection hashes
are unchanged. Values below use the encoder's identical order2/10Hz filtering
on both simulation training trajectories and the real exploration.

| Force finding | Value |
|---|---:|
| Simulation force norm maximum after filtering | 3.241 N |
| Real force norm range after filtering | 17.580-30.357 N |
| Real initial raw force norm | 28.503 N |
| Simulation unloaded force norm, approximately | 0.2943 N |
| Maximum force change after counterfactual start subtraction | 11.236 N |
| Maximum exploration orientation change | 1.041 degrees |

All400 real filtered force norms exceed the simulation train maximum. This rules
out a pure rigid-frame rotation/origin-shift fix for the raw force magnitude
mismatch. The raw unfiltered simulation maximum is3.384N; do not mix it with the
filtered maximum. Neither28.503N nor the counterfactual11.236N is an identified
contact load. Units remain manufacturer-converted N/Nm; no divide-by9.81 fix is
supported. The unloaded simulation corresponds to roughly30g of sensed weight,
not evidence of the real mounted tool's mass. No mass, bias or COM was fitted to
loaded real trajectories.

### Exploration Motion

The original log explicitly says command completion, `encoder_ready=false` and
`position_rms_within_simulation_1mm=false`. Its motion is not a precise physical
reproduction of the nominal exploration. Axis-wise RMS errors are X0.075,
Y6.455 and Z1.434mm; logged vector RMS6.612mm. At2/3/4s the measured changes from
the initial pose were respectively:

| Time | Y displacement | Z displacement | Nominal Y / Z |
|---|---:|---:|---:|
| 2s | 0.0002mm | -17.855mm | 0 / -20mm |
| 3s | 39.166mm | -20.074mm | 50 / -20mm |
| 4s | 10.509mm | -20.105mm | 0 / -20mm |

The 400 scheduled observations are valid recorded observations, not proof of400
independent sensor frames or precise tracking. A software import must not relabel
them as physically accepted encoder exploration.

### Demonstration Alignment

Initial X positions span14.647mm; Y positions span50.745mm. With a sustained2mm
displacement threshold, motion starts at [2.03,1.32,2.01,1.52,1.15,1.68,2.51,3.09]s.
The conclusion is robust at1mm and5mm thresholds. Maximum causal pose ages are
10.10ms or less and received-FT ages18.12ms or less: these recorded sampling ages
do not explain seconds of differences in motion onset. This does not identify
unrecorded sensor-internal latency.

| Diagnostic | Coordinate RMSE |
|---|---:|
| Absolute XY, 25 policy times, leave-one-out mean | 17.737mm |
| Initial-position relative XY, same25 times | 17.285mm |
| Relative XY, common6.91s, before onset alignment | 18.818mm |
| Relative XY, common6.91s, after2mm-onset alignment | 17.902mm |
| Complete-path arc-length shape comparison, oracle | 14.517mm |

Only start-relative scoring could be turned into a causal start-conditioned
model without seeing future targets; this audit does not do so. Its2.55% RMSE
improvement is too small to call the XY problem fixed. Onset alignment improves
the matched-duration diagnostic4.87%, but uses future motion of the held-out
demonstration. Arc-length normalization still leaves differing path shapes and
endpoints. Path lengths range210-331mm and Y extents96-141mm. No episode has been
discarded or selected as the "correct" trajectory.

Within each demo, SDK orientation changes reach12.834-17.197degrees. The recorded
SDK end is not the wiping-face TCP; orientation-dependent offsets and gravity
therefore matter. The nominal sensor-to-face46.84mm offset is not an SDK offset,
so it cannot simply be added along SDK Z to fix these trajectories. All eight
demos share exactly one sponge embedding; a deterministic sponge-only XY network
necessarily returns one trajectory rather than eight individually timed paths.

## What Can Be Done Next

1. Keep existing data, models and warnings. Do not apply static start-tare,
   axis swapping, normalization clipping or time warping as a silent correction.
2. First necessary onsite step: an attended, contact-free multi-orientation
   static recording with unchanged mount/sponge, raw FT and SDK pose logged
   together, without manual tare. Several safely reachable distinct orientations
   are needed (plan approximately6-8, subject to conditioning checks), including
   a repeated initial pose for drift assessment. A single suspended pose cannot
   separate sensor bias from gravity. Do not load/touch the sensor-side tool or
   sponge during settled recording. Use trained onsite supervision and the
   manufacturer's supported motion mode; no automatic motion is authorized by
   this audit and no pose/angle is prescribed without a workspace safety check.
3. Fit and validate bias/gravity and sensor-axis relationships only after those
   records exist, checking identifiability and held-out orientations. Keep any
   estimate unapproved until validated. Resolve the physical SDK-to-wiping-TCP
   reference before producing calibrated training positions. Do not infer that
   unloaded gravity calibration identifies the wiping-face contact origin.
4. Recheck the existing exploration under the validated convention. If loading
   or tracking still mismatches, use a separately supervised exploration after
   addressing its cause. An original4s completed command sequence is not an
   automatic substitute for this verification.
5. Then decide whether existing demos can support a start-conditioned/phase-aware
   model, or whether a small consistent reference set is needed. That choice
   changes the current sponge-only model contract and should be explicit. There
   is no evidence-based reason to discard and recollect all eight immediately.

The next onsite calibration is a new activity requiring operator readiness; it
has not been started. No additional training, policy export, calibration flag
change, sensor tare, hardware connection or movement occurred in this audit.
