# ODR ring network: original paper's setup vs. ours

Reference paper: Compte, Brunel, Goldman-Rakic & Wang (2000), "Synaptic Mechanisms and Network Dynamics Underlying Spatial Working Memory in a Cortical Network Model," *Cerebral Cortex* 10(9):910-923. Numbers below are from the paper itself where noted, and from the closest faithful teaching-code port of it (EPFL's `neurodynex3.working_memory_network`, a half-scale Brian2 reimplementation) where the paper's own text doesn't give an explicit value.

This note covers only three things: how input (noise, cue, distractor) is generated, how the ring is structured and connected, and how the memory location is decoded. It leaves out the neuron model itself and the bistability parameter (w0) — those are covered elsewhere.

## 1. Input: background noise, cue, distractor

**Original paper.** Every neuron receives input representing the rest of cortex, modeled as a bombardment of independent Poisson spike trains: about 1000 afferents per neuron, each firing at 1.8 Hz, for a total aggregate rate of about 1.8 kHz per neuron. This drive is delivered exclusively through fast AMPA synapses.

The cue is delivered as a transient *current* injection — not spikes — to the excitatory neurons whose preferred angle is close to the true cue location: an extra 250 ms pulse of about 200 pA. Nothing else changes during the rest of the trial; once the cue period ends, no more selective current is injected.

A distractor, when used, is built exactly the same way as the cue — same strength, same 250 ms duration — the only differences are that it targets a different angular location and it is switched on partway through the delay period instead of at trial start.

**Our changes.** We kept the background noise exactly as Poisson spikes, same mechanism. The one real change is the cue and distractor: we turned them from a smooth, deterministic *current* into *spikes* as well.

Concretely: the same spatial profile that used to set a current amplitude now sets a spike **rate** instead (in Hz). At every simulation step, that rate is used to draw an actual random spike count for every neuron from a Poisson distribution, and that count is converted into current the same way a background spike is (count × a fixed weight-per-spike):

```
rate(angle)  = peak_rate * exp( -0.5 * (angle_from_center / sigma)^2 )      # Gaussian falloff, sigma = width/2.355
count        ~ Poisson( rate(angle) * dt )                                  # dt = 1 ms
current      = count * weight_per_spike
```

So background noise, cue, and distractor are now all the *same kind* of event — an independent Poisson barrage hitting the ring — and differ only in their rate, their location, and when they're switched on. The cue is consequently genuinely stochastic from trial to trial now (like a real afferent volley), rather than a perfectly smooth, identical pulse every time, and it shows up as actual dots on a raster plot instead of only as a current trace.

One added detail specific to using a rate instead of a current: the cue/distractor's Gaussian spatial profile is cut off at 2.5σ from its center. With a current, a very small amplitude far from the center is negligible. With a genuine Poisson rate, a small nonzero rate spread over hundreds of neurons and hundreds of milliseconds adds up to real, spurious spikes across the whole ring — so the tail is truncated to keep the drive actually localized, closer in spirit to the paper's own sharp-edged stimulation window.

## 2. Ring structure and connectivity

**Both the paper and our version** use the same ring layout and the same connectivity rule.

Excitatory ("pyramidal") neurons are arranged around a ring, each with its own preferred angle θ spread evenly over 0-360°. Inhibitory neurons are unstructured — they don't have a preferred angle. The two populations are kept at a fixed 4:1 excitatory-to-inhibitory ratio (paper: 2048/512; our code, like the teaching port we built on, uses 1024/256 by default, same ratio, and we sweep this ratio down much further).

Excitatory-to-excitatory connectivity is the "Mexican hat": neurons with similar preferred angle excite each other strongly, while the connection strength falls to a small, uniform, slightly negative baseline for neurons with very different preferred angles. The weight between a neuron preferring θᵢ and one preferring θⱼ is

```
W(θᵢ - θⱼ) = J⁻ + (J⁺ - J⁻) * exp( -(θᵢ - θⱼ)² / (2·σ²) )
```

with σ the tuning width in degrees and J⁺ the peak (iso-preference) weight. The paper uses σ=18°, J⁺=1.62; we use σ=20°, J⁺=1.6 (the same slightly-rounded values as the teaching port). J⁻ is not a free parameter — it's fixed by requiring the *average* connection weight over the whole ring to equal 1, so the network's total recurrent excitation doesn't depend on how finely the ring is discretized. Solving that condition on a ring of finite circumference gives (this is the port's closed form, a specific realization of the paper's stated normalization principle):

```
tmp = sqrt(2*pi) * sigma * erf(180 / (sqrt(2) * sigma)) / 360
J⁻  = (1 - J⁺ * tmp) / (1 - tmp)
```

All other connections — excitatory-to-inhibitory, inhibitory-to-excitatory, and inhibitory-to-inhibitory — are unstructured in both the paper's main simulations and our version: every neuron in the target population receives the same weight from every neuron in the source population, with no angular dependence at all.

One change on our side, needed because we deliberately test very small rings (down to well under a hundred excitatory neurons, versus the paper's fixed 2048): every weight (recurrent and unstructured alike) is scaled by 1/N of the relevant population, so the total amount of recurrent drive a neuron receives stays roughly the same regardless of how many neurons that drive is split across. The paper and the teaching port don't need this because they only ever run one fixed network size.

## 3. Decoding the memory location

**Both the paper and our version** use the same read-out method: the population vector (Georgopoulos et al., 1986). Each excitatory neuron contributes a unit vector pointing in the direction of its own preferred angle, scaled by its firing rate in the time window being read; summing all of these vectors gives one combined vector whose angle is the network's best estimate of the remembered cue location, and whose length reflects how sharply localized ("bump-like") the population activity is:

```
x = sum_i  rate_i * cos(theta_i)
y = sum_i  rate_i * sin(theta_i)
decoded_angle    = atan2(y, x)
resultant_length = sqrt(x^2 + y^2) / sum_i(rate_i)      # 0 = no bump, 1 = a perfectly sharp bump
```

We use exactly this formula, computed at every simulation step rather than over one fixed analysis window, so we get a full decoded-angle-over-time trace rather than a single number per trial. The one difference is in how we collapse that trace into a single "final" decoded angle for scoring: instead of a plain average over the read-out window, we take a circular mean of the per-step vectors weighted by each step's own resultant length, over the trailing 400 ms of the delay.
