### Theoretical Verification

Thin lens equation:
$$\frac{1}{s} + \frac{1}{s'} = \frac{1}{f}$$

For thick lens, principal planes shift. Effective focal length $f_{eff}$ calculation:
$$\frac{1}{f_{eff}} = (n-1) \left[ \frac{1}{R_1} - \frac{1}{R_2} + \frac{(n-1)d}{n R_1 R_2} \right]$$

* $s$: Object distance.
* $s'$: Image distance.
* $d$: Lens thickness.
* $R_1, R_2$: Surface radii.

Use paraxial approximation ($\sin \theta \approx \theta$) for thin lens validation. Compare numerical result from code above with thin lens formula. Discrepancies indicate spherical aberration inherent in thick lens model.

What is the specific focal length/curvature configuration required for your device?