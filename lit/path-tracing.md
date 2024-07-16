# Path tracing
This is where all the physics happens. We need to generate random numbers.

```rust #imports
extern crate rand;
use rand::Rng;
```

```rust #constants
use std::f64::consts::PI;
```

The `radiance` function computes how many photons are traveling at a certain position in space from a certain direction.

```admonish info title="Recursion"
One major change with respect to the original SmallPT is the recursion. SmallPT uses true recursion to compute the radiance of a ray. In Rust, this has led to some instances where a stack overflow was triggered. We may use a stack based implementation to prevent this from happening, but this has proven to be quite a bit slower. I ended up using a `loop`, modifying the traced path in place. Only in the case of partial reflection do we recursively call the `radiance` function.

The result of each recursive radiance computation goes into an affine transformation (\\(ax + b\\)). We may compose two transformations

\\[(x \to ax + b) \circ (y \to cy + d) = y \to a(cy + d) + b = y \to acy + ad + b,\\]

meaning that if we express an affine transformation as a pair \\((a, b)\\) and a second \\((c, d)\\), we have \\((a, b) \circ (c, d) = (ac, ad + b)\\). This means we have a compact way to codify the contribution of each scattered ray.
```

```rust #path-tracing
fn radiance(ray: &mut Ray, mut depth: u16) -> RGBColour {
    let mut rng = rand::thread_rng();
    let mut colour = WHITE;
    let mut output = BLACK;

    loop {
        <<do-intersect>>
        <<russian-roulette-1>>
        <<compute-normal>>
        <<do-reflect>>
    }
}
```

The second argument keeps track of how deep we are tracing. It is used as a control to switch between sampling methods. One method is to reduce the brightness of the ray at every reflection off a diffuse object until we hit a light source. The second method, also known as *Russion Roulette*, is to keep the brightness of the ray constant, but only reflect with a probability given by the colour of the object. The first method will always give a nice smooth image but may take a long time wasted on very dim rays. The Russian Roulette wastes less time per sample, but produces grainy images at low sample rates. That is why SmallPt switches sampling methods if we are deeper than \\(n\\) reflections.

First, we need to see if the ray intersects any object in the scene; if not, we return the colour `BLACK`.

```rust #do-intersect
let hit = intersect(&ray);
if hit.is_none() { return output; }
let (distance, object) = hit.unwrap();
output = output + object.emission * colour;
```

## Russian Roulette 1
The colour \\(f\\) of an object reduces the radiance of a ray compared to the radiance of the reflected ray.

\\[r = f r_{\rm refl}.\\]{#eq:reflected-radiance}

The first Russian Roulette happens at an integration depth of 5. The value \\(p\\) is the probability of the ray being reflected. The value of \\(p\\) can be anything between \\(0\\) and \\(1\\), and the math would still work out, however we choose it to be the maximum value of the colour of the object. Once the ray has overcome the odds of being absorbed, we have to renormalize the colour. If \\(p = 1\\) the colour should remain the same. In other words,

\\[r = \frac{1}{N}\sum_{\rm N} p f' r_{\rm refl} = f r_{\rm refl},\\]{#eq:russian-roulette}

meaning that \\[f' = f / p\\]. If the ray got absorbed, the radiance equals the emission of the object.

```rust #russian-roulette-1
let mut f = object.colour;
let p = f.max();
depth += 1;
if depth > 5 {
    if rng.gen::<f64>() < p {
        f = f * (1. / p);
    } else {
        return output;
        // current = stack.pop();
        // continue;
    }
}
```

## Normals
Now that we know that we hit an object, we need to compute the normal vector. Let \\(x\\) be the position where the ray hits the object, and \\(\vec{n}\\) be the normal vector (outward pointing) of the object.

```rust #compute-normal
let x = ray.origin + ray.direction * distance;
let n = (x - object.position).normalize();
```

It could be that we're inside the object. In that case, the normal of reflection is opposite the normal of the object.

```rust #compute-normal
let n_refl = if n * ray.direction < 0. { n } else { -n };
```

## Reflection
We're at the point that we need to compute how much light is reflected. Of course, this depends on the type of material that the object is made of. SmallPt has three material types, *diffuse*, *specular*, and *refractive*, that each have their own physics.

```rust #do-reflect
match object.reflection {
    Reflection::Diffuse => {
        <<diffuse-reflection>>
    }
    Reflection::Specular => {
        <<specular-reflection>>
    }
    Reflection::Refractive => {
        <<refractive-reflection>>
    }
};
```

### Diffuse reflection
There are many types of diffuse reflection. You could imagine a surface where rays have equal probability of reflecting to any direction. This would mean sampling vectors on a hemisphere. We have a uniform probability over longitude:

```rust #diffuse-reflection
let phi = 2.*PI * rng.gen::<f64>();
```

Taking \\(\theta\\) to be the angle of incidence to the normal of the surface, we have a \\(p(\theta) \sim \sin \theta\\) probability over latitude. The inverse CDF sampling method then gives than \\(\cos \theta\\) has a uniform distribution in the interval \\([0, 1]\\).

However, there is a second effect. If you shine a uniform bundle of light on a surface at an angle, the light intensity drops with a factor \\(\cos \theta\\). The combination of sampling the hemisphere and the lighting is known has *cosine-weighted sampling*, and there is a trick called *Malley's Method*.
We can sample points on a uniform disc, and project those onto the hemisphere {{#cite Pbr-13.6.3}}.

On a disc we have \\(p(r) \sim r\\), so \\(p(r^2) \sim 1\\),

```rust #diffuse-reflection
let r2 : f64 = rng.gen();
let r = r2.sqrt();
```

We need a set of orthogonal axes in the plane of reflection. We pick a vector to start with, and compute the outer product with the normal to give one vector \\(\vec{u}\\) orthogonal to \\(\vec{n}\\). To prevent numberloss, the first vector should not be too close to the normal. The second vector \\(\vec{v}\\) is found by taking another outer product of \\(\vec{u} \times \vec{n}\\).

```rust #diffuse-reflection
let ncl = if n_refl.x.abs() > 0.1 { vec(0., 1., 0.) } else { vec(1., 0., 0.) };
let u = (ncl % n_refl).normalize();
let v = n_refl % u;
```

The direction of the reflected ray is now known.

```rust #diffuse-reflection
let d = (u*phi.cos()*r + v*phi.sin()*r + n_refl*(1.-r2).sqrt()).normalize();
```

To compute the radiance, we need to know the radiance from the reflected ray.

```rust #diffuse-reflection
*ray = Ray {origin: x, direction: d};
colour = f * colour;
```

### Specular reflection
Specular reflection means we have to mirror the incident ray with respect to the normal. This means that only the \\(\vec{n}\\) component of the direction flips,

\\[\vec{\hat{d}}' = \vec{\hat{d}} - 2 \vec{\hat{n}} (\vec{\hat{n}} \cdot \vec{\hat{d}})\\].

```rust #specular-reflection
let d = ray.direction - n * 2.*(n*ray.direction);
*ray = Ray {origin: x, direction: d};
colour = f * colour;
```

### Refraction
Now some real optics! Discarding polarisation, there are several ways a photon may go at the boundary between two transparent media: *total internal reflection*, *refraction*, or *partial reflection*.

There is always a reflective component,

```rust #refractive-reflection
let d = ray.direction - n * 2.*(n*ray.direction);
let reflected_ray = Ray { origin: x, direction: d };
```

We need to know if we're moving into or out of the object.

```rust #refractive-reflection
let into = n * n_refl > 0.;
```

### Refractive index
The refractive index of glass can vary, but \\(n = 1.5\\) seems reasonable.

```rust #constants
const N_GLASS: f64 = 1.5;
const N_AIR: f64 = 1.0;
```

Depending on whether we're entering or leaving the glass object, the effective index of refraction is
\\(n_{\rm air} / n_{\rm glass}\\) or \\(n_{\rm glass} / n_{\rm air}\\).

```rust #refractive-reflection
let n_eff = if into { N_AIR / N_GLASS } else { N_GLASS / N_AIR };
```

### Total internal reflection
Total internal reflection happens if the angle of incidence is larger than some critical angle \\(\theta_c\\), given by

\\[\theta_c = \arcsin \frac{n_{\rm outside}}{n_{\rm inside}}.\\]{#eq:tir-critical-angle}

We can easily compute \\(\mu = \cos \theta\\) using the inner product, so with a little algebra, total-internal-reflection happens if,

\\[\begin{align}
\sin \theta &> {n_o \over n_i}\\
\sqrt{1 - \cos^2 \theta} &> {n_o \over n_i}\\
1 - \mu^2 &> \left({n_o \over n_i}\right)^2\\
n_{\rm eff}^2 \left(1 - \mu^2\right) &> 1
\end{align}\\]

```rust #refractive-reflection
let mu = ray.direction * n_refl;
let cos2t = 1. - n_eff*n_eff*(1. - mu*mu);
if cos2t < 0. {
    <<total-internal-reflection>>
} else {
    <<partial-reflection>>
}
```

In that case, we recurse with the reflected ray.

```rust #total-internal-reflection
*ray = reflected_ray;
colour = f * colour;
```

### Partial reflection
In the case of partial reflection, we need to compute also the angle of the refracted ray. We have Snell's law,

\\[{\sin \theta_i \over \sin \theta_o} = {n_o \over n_i} = {1 \over n_{\rm eff}}.\\]{#eq:snellius}

We can decompose the incident ray direction into a normal component \\(d_n\\) and a transverse component \\(d_t\\). Then \\(|d_t| = \sin \theta_i\\), and \\(|d_n| = \vec{d} \cdot \vec{n} = \cos \theta_i\\). Similarly we can decompose the outgoing ray direction \\(\vec{d}'\\),

\\[\begin{align}
d_t' &= n_{\rm eff} (\vec{d} - \mu \vec{n})\\
d_n' &= \sqrt{1 - n_{\rm eff}^2 |d_t|^2} \vec{n},
\end{align}\\]

where \\(|d_t|^2 = 1 - \mu^2\\). That is convenient, because it turns out we have already computed \\(|d_n'|\\), it is the square root of `cos2t`. Now, we also see where the total internal reflection comes from; there is no solution to Snell's law for those angles.

```rust #partial-reflection
let tdir = (ray.direction * n_eff - n_refl * (mu*n_eff + cos2t.sqrt())).normalize();
```

Next, we need to compute the fraction of light that is reflected. The Fresnel equations describe this process, but they are very complicated and also deal with polarisation. We use Schlick's approximation instead {{#cite Schlick1994}},

\\[R(\theta) = R_0 + (1 - R_0) (1 - \mu)^5,\\]

where

\\[R_0 = \left(\frac{n_i - n_o}{n_i + n_o}\right)^2.\\]

```rust #constants
const R0: f64 =  (N_GLASS - N_AIR) * (N_GLASS - N_AIR)
              / ((N_GLASS + N_AIR) * (N_GLASS + N_AIR));
```

```rust #partial-reflection
let c = 1. - (if into { -mu } else {tdir * n});
let re = R0 + (1. - R0) * c.powf(5.0);
let tr = 1. - re;
```

### Russian Roulette 2

```rust #partial-reflection
let p = 0.25 + 0.5*re;
let rp = re/p;
let tp = tr/(1.-p);
if depth > 2 {
    if rng.gen::<f64>() < p {
        *ray = reflected_ray;
        colour = f * colour * rp;
    } else {
        *ray = Ray { origin: x, direction: tdir };
        colour = f * colour * tp;
    }
} else {
    let r = radiance(&mut Ray {origin: x, direction: tdir}, depth);
    output = output + r * f * colour * tr;
    *ray = reflected_ray;
    colour = f * colour * re;
}
```
