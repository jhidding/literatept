\\[\renewcommand{\vec}[1]{{\bf #1}}\\]

# Geometry
With floating-point calculations, round-off can become a problem. If we bounce a ray off a sphere, how do we make sure that we don't detect another intersection with the same sphere? One way is to make sure that every ray travels a mininum distance before bouncing off anything. We'll call this distance `EPS`, short for *epsilon*, being the greek letter \\(\epsilon\\), generally denoting small quantities.

```rust #constants
const EPS: f64 = 1e-4;
```

## Objects
The only objects in our scene are spheres. When we do path tracing, we also need rays.

```rust #ray
struct Ray
    { pub origin: Vec3
    , pub direction: Vec3 }
```

```rust #sphere
struct Sphere
    { pub radius: f64
    , pub position: Vec3
    <<sphere-members>>
    }
```

## Intersections
The `Shpere` has a method to detect intersection with a `Ray`.

```rust #sphere
impl Sphere {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        <<sphere-ray-intersect>>
    }
}
```

The equation for the surface of a sphere at position \\(\vec{p}\\) and radius \\(r\\) is,

\\[S:\ (\vec{p} - \vec{x})^2 = r^2,\\]

and a ray from origin $\vec{o}$ and direction \\(\vec{\hat{d}}\\) describes the half-line,

\\[L:\ \vec{x} = \vec{o} + t\vec{\hat{d}}.\\]

Equating these gives a quadratic equation for \\(t\\), taking \\(\vec{q} = \vec{p} - \vec{o}\\),

$$\begin{align}
S \cap L:\ &(\vec{p} - \vec{o} - t\vec{\hat{d}})^2 = r^2\\\\
           &t^2 - 2t\vec{\hat{d}}\vec{q} + \vec{q}^2 - r^2 = 0\\\\
           &t = \vec{\hat{d}}\vec{q} \pm \sqrt{(\vec{\hat{d}}\vec{q})^2 - \vec{q}^2 + r^2}.
\end{align}$$

We first compute the determinant (part under the square root),

```rust #sphere-ray-intersect
let q = self.position - ray.origin;
let b = ray.direction * q;
let r = self.radius;
let det = b*b - q*q + r*r;
```

If it is negative, there is no solution and the ray does not intersect with the sphere.

```rust #sphere-ray-intersect
if det < 0. {
    return None;
}
```

Otherwise, it is safe to compute the square-root and return the first intersection at a distance larger than `EPS`.

```rust #sphere-ray-intersect
let rdet = det.sqrt();
if b - rdet > EPS {
    Some(b - rdet)
} else if b + rdet > EPS {
    Some(b + rdet)
} else {
    None
}
```

## Properties
A sphere has material properties. We can choose between *diffuse*, *specular* and *refractive* type.

```rust #material
enum Reflection
    { Diffuse
    , Specular
    , Refractive }
```

```admonish info title="Sum types"
Note that the Rust `enum` types are much richer than the `enum` you may be used to from C/C++. Together with `struct`, `enum` gives the corner stones of *algebraic data types*. Where a `struct` collects different members into a *product type*, an `enum` is a *sum type*, meaning that it either contains one value or the other.
```

```rust #sphere-members
, pub emission: RGBColour
, pub colour: RGBColour
, pub reflection: Reflection
```
