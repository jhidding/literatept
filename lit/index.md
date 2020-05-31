---
title: LiteratePT
subtitle: a translation of smallpt
---

# LiteratePT

SmallPT is a global illumination ray tracer in 100 lines of C++. Let's translate it to Rust; not in a 100 lines, but extremely literate. I'll sacrifice some of SmallPt's compactness for better semantics.

This uses the following programming techniques:

- Property testing
* Traits
- Algebraic data types

``` {.toml file=Cargo.toml}
[package]
name = "literatept"
version = "0.1.0"
authors = ["Johan Hidding <j.hidding@esciencecenter.nl>"]
edition = "2018"

[profile.release]
opt-level = 3

[dependencies]
<<dependencies>>

[dev-dependencies]
<<dev-dependencies>>
```

- [ ] The package description can be extended using [more keys and their definitions](https://doc.rust-lang.org/cargo/reference/manifest.html)

``` {.toml #dependencies}
rand = "0.7.3"
rayon = "1.3.0"
```

# Vectors
$\renewcommand{\vec}[1]{{\bf #1}}$
The `Vec` type has three public members $x$, $y$ and $z$.

``` {.rust #vector}
#[derive(Clone,Copy,Debug)]
struct Vec3
    { pub x: f64, pub y: f64, pub z: f64 }

const fn vec(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3 { x: x, y: y, z: z }
}
```

We derive the `Clone`, `Copy`, and `Debug` traits, meaning that we can print debug statements involving `Vec` instances, and that we can clone instances usinge the `.clone()` method. The `Copy` trait means that the `Vec` can be copied implicitly, creating call-by-value semantics on this type.

### Operators
Each of the operators only occupy a single line of code in SmallPt, but this is probably better.

``` {.rust #vector}
impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x
             , y: self.y + other.y
             , z: self.z + other.z }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x
             , y: self.y - other.y
             , z: self.z - other.z }
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self { x: -self.x, y: -self.y, z: -self.z }
    }
}
```

SmallPt defines four kinds of multiplication: scaling, point-wise multiplication, dot product and outer product. The point-wise multiplication is only used to manipulate colors, for which we'll use separate structures.

Here's scaling,

``` {.rust #vector}
impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self { x: self.x * s
             , y: self.y * s
             , z: self.z * s }
    }
}
```

the dot-product,

$$\vec{a} \cdot \vec{b} = a_x b_x + a_y b_y + a_z b_z$$

``` {.rust #vector}
impl std::ops::Mul<Vec3> for Vec3 {
    type Output = f64;
    fn mul(self, other: Self) -> f64 {
        self.x * other.x +
        self.y * other.y +
        self.z * other.z
    }
}
```

and outer product for which we abuse the `%` operator,

$$\vec{a} \wedge \vec{b} = \det \begin{pmatrix}
\hat{x} & \hat{y} & \hat{z}\\
a_x & a_y & a_z \\
b_x & b_y & b_z
\end{pmatrix}$$

``` {.rust #vector}
impl std::ops::Rem for Vec3 {
    type Output = Self;
    fn rem(self, other: Self) -> Self {
        Self { x: self.y * other.z - self.z * other.y
             , y: self.z * other.x - self.x * other.z
             , z: self.x * other.y - self.y * other.x }
    }
}
```

Vectors can be normalized to a unit-vector.

``` {.rust #vector}
impl Vec3 {
    fn abs(self) -> f64 {
        (self * self).sqrt()
    }

    fn normalize(self) -> Self {
        self * (1.0 / self.abs())
    }
}
```

## Tests
We use the `quickcheck` crate to do some property testing on the `Vec` type.

``` {.toml #dev-dependencies}
quickcheck = "0.9"
quickcheck_macros = "0.9"
```

``` {.rust #import-quickcheck}
#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;
```

We need to be able to generate `Arbitrary` instances of `Vec`. I'm not sure if this will ever yield a zero-vector, or a sequence of vectors that lie in the same plane.

``` {.rust #vector-tests}
impl Arbitrary for Vec3 {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let x = f64::arbitrary(g);
        let y = f64::arbitrary(g);
        let z = f64::arbitrary(g);
        vec(x, y, z)
    }
}
```

Now we can check that for any vectors $\vec{a}$ and $\vec{b}$, we have,

$$(\vec{a} \wedge \vec{b}) \cdot \vec{a} = 0,$$

``` {.rust #vector-tests}
#[quickcheck]
fn outer_product_orthogonal(a: Vec3, b: Vec3) -> bool {
    let c = a % b;
    (a * c).abs() < 1e-6 && (b * c).abs() < 1e-6
}
```

that any normalized vector has length 1,

``` {.rust #vector-tests}
#[quickcheck]
fn normalized_vec_length(a: Vec3) -> bool {
    if (a * a) == 0.0 { return true; }
    let b = a.normalize();
    (1.0 - b * b).abs() < 1e-6
}
```

and that the outer product upholds anti-symmetry,

$$\vec{a} \wedge \vec{b} = - \vec{b} \wedge \vec{a}.$$

``` {.rust #vector-tests}
#[quickcheck]
fn outer_product_anti_symmetry(a: Vec3, b: Vec3) -> bool {
    let c = a % b;
    let d = b % a;
    (c + d).abs() < 1e-6
}
```

# Colour
A sphere has material properties. We can choose between *diffuse*, *specular* and *refractive* type.

``` {.rust #material}
enum Reflection
    { Diffuse
    , Specular
    , Refractive }
```

:::: {.alert .alert-info}
Note that the Rust `enum` types are much richer than the `enum` you may be used to from C/C++. Together with `struct`, `enum` gives the corner stones of *algebraic data types*. Where a `struct` collects different members into a *product type*, an `enum` is a *sum type*, meaning that it either contains one value or the other.
::::

``` {.rust #colour}
#[inline]
fn clamp(x: f64) -> f64 { if x < 0. { 0. } else if x > 1. { 1. } else { x } }

trait Colour: Sized
            + std::ops::Add<Output=Self>
            + std::ops::Mul<Output=Self>
            + std::ops::Mul<f64, Output=Self> {
    fn to_rgb(&self) -> (f64, f64, f64);
    fn clamp(&self) -> Self;

    fn max(&self) -> f64 {
        let (r, g, b) = self.to_rgb();
        if r > g && r > b { r }
        else if g > b { g }
        else { b }
    }

    fn to_u24(&self) -> (u8, u8, u8) {
        let to_int = |x| (clamp(x).powf(1./2.2) * 255. + 0.5).floor() as u8;
        let (r, g, b) = self.to_rgb();
        (to_int(r), to_int(g), to_int(b))
    }
}

#[derive(Clone,Copy,Debug)]
struct RGBColour (f64, f64, f64);

const fn rgb(r: f64, g: f64, b: f64) -> RGBColour {
    RGBColour (r, g, b)
}

const BLACK: RGBColour = rgb(0.0, 0.0, 0.0);

impl std::ops::Add for RGBColour {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let RGBColour(r1,g1,b1) = self;
        let RGBColour(r2,g2,b2) = other;
        RGBColour(r1+r2,g1+g2,b1+b2)
    }
}

impl std::ops::Mul for RGBColour {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let RGBColour(r1,g1,b1) = self;
        let RGBColour(r2,g2,b2) = other;
        RGBColour(r1*r2,g1*g2,b1*b2)
    }
}

impl std::ops::Mul<f64> for RGBColour {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        let RGBColour(r1,g1,b1) = self;
        RGBColour(r1*s,g1*s,b1*s)
    }
}

impl Colour for RGBColour {
    fn to_rgb(&self) -> (f64, f64, f64) {
        let RGBColour(r, g, b) = self;
        (*r, *g, *b)
    }

    fn clamp(&self) -> Self {
        let RGBColour(r, g, b) = self;
        RGBColour(clamp(*r), clamp(*g), clamp(*b))
    }
}
```

# Geometry
With floating-point calculations, round-off can become a problem. If we bounce a ray off a sphere, how do we make sure that we don't detect another intersection with the same sphere? One way is to make sure that every ray travels a mininum distance before bouncing off anything. We'll call this distance `EPS`, short for *epsilon*, being the greek letter $\epsilon$, generally denoting small quantities.

``` {.rust #constants}
const EPS: f64 = 1e-4;
```

We now define the `Ray` and `Sphere` types.

``` {.rust #ray}
struct Ray
    { pub origin: Vec3
    , pub direction: Vec3 }
```

``` {.rust #sphere}
struct Sphere
    { pub radius: f64
    , pub position: Vec3
    <<sphere-members>>
    }
```

The `Shpere` has a method to detect intersection with a `Ray`.

``` {.rust #sphere}
impl Sphere {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        <<sphere-ray-intersect>>
    }
}
```

The equation for the surface of a sphere at position $\vec{p}$ and radius $r$ is,

$$S:\ (\vec{p} - \vec{x})^2 = r^2,$${#eq:sphere}

and a ray from origin $\vec{o}$ and direction $\vec{\hat{d}}$ describes the half-line,

$$L:\ \vec{x} = \vec{o} + t\vec{\hat{d}}.$${#eq:ray}

Equating these gives a quadratic equation for $t$, taking $\vec{q} = \vec{p} - \vec{o}$,

$$\begin{align}
S \cap L:\ &(\vec{p} - \vec{o} - t\vec{\hat{d}})^2 = r^2\\
           &t^2 - 2t\vec{\hat{d}}\vec{q} + \vec{q}^2 - r^2 = 0\\
           &t = \vec{\hat{d}}\vec{q} \pm \sqrt{(\vec{\hat{d}}\vec{q})^2 - \vec{q}^2 + r^2}.
\end{align}$${#eq:sphere-ray-intersect}

We first compute the determinant (part under the square root),

``` {.rust #sphere-ray-intersect}
let q = self.position - ray.origin;
let b = ray.direction * q;
let r = self.radius;
let det = b*b - q*q + r*r;
```

If it is negative, there is no solution and the ray does not intersect with the sphere.

``` {.rust #sphere-ray-intersect}
if det < 0. {
    return None;
}
```

Otherwise, it is safe to compute the square-root and return the first intersection at a distance larger than `EPS`.

``` {.rust #sphere-ray-intersect}
let rdet = det.sqrt();
if b - rdet > EPS {
    Some(b - rdet)
} else if b + rdet > EPS {
    Some(b + rdet)
} else {
    None
}
```

# Scene
The scene in SmallPt is an adaptation of the Cornell box.

``` {.rust #sphere-members}
, pub emission: RGBColour
, pub colour: RGBColour
, pub reflection: Reflection
```

``` {.rust #scene}
const SPHERES: [Sphere;9] =
    <<scene-spheres>>
```

The scene consists of a red ceiling,

``` {.rust #scene-spheres}
[ Sphere { radius:  1e5,  position: vec(1e5+1.,   40.8, 81.6), emission: BLACK
         , colour: rgb(0.75, 0.25, 0.25), reflection: Reflection::Diffuse }
```

four grey walls, one of which is black to emulate photons escaping,

``` {.rust #scene-spheres}
, Sphere { radius:  1e5,  position: vec(50., 40.8, 1e5),       emission: BLACK
         , colour: rgb(0.75, 0.75, 0.75), reflection: Reflection::Diffuse }
, Sphere { radius:  1e5,  position: vec(50., 40.8, -1e5+170.),  emission: BLACK
         , colour: BLACK,                 reflection: Reflection::Diffuse }
, Sphere { radius:  1e5,  position: vec(50., 1e5, 81.6),       emission: BLACK
         , colour: rgb(0.75, 0.75, 0.75), reflection: Reflection::Diffuse }
, Sphere { radius:  1e5,  position: vec(50., -1e5+81.6, 81.6), emission: BLACK
         , colour: rgb(0.75, 0.75, 0.75), reflection: Reflection::Diffuse }
```

a blue floor,

``` {.rust #scene-spheres}
, Sphere { radius:  1e5,  position: vec(-1e5+99., 40.8, 81.6), emission: BLACK
         , colour: rgb(0.25, 0.25, 0.75), reflection: Reflection::Diffuse }
```

a glass and a metal sphere,

``` {.rust #scene-spheres}
, Sphere { radius: 16.5,  position: vec(27., 16.5, 47.), emission: BLACK
         , colour: rgb(0.999, 0.999, 0.999), reflection: Reflection::Specular }
, Sphere { radius: 16.5,  position: vec(73., 16.5, 78.), emission: BLACK
         , colour: rgb(0.999, 0.999, 0.999), reflection: Reflection::Refractive }
```

and a plafonniere

``` {.rust #scene-spheres}
, Sphere { radius:  600.,  position: vec(50., 681.6-0.27, 81.6)
         , emission: rgb(12.0, 12.0, 12.0), colour: BLACK
         , reflection: Reflection::Diffuse } ];
```

Given this scene, we can define the function `intersect` which computes the first intersection of a ray with any of the objects in the scene. If the ray intersects, a tuple is returned giving the distance and reference to the obstructing object.

``` {.rust #scene}
fn intersect(ray: &Ray) -> Option<(f64, &'static Sphere)> {
    let mut result : Option<(f64, &Sphere)> = None;
    for s in &SPHERES {
        if let Some(d) = s.intersect(ray) {
            if result.is_none() || result.unwrap().0 > d {
                result = Some((d, s));
            }
        }
    }
    result
}
```

It feel like we've done a lot of work here, but we've only arrived at line 48 of SmallPt.

# Path tracing

``` {.rust #import-rand}
extern crate rand;
use rand::Rng;
```

``` {.rust #imports}
extern crate rayon;

use rayon::prelude::*;
use std::f64::consts::PI;
```

``` {.rust #path-tracing}
fn radiance(ray: &Ray, mut depth: u16) -> RGBColour {
    let mut rng = rand::thread_rng();
    let hit = intersect(ray);
    if hit.is_none() { return BLACK; }
    let (d, obj) = hit.unwrap();
    let x = ray.origin + ray.direction * d;
    let n = (x - obj.position).normalize();
    let normal = if n * ray.direction < 0. { n } else { -n };
    let mut f = obj.colour;
    let p = f.max();
    depth += 1;
    if depth > 5 {
        if rng.gen::<f64>() < p {
            f = f * (1. / p);
        } else {
            return obj.emission;
        }
    }
    let light = match obj.reflection {
        Reflection::Diffuse => {
            let r1 : f64 = 2.*PI*rng.gen::<f64>();
            let r2 : f64 = rng.gen();
            let r2s = r2.sqrt();
            let ncl = if normal.x.abs() > 0.1 { vec(0., 1., 0.) } else { vec(1., 0., 0.) };
            let u = (ncl % normal).normalize();
            let v = normal % u;
            let d = (u*r1.cos()*r2s + v*r1.sin()*r2s + normal*(1.-r2).sqrt()).normalize();
            radiance(&Ray {origin: x, direction: d}, depth)
        }
        Reflection::Specular => {
            let d = ray.direction - n * 2.*(n*ray.direction);
            radiance(&Ray {origin: x, direction: d}, depth)
        }
        Reflection::Refractive => {
            let d = ray.direction - n * 2.*(n*ray.direction);
            let refl_ray = Ray { origin: x, direction: d };
            let into = n * normal > 0.;
            let nc = 1.;
            let nt = 1.5;
            let nnt = if into { nc / nt } else { nt / nc };
            let ddn = ray.direction * normal;
            let cos2t = 1. - nnt*nnt*(1. - ddn*ddn);
            if cos2t < 0. { // total internal reflection
                radiance(&refl_ray, depth)
            } else {
                let tdir = (ray.direction * nnt - n*(if into {1.} else {-1.}) * (ddn*nnt + cos2t.sqrt())).normalize();
                let a = nt - nc;
                let b = nt + nc;
                let r0 = a*a/(b*b);
                let c = 1. - (if into { -ddn } else {tdir * n});
                let re = r0 + (1.-r0) * c.powf(5.0);
                let tr = 1.-re;
                let p = 0.25 + 0.5*re;
                let rp = re/p;
                let tp = tr/(1.-p);
                if depth > 2 {
                    if rng.gen::<f64>() < p {
                        radiance(&refl_ray, depth) * rp
                    } else {
                        radiance(&Ray { origin: x, direction: tdir }, depth) * tp
                    }
                } else {
                    radiance(&refl_ray, depth) * re + radiance(&Ray { origin: x, direction: tdir }, depth) * tr
                }
            }
        }
    };
    obj.emission + f * light
}
```

# Image

``` {.rust #image}
struct Image
    { width: usize
    , height: usize
    , pub data: Vec<RGBColour> }

impl Image {
    fn new(width: usize, height: usize) -> Image {
        Image {
            width: width, height: height,
            data: vec![BLACK; width*height]
        }
    }

    fn for_each<F: std::marker::Sync +  Fn (usize, usize, &mut RGBColour)>(&mut self, f: F) {
        let w = self.width;
        self.data.par_iter_mut().enumerate().for_each(
            |(i, c)| {
                let x = i % w;
                let y = i / w;
                f(x, y, c);
            }
        );
    }

    fn size(&self) -> usize { self.width * self.height }

    fn print_ppm(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut out = File::create(path)?;
        write!(&mut out, "P3\n{} {}\n{}\n", self.width, self.height, 255)?;

        for i in 0..self.size() {
            let (r, g, b) = self.data[i].to_u24();
            write!(&mut out, "{} {} {} ", r, g, b)?;
        }
        Ok(())
    }
}
```

# Main

``` {.rust file=src/main.rs}
<<import-quickcheck>>
<<import-rand>>
<<imports>>

<<constants>>
<<vector>>
<<colour>>
<<ray>>
<<material>>
<<sphere>>
<<scene>>
<<path-tracing>>
<<image>>

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::*;

    <<vector-tests>>
}

fn main() -> std::io::Result<()> {
    use rayon::current_thread_index;

    let w: usize = 640;
    let h: usize = 480;
    let samps: usize = 1000;
    let cam = Ray { origin: vec(50., 52., 295.6), direction: vec(0.0, -0.045, -1.0).normalize() };
    let cx = vec(w as f64 * 0.510 / h as f64, 0., 0.);
    let cy = (cx % cam.direction).normalize() * 0.510;

    let mut img = Image::new(w, h);
    img.for_each(|x, y, c| {
        let mut rng = rand::thread_rng();
        if current_thread_index() == Some(0) {
            eprint!("\rRendering ({} spp) {:5.2}%", samps*4, 100.*(y as f64 / (h-1) as f64));
        }
        for sy in 0..2 {
            for sx in 0..2 {
                let mut r = BLACK.clone();
                for _ in 0..samps {
                    let r1 = 2. * rng.gen::<f64>();
                    let dx = if r1 < 1. { r1.sqrt() - 1. } else { 1. - (2. - r1).sqrt() };
                    let r2 = 2. * rng.gen::<f64>();
                    let dy = if r2 < 1. { r2.sqrt() - 1. } else { 1. - (2. - r2).sqrt() };
                    let d = cx*( ( (sx as f64 + 0.5 + dx) / 2. + x as f64) / w as f64 - 0.5 )
                          + cy*( ( (sy as f64 + 0.5 + dy) / 2. + (h - y - 1) as f64) / h as f64 - 0.5 )
                          + cam.direction;
                    r = r + radiance(&Ray {origin: cam.origin + d*140., direction: d.normalize()}, 0) * (1./samps as f64);
                }
                *c = *c + r.clamp() * 0.25;
            }
        }
    });

    eprintln!("\nWriting image.");
    img.print_ppm("image_rust.ppm")
}
```

