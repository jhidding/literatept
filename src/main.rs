// ~\~ language=Rust filename=src/main.rs
// ~\~ begin <<lit/index.md|src/main.rs>>[0]
// ~\~ begin <<lit/index.md|import-quickcheck>>[0]
#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;
// ~\~ end
// ~\~ begin <<lit/index.md|import-rand>>[0]
extern crate rand;
use rand::Rng;
// ~\~ end
// ~\~ begin <<lit/index.md|imports>>[0]
extern crate rayon;

use rayon::prelude::*;
use std::f64::consts::PI;
// ~\~ end

// ~\~ begin <<lit/index.md|constants>>[0]
const EPS: f64 = 1e-4;
// ~\~ end
// ~\~ begin <<lit/index.md|constants>>[1]
const N_GLASS: f64 = 1.5;
const N_AIR: f64 = 1.0;
// ~\~ end
// ~\~ begin <<lit/index.md|constants>>[2]
const R0: f64 =  (N_GLASS - N_AIR) * (N_GLASS - N_AIR)
              / ((N_GLASS + N_AIR) * (N_GLASS + N_AIR));
// ~\~ end
// ~\~ begin <<lit/index.md|vector>>[0]
#[derive(Clone,Copy,Debug)]
struct Vec3
    { pub x: f64, pub y: f64, pub z: f64 }

const fn vec(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3 { x: x, y: y, z: z }
}
// ~\~ end
// ~\~ begin <<lit/index.md|vector>>[1]
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
// ~\~ end
// ~\~ begin <<lit/index.md|vector>>[2]
impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self { x: self.x * s
             , y: self.y * s
             , z: self.z * s }
    }
}
// ~\~ end
// ~\~ begin <<lit/index.md|vector>>[3]
impl std::ops::Mul<Vec3> for Vec3 {
    type Output = f64;
    fn mul(self, other: Self) -> f64 {
        self.x * other.x +
        self.y * other.y +
        self.z * other.z
    }
}
// ~\~ end
// ~\~ begin <<lit/index.md|vector>>[4]
impl std::ops::Rem for Vec3 {
    type Output = Self;
    fn rem(self, other: Self) -> Self {
        Self { x: self.y * other.z - self.z * other.y
             , y: self.z * other.x - self.x * other.z
             , z: self.x * other.y - self.y * other.x }
    }
}
// ~\~ end
// ~\~ begin <<lit/index.md|vector>>[5]
impl Vec3 {
    fn abs(self) -> f64 {
        (self * self).sqrt()
    }

    fn normalize(self) -> Self {
        self * (1.0 / self.abs())
    }
}
// ~\~ end
// ~\~ begin <<lit/index.md|colour>>[0]
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
// ~\~ end
// ~\~ begin <<lit/index.md|ray>>[0]
struct Ray
    { pub origin: Vec3
    , pub direction: Vec3 }
// ~\~ end
// ~\~ begin <<lit/index.md|material>>[0]
enum Reflection
    { Diffuse
    , Specular
    , Refractive }
// ~\~ end
// ~\~ begin <<lit/index.md|sphere>>[0]
struct Sphere
    { pub radius: f64
    , pub position: Vec3
    // ~\~ begin <<lit/index.md|sphere-members>>[0]
    , pub emission: RGBColour
    , pub colour: RGBColour
    , pub reflection: Reflection
    // ~\~ end
    }
// ~\~ end
// ~\~ begin <<lit/index.md|sphere>>[1]
impl Sphere {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        // ~\~ begin <<lit/index.md|sphere-ray-intersect>>[0]
        let q = self.position - ray.origin;
        let b = ray.direction * q;
        let r = self.radius;
        let det = b*b - q*q + r*r;
        // ~\~ end
        // ~\~ begin <<lit/index.md|sphere-ray-intersect>>[1]
        if det < 0. {
            return None;
        }
        // ~\~ end
        // ~\~ begin <<lit/index.md|sphere-ray-intersect>>[2]
        let rdet = det.sqrt();
        if b - rdet > EPS {
            Some(b - rdet)
        } else if b + rdet > EPS {
            Some(b + rdet)
        } else {
            None
        }
        // ~\~ end
    }
}
// ~\~ end
// ~\~ begin <<lit/index.md|scene>>[0]
const SPHERES: [Sphere;9] =
    // ~\~ begin <<lit/index.md|scene-spheres>>[0]
    [ Sphere { radius:  1e5,  position: vec(1e5+1.,   40.8, 81.6), emission: BLACK
             , colour: rgb(0.75, 0.25, 0.25), reflection: Reflection::Diffuse }
    // ~\~ end
    // ~\~ begin <<lit/index.md|scene-spheres>>[1]
    , Sphere { radius:  1e5,  position: vec(50., 40.8, 1e5),       emission: BLACK
             , colour: rgb(0.75, 0.75, 0.75), reflection: Reflection::Diffuse }
    , Sphere { radius:  1e5,  position: vec(50., 40.8, -1e5+170.),  emission: BLACK
             , colour: BLACK,                 reflection: Reflection::Diffuse }
    , Sphere { radius:  1e5,  position: vec(50., 1e5, 81.6),       emission: BLACK
             , colour: rgb(0.75, 0.75, 0.75), reflection: Reflection::Diffuse }
    , Sphere { radius:  1e5,  position: vec(50., -1e5+81.6, 81.6), emission: BLACK
             , colour: rgb(0.75, 0.75, 0.75), reflection: Reflection::Diffuse }
    // ~\~ end
    // ~\~ begin <<lit/index.md|scene-spheres>>[2]
    , Sphere { radius:  1e5,  position: vec(-1e5+99., 40.8, 81.6), emission: BLACK
             , colour: rgb(0.25, 0.25, 0.75), reflection: Reflection::Diffuse }
    // ~\~ end
    // ~\~ begin <<lit/index.md|scene-spheres>>[3]
    , Sphere { radius: 16.5,  position: vec(27., 16.5, 47.), emission: BLACK
             , colour: rgb(0.999, 0.999, 0.999), reflection: Reflection::Specular }
    , Sphere { radius: 16.5,  position: vec(73., 16.5, 78.), emission: BLACK
             , colour: rgb(0.999, 0.999, 0.999), reflection: Reflection::Refractive }
    // ~\~ end
    // ~\~ begin <<lit/index.md|scene-spheres>>[4]
    , Sphere { radius:  600.,  position: vec(50., 681.6-0.27, 81.6)
             , emission: rgb(12.0, 12.0, 12.0), colour: BLACK
             , reflection: Reflection::Diffuse } ];
    // ~\~ end
// ~\~ end
// ~\~ begin <<lit/index.md|scene>>[1]
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
// ~\~ end
// ~\~ begin <<lit/index.md|path-tracing>>[0]
fn radiance(ray: &Ray, mut depth: u16) -> RGBColour {
    let mut rng = rand::thread_rng();
    // ~\~ begin <<lit/index.md|do-intersect>>[0]
    let hit = intersect(ray);
    if hit.is_none() { return BLACK; }
    let (distance, object) = hit.unwrap();
    // ~\~ end
    // ~\~ begin <<lit/index.md|russian-roulette-1>>[0]
    let mut f = object.colour;
    let p = f.max();
    depth += 1;
    if depth > 5 {
        if rng.gen::<f64>() < p {
            f = f * (1. / p);
        } else {
            return object.emission;
        }
    }
    // ~\~ end
    // ~\~ begin <<lit/index.md|compute-normal>>[0]
    let x = ray.origin + ray.direction * distance;
    let n = (x - object.position).normalize();
    // ~\~ end
    // ~\~ begin <<lit/index.md|compute-normal>>[1]
    let n_refl = if n * ray.direction < 0. { n } else { -n };
    // ~\~ end
    // ~\~ begin <<lit/index.md|do-reflect>>[0]
    let light = match object.reflection {
        Reflection::Diffuse => {
            // ~\~ begin <<lit/index.md|diffuse-reflection>>[0]
            let phi = 2.*PI * rng.gen::<f64>();
            // ~\~ end
            // ~\~ begin <<lit/index.md|diffuse-reflection>>[1]
            let r2 : f64 = rng.gen();
            let r = r2.sqrt();
            // ~\~ end
            // ~\~ begin <<lit/index.md|diffuse-reflection>>[2]
            let ncl = if n_refl.x.abs() > 0.1 { vec(0., 1., 0.) } else { vec(1., 0., 0.) };
            let u = (ncl % n_refl).normalize();
            let v = n_refl % u;
            // ~\~ end
            // ~\~ begin <<lit/index.md|diffuse-reflection>>[3]
            let d = (u*phi.cos()*r + v*phi.sin()*r + n_refl*(1.-r2).sqrt()).normalize();
            // ~\~ end
            // ~\~ begin <<lit/index.md|diffuse-reflection>>[4]
            radiance(&Ray {origin: x, direction: d}, depth)
            // ~\~ end
        }
        Reflection::Specular => {
            // ~\~ begin <<lit/index.md|specular-reflection>>[0]
            let d = ray.direction - n * 2.*(n*ray.direction);
            radiance(&Ray {origin: x, direction: d}, depth)
            // ~\~ end
        }
        Reflection::Refractive => {
            // ~\~ begin <<lit/index.md|refractive-reflection>>[0]
            let d = ray.direction - n * 2.*(n*ray.direction);
            let reflected_ray = Ray { origin: x, direction: d };
            // ~\~ end
            // ~\~ begin <<lit/index.md|refractive-reflection>>[1]
            let into = n * n_refl > 0.;
            // ~\~ end
            // ~\~ begin <<lit/index.md|refractive-reflection>>[2]
            let n_eff = if into { N_AIR / N_GLASS } else { N_GLASS / N_AIR };
            // ~\~ end
            // ~\~ begin <<lit/index.md|refractive-reflection>>[3]
            let mu = ray.direction * n_refl;
            let cos2t = 1. - n_eff*n_eff*(1. - mu*mu);
            if cos2t < 0. {
                // ~\~ begin <<lit/index.md|total-internal-reflection>>[0]
                radiance(&reflected_ray, depth)
                // ~\~ end
            } else {
                // ~\~ begin <<lit/index.md|partial-reflection>>[0]
                let tdir = (ray.direction * n_eff - n_refl * (mu*n_eff + cos2t.sqrt())).normalize();
                // ~\~ end
                // ~\~ begin <<lit/index.md|partial-reflection>>[1]
                let c = 1. - (if into { -mu } else {tdir * n});
                let re = R0 + (1. - R0) * c.powf(5.0);
                let tr = 1. - re;
                // ~\~ end
                // ~\~ begin <<lit/index.md|partial-reflection>>[2]
                let p = 0.25 + 0.5*re;
                let rp = re/p;
                let tp = tr/(1.-p);
                if depth > 2 {
                    if rng.gen::<f64>() < p {
                        radiance(&reflected_ray, depth) * rp
                    } else {
                        radiance(&Ray { origin: x, direction: tdir }, depth) * tp
                    }
                } else {
                    radiance(&reflected_ray, depth) * re
                    + radiance(&Ray { origin: x, direction: tdir }, depth) * tr
                }
                // ~\~ end
            }
            // ~\~ end
        }
    };
    // ~\~ end
    object.emission + f * light
}
// ~\~ end
// ~\~ begin <<lit/index.md|image>>[0]
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
// ~\~ end

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::*;

    // ~\~ begin <<lit/index.md|vector-tests>>[0]
    impl Arbitrary for Vec3 {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            let x = f64::arbitrary(g);
            let y = f64::arbitrary(g);
            let z = f64::arbitrary(g);
            vec(x, y, z)
        }
    }
    // ~\~ end
    // ~\~ begin <<lit/index.md|vector-tests>>[1]
    #[quickcheck]
    fn outer_product_orthogonal(a: Vec3, b: Vec3) -> bool {
        let c = a % b;
        (a * c).abs() < 1e-6 && (b * c).abs() < 1e-6
    }
    // ~\~ end
    // ~\~ begin <<lit/index.md|vector-tests>>[2]
    #[quickcheck]
    fn normalized_vec_length(a: Vec3) -> bool {
        if (a * a) == 0.0 { return true; }
        let b = a.normalize();
        (1.0 - b * b).abs() < 1e-6
    }
    // ~\~ end
    // ~\~ begin <<lit/index.md|vector-tests>>[3]
    #[quickcheck]
    fn outer_product_anti_symmetry(a: Vec3, b: Vec3) -> bool {
        let c = a % b;
        let d = b % a;
        (c + d).abs() < 1e-6
    }
    // ~\~ end
}

fn main() -> std::io::Result<()> {
    use rayon::current_thread_index;

    let w: usize = 640;
    let h: usize = 480;
    let samps: usize = 100;
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
// ~\~ end
