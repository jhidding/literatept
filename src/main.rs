// ~\~ language=Rust filename=src/main.rs
// ~\~ begin <<lit/index.md|src/main.rs>>[init]
// ~\~ begin <<lit/index.md|import-quickcheck>>[init]
#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;
// ~\~ end
// ~\~ begin <<lit/index.md|import-rand>>[init]
extern crate rand;
use rand::Rng;
// ~\~ end
// ~\~ begin <<lit/index.md|imports>>[init]
extern crate rayon;

use rayon::prelude::*;
// ~\~ end
extern crate indicatif;
extern crate argh;
use argh::FromArgs;

mod vec3;
use vec3::*;

mod colour;
use colour::*;

// ~\~ begin <<lit/index.md|constants>>[init]
const EPS: f64 = 1e-4;
const SAMPLES: usize = 100;
const WIDTH: usize = 640;
const HEIGHT: usize = 480;
// ~\~ end
// ~\~ begin <<lit/index.md|constants>>[1]
use std::f64::consts::PI;
// ~\~ end
// ~\~ begin <<lit/index.md|constants>>[2]
const N_GLASS: f64 = 1.5;
const N_AIR: f64 = 1.0;
// ~\~ end
// ~\~ begin <<lit/index.md|constants>>[3]
const R0: f64 =  (N_GLASS - N_AIR) * (N_GLASS - N_AIR)
              / ((N_GLASS + N_AIR) * (N_GLASS + N_AIR));
// ~\~ end
// ~\~ begin <<lit/index.md|ray>>[init]
struct Ray
    { pub origin: Vec3
    , pub direction: Vec3 }
// ~\~ end
// ~\~ begin <<lit/index.md|material>>[init]
enum Reflection
    { Diffuse
    , Specular
    , Refractive }
// ~\~ end
// ~\~ begin <<lit/index.md|sphere>>[init]
struct Sphere
    { pub radius: f64
    , pub position: Vec3
    // ~\~ begin <<lit/index.md|sphere-members>>[init]
    , pub emission: RGBColour
    , pub colour: RGBColour
    , pub reflection: Reflection
    // ~\~ end
    }
// ~\~ end
// ~\~ begin <<lit/index.md|sphere>>[1]
impl Sphere {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        // ~\~ begin <<lit/index.md|sphere-ray-intersect>>[init]
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
// ~\~ begin <<lit/index.md|scene>>[init]
const SPHERES: [Sphere;9] =
    // ~\~ begin <<lit/index.md|scene-spheres>>[init]
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
// ~\~ begin <<lit/index.md|path-tracing>>[init]
fn radiance(ray: &mut Ray, mut depth: u16) -> RGBColour {
    // let mut current = SomeRadianceCall { ray: ray, depth: depth, colour: WHITE });
    // let mut stack = Vec::new();
    let mut rng = rand::thread_rng();
    let mut colour = WHITE;
    let mut output = BLACK;

    // while let Some(RadianceCall { ref mut ray, ref mut depth, ref mut colour }) = current
    loop {
        // ~\~ begin <<lit/index.md|do-intersect>>[init]
        let hit = intersect(&ray);
        if hit.is_none() { return BLACK; }
        let (distance, object) = hit.unwrap();
        output = output + object.emission * colour;
        // ~\~ end
        // ~\~ begin <<lit/index.md|russian-roulette-1>>[init]
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
        // ~\~ end
        // ~\~ begin <<lit/index.md|compute-normal>>[init]
        let x = ray.origin + ray.direction * distance;
        let n = (x - object.position).normalize();
        // ~\~ end
        // ~\~ begin <<lit/index.md|compute-normal>>[1]
        let n_refl = if n * ray.direction < 0. { n } else { -n };
        // ~\~ end
        // ~\~ begin <<lit/index.md|do-reflect>>[init]
        match object.reflection {
            Reflection::Diffuse => {
                // ~\~ begin <<lit/index.md|diffuse-reflection>>[init]
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
                *ray = Ray {origin: x, direction: d};
                colour = f * colour;
                // push(&mut stack, Ray {origin: x, direction: d}, depth, f * colour);
                // ~\~ end
            }
            Reflection::Specular => {
                // ~\~ begin <<lit/index.md|specular-reflection>>[init]
                let d = ray.direction - n * 2.*(n*ray.direction);
                *ray = Ray {origin: x, direction: d};
                colour = f * colour;
                // push(&mut stack, Ray {origin: x, direction: d}, depth, f * colour);
                // ~\~ end
            }
            Reflection::Refractive => {
                // ~\~ begin <<lit/index.md|refractive-reflection>>[init]
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
                    // ~\~ begin <<lit/index.md|total-internal-reflection>>[init]
                    *ray = reflected_ray;
                    colour = f * colour;
                    // push(&mut stack, reflected_ray, depth, f * colour);
                    // ~\~ end
                } else {
                    // ~\~ begin <<lit/index.md|partial-reflection>>[init]
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
                            *ray = reflected_ray;
                            colour = f * colour * rp;
                        } else {
                            *ray = reflected_ray;
                            colour = f * colour * tp;
                        }
                    } else {
                        let r = radiance(&mut Ray {origin: x, direction: tdir}, depth);
                        output = output + r * f * colour * tr;
                        *ray = reflected_ray;
                        colour = f * colour * re;
                    }
                    // ~\~ end
                }
                // ~\~ end
            }
        };
        // ~\~ end
    }
}
// ~\~ end
// ~\~ begin <<lit/index.md|image>>[init]
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

    fn for_each<F>(&mut self, f: F)
        where F: Send + Sync + Fn(usize, usize, &mut RGBColour)
    {
        use indicatif::ParallelProgressIterator;
        // use rayon::iter::{ParallelIterator, IntoParallelRefIterator};

        let w = self.width;
        let size = self.size() as u64;
        self.data
            .par_iter_mut().progress_count(size)
            .enumerate()
            .for_each(|(i, c)| {
                let x = i % w;
                let y = i / w;
                f(x, y, c);
            });
    }

    fn size(&self) -> usize { self.width * self.height }

    // ~\~ begin <<lit/index.md|print-ppm>>[init]
    fn print_ppm(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let file = File::create(path)?;
        let mut out = std::io::BufWriter::new(file);
        write!(&mut out, "P3\n{} {}\n{}\n", self.width, self.height, 255)?;

        for i in 0..self.size() {
            let (r, g, b) = self.data[i].to_u24();
            write!(&mut out, "{} {} {} ", r, g, b)?;
        }
        Ok(())
    }
    // ~\~ end
}
// ~\~ end

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::*;

    // ~\~ begin <<lit/index.md|vector-tests>>[init]
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

#[derive(FromArgs)]
/// Renders the Cornell box as interpreted by Kevin Beason's SmallPt
pub struct Arghs {
    /// optional sample size (100)
    #[argh(option, short = 's', default = "SAMPLES")]
    samples: usize,

    /// optional thread count
    /// the default (0) will take the systems logical cpu count
    #[argh(option, short = 't', default = "0")]
    threads: usize,

    /// optional stack size in MB per thread
    #[argh(option, short = 'z', default = "8")]
    stack: usize,

    /// optional image size dimensions WxH (640x480)
    #[argh(option, from_str_fn(into_plot_dimensions), default = "(WIDTH, HEIGHT)")]
    wxh: (usize, usize),
}

// Helper function for parsing plot dimensions from command line arguments.
fn into_plot_dimensions(dim: &str) -> Result<(usize, usize), String> {
    let (w, h) = dim
        .split_once('x')
        .ok_or("dimensions do not parse, no delimiter?")?;
    let w = w.parse::<usize>().map_err(|e| e.to_string())?;
    let h = h.parse::<usize>().map_err(|e| e.to_string())?;
    Ok((w, h))
}

fn main() -> std::io::Result<()> {
    // use rayon::current_thread_index;
    // rayon::ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();
    let args: Arghs = argh::from_env();
    let (w, h) = args.wxh;
    let samps = args.samples;

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .stack_size(args.stack * 1024 * 1024)
        .build_global()
        .unwrap();

    let cam = Ray { origin: vec(50., 52., 295.6), direction: vec(0.0, -0.045, -1.0).normalize() };
    let cx = vec(w as f64 * 0.510 / h as f64, 0., 0.);
    let cy = (cx % cam.direction).normalize() * 0.510;

    let mut img = Image::new(w, h);
    eprintln!("Rendering ({} spp)", samps*4);

    img.for_each(|x, y, c| {
        let mut rng = rand::thread_rng();
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
                    r = r + radiance(&mut Ray {origin: cam.origin + d*140., direction: d.normalize()}, 0) * (1./samps as f64);
                }
                *c = *c + r.clamp() * 0.25;
            }
        }
    });

    eprintln!("\nWriting image.");
    img.print_ppm("image_rust.ppm")
}
// ~\~ end
