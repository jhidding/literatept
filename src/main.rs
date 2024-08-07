// ~/~ begin <<lit/main-app.md#src/main.rs>>[init]
// ~/~ begin <<lit/vectors.md#import-quickcheck>>[init]
#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;
// ~/~ end
// ~/~ begin <<lit/index.md#imports>>[init]
extern crate rayon;

use rayon::prelude::*;
// ~/~ end
// ~/~ begin <<lit/path-tracing.md#imports>>[0]
extern crate rand;
use rand::Rng;
// ~/~ end
// ~/~ begin <<lit/main-app.md#imports>>[0]
extern crate argh;
use argh::FromArgs;
// ~/~ end
extern crate indicatif;
mod vec3;
use vec3::*;

mod colour;
use colour::*;

// ~/~ begin <<lit/geometry.md#constants>>[init]
const EPS: f64 = 1e-4;
// ~/~ end
// ~/~ begin <<lit/path-tracing.md#constants>>[0]
use std::f64::consts::PI;
// ~/~ end
// ~/~ begin <<lit/path-tracing.md#constants>>[1]
const N_GLASS: f64 = 1.5;
const N_AIR: f64 = 1.0;
// ~/~ end
// ~/~ begin <<lit/path-tracing.md#constants>>[2]
const R0: f64 =  (N_GLASS - N_AIR) * (N_GLASS - N_AIR)
              / ((N_GLASS + N_AIR) * (N_GLASS + N_AIR));
// ~/~ end
// ~/~ begin <<lit/main-app.md#constants>>[0]
const SAMPLES: usize = 100;
const WIDTH: usize = 640;
const HEIGHT: usize = 480;
// ~/~ end
// ~/~ begin <<lit/geometry.md#ray>>[init]
struct Ray
    { pub origin: Vec3
    , pub direction: Vec3 }
// ~/~ end
// ~/~ begin <<lit/geometry.md#material>>[init]
enum Reflection
    { Diffuse
    , Specular
    , Refractive }
// ~/~ end
// ~/~ begin <<lit/geometry.md#sphere>>[init]
struct Sphere
    { pub radius: f64
    , pub position: Vec3
    // ~/~ begin <<lit/geometry.md#sphere-members>>[init]
    , pub emission: RGBColour
    , pub colour: RGBColour
    , pub reflection: Reflection
    // ~/~ end
    }
// ~/~ end
// ~/~ begin <<lit/geometry.md#sphere>>[1]
impl Sphere {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        // ~/~ begin <<lit/geometry.md#sphere-ray-intersect>>[init]
        let q = self.position - ray.origin;
        let b = ray.direction * q;
        let r = self.radius;
        let det = b*b - q*q + r*r;
        // ~/~ end
        // ~/~ begin <<lit/geometry.md#sphere-ray-intersect>>[1]
        if det < 0. {
            return None;
        }
        // ~/~ end
        // ~/~ begin <<lit/geometry.md#sphere-ray-intersect>>[2]
        let rdet = det.sqrt();
        if b - rdet > EPS {
            Some(b - rdet)
        } else if b + rdet > EPS {
            Some(b + rdet)
        } else {
            None
        }
        // ~/~ end
    }
}
// ~/~ end
// ~/~ begin <<lit/scene.md#scene>>[init]
const SPHERES: [Sphere;9] =
    // ~/~ begin <<lit/scene.md#scene-spheres>>[init]
    [ Sphere { radius:  1e5,  position: vec(1e5+1.,   40.8, 81.6), emission: BLACK
             , colour: rgb(0.75, 0.25, 0.25), reflection: Reflection::Diffuse }
    // ~/~ end
    // ~/~ begin <<lit/scene.md#scene-spheres>>[1]
    , Sphere { radius:  1e5,  position: vec(50., 40.8, 1e5),       emission: BLACK
             , colour: rgb(0.75, 0.75, 0.75), reflection: Reflection::Diffuse }
    , Sphere { radius:  1e5,  position: vec(50., 40.8, -1e5+170.),  emission: BLACK
             , colour: BLACK,                 reflection: Reflection::Diffuse }
    , Sphere { radius:  1e5,  position: vec(50., 1e5, 81.6),       emission: BLACK
             , colour: rgb(0.75, 0.75, 0.75), reflection: Reflection::Diffuse }
    , Sphere { radius:  1e5,  position: vec(50., -1e5+81.6, 81.6), emission: BLACK
             , colour: rgb(0.75, 0.75, 0.75), reflection: Reflection::Diffuse }
    // ~/~ end
    // ~/~ begin <<lit/scene.md#scene-spheres>>[2]
    , Sphere { radius:  1e5,  position: vec(-1e5+99., 40.8, 81.6), emission: BLACK
             , colour: rgb(0.25, 0.25, 0.75), reflection: Reflection::Diffuse }
    // ~/~ end
    // ~/~ begin <<lit/scene.md#scene-spheres>>[3]
    , Sphere { radius: 16.5,  position: vec(27., 16.5, 47.), emission: BLACK
             , colour: rgb(0.999, 0.999, 0.999), reflection: Reflection::Specular }
    , Sphere { radius: 16.5,  position: vec(73., 16.5, 78.), emission: BLACK
             , colour: rgb(0.999, 0.999, 0.999), reflection: Reflection::Refractive }
    // ~/~ end
    // ~/~ begin <<lit/scene.md#scene-spheres>>[4]
    , Sphere { radius:  600.,  position: vec(50., 681.6-0.27, 81.6)
             , emission: rgb(12.0, 12.0, 12.0), colour: BLACK
             , reflection: Reflection::Diffuse } ];
    // ~/~ end
// ~/~ end
// ~/~ begin <<lit/scene.md#scene>>[1]
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
// ~/~ end
// ~/~ begin <<lit/path-tracing.md#path-tracing>>[init]
fn radiance(ray: &mut Ray, mut depth: u16) -> RGBColour {
    let mut rng = rand::thread_rng();
    let mut colour = WHITE;
    let mut output = BLACK;

    loop {
        // ~/~ begin <<lit/path-tracing.md#do-intersect>>[init]
        let hit = intersect(&ray);
        if hit.is_none() { return output; }
        let (distance, object) = hit.unwrap();
        output = output + object.emission * colour;
        // ~/~ end
        // ~/~ begin <<lit/path-tracing.md#russian-roulette-1>>[init]
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
        // ~/~ end
        // ~/~ begin <<lit/path-tracing.md#compute-normal>>[init]
        let x = ray.origin + ray.direction * distance;
        let n = (x - object.position).normalize();
        // ~/~ end
        // ~/~ begin <<lit/path-tracing.md#compute-normal>>[1]
        let n_refl = if n * ray.direction < 0. { n } else { -n };
        // ~/~ end
        // ~/~ begin <<lit/path-tracing.md#do-reflect>>[init]
        match object.reflection {
            Reflection::Diffuse => {
                // ~/~ begin <<lit/path-tracing.md#diffuse-reflection>>[init]
                let phi = 2.*PI * rng.gen::<f64>();
                // ~/~ end
                // ~/~ begin <<lit/path-tracing.md#diffuse-reflection>>[1]
                let r2 : f64 = rng.gen();
                let r = r2.sqrt();
                // ~/~ end
                // ~/~ begin <<lit/path-tracing.md#diffuse-reflection>>[2]
                let ncl = if n_refl.x.abs() > 0.1 { vec(0., 1., 0.) } else { vec(1., 0., 0.) };
                let u = (ncl % n_refl).normalize();
                let v = n_refl % u;
                // ~/~ end
                // ~/~ begin <<lit/path-tracing.md#diffuse-reflection>>[3]
                let d = (u*phi.cos()*r + v*phi.sin()*r + n_refl*(1.-r2).sqrt()).normalize();
                // ~/~ end
                // ~/~ begin <<lit/path-tracing.md#diffuse-reflection>>[4]
                *ray = Ray {origin: x, direction: d};
                colour = f * colour;
                // ~/~ end
            }
            Reflection::Specular => {
                // ~/~ begin <<lit/path-tracing.md#specular-reflection>>[init]
                let d = ray.direction - n * 2.*(n*ray.direction);
                *ray = Ray {origin: x, direction: d};
                colour = f * colour;
                // ~/~ end
            }
            Reflection::Refractive => {
                // ~/~ begin <<lit/path-tracing.md#refractive-reflection>>[init]
                let d = ray.direction - n * 2.*(n*ray.direction);
                let reflected_ray = Ray { origin: x, direction: d };
                // ~/~ end
                // ~/~ begin <<lit/path-tracing.md#refractive-reflection>>[1]
                let into = n * n_refl > 0.;
                // ~/~ end
                // ~/~ begin <<lit/path-tracing.md#refractive-reflection>>[2]
                let n_eff = if into { N_AIR / N_GLASS } else { N_GLASS / N_AIR };
                // ~/~ end
                // ~/~ begin <<lit/path-tracing.md#refractive-reflection>>[3]
                let mu = ray.direction * n_refl;
                let cos2t = 1. - n_eff*n_eff*(1. - mu*mu);
                if cos2t < 0. {
                    // ~/~ begin <<lit/path-tracing.md#total-internal-reflection>>[init]
                    *ray = reflected_ray;
                    colour = f * colour;
                    // ~/~ end
                } else {
                    // ~/~ begin <<lit/path-tracing.md#partial-reflection>>[init]
                    let tdir = (ray.direction * n_eff - n_refl * (mu*n_eff + cos2t.sqrt())).normalize();
                    // ~/~ end
                    // ~/~ begin <<lit/path-tracing.md#partial-reflection>>[1]
                    let c = 1. - (if into { -mu } else {tdir * n});
                    let re = R0 + (1. - R0) * c.powf(5.0);
                    let tr = 1. - re;
                    // ~/~ end
                    // ~/~ begin <<lit/path-tracing.md#partial-reflection>>[2]
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
                    // ~/~ end
                }
                // ~/~ end
            }
        };
        // ~/~ end
    }
}
// ~/~ end
// ~/~ begin <<lit/images.md#image>>[init]
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

    // ~/~ begin <<lit/images.md#print-ppm>>[init]
    fn print_ppm(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let file = File::create(path)?;
        let mut out = std::io::BufWriter::new(file);
        write!(&mut out, "P3\n{} {}\n{}\n", self.width, self.height, 255)?;

        for rgb in &self.data {
            let (r, g, b) = rgb.to_u24();
            write!(&mut out, "{} {} {} ", r, g, b)?;
        }
        Ok(())
    }
    // ~/~ end
}
// ~/~ end
// ~/~ begin <<lit/main-app.md#arghs>>[init]
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
// ~/~ end


fn main() -> std::io::Result<()> {
    // use rayon::current_thread_index;
    // rayon::ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();
    let args: Arghs = argh::from_env();
    let (w, h) = args.wxh;
    let samps = args.samples / 4;

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
// ~/~ end