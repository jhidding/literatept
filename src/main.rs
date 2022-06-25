// ~\~ language=Rust filename=src/main.rs
// ~\~ begin <<lit/index.md|src/main.rs>>[0]
// ~\~ begin <<lit/index.md|import-quickcheck>>[0]
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;
// ~\~ end
// ~\~ begin <<lit/index.md|import-rand>>[0]
// use rand::Rng;
use nanorand::Rng;
// ~\~ end
// ~\~ begin <<lit/index.md|imports>>[0]
use rayon::prelude::*;
// ~\~ end
use argh::FromArgs;
mod colour;
mod vec3;
use colour::*;
use vec3::*;
// ~\~ begin <<lit/index.md|constants>>[0]
const EPS: f64 = 1e-4;
use std::{f64::consts::PI, io::Write};
// ~\~ end
// ~\~ begin <<lit/index.md|constants>>[2]
const N_GLASS: f64 = 1.5;
const N_AIR: f64 = 1.0;
// ~\~ end
// ~\~ begin <<lit/index.md|constants>>[3]
const R0: f64 = (N_GLASS - N_AIR) * (N_GLASS - N_AIR) / ((N_GLASS + N_AIR) * (N_GLASS + N_AIR));
const WIDTH: usize = 640;
const HEIGHT: usize = 480;
const SAMPLES: usize = 500;
// ~\~ end

// ~\~ begin <<lit/index.md|ray>>[0]
struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}
// ~\~ end
// ~\~ begin <<lit/index.md|material>>[0]
enum Reflection {
    Diffuse,
    Specular,
    Refractive,
}
// ~\~ end
// ~\~ begin <<lit/index.md|sphere>>[0]
struct Sphere {
    pub radius: f64,
    pub position: Vec3, // ~\~ begin <<lit/index.md|sphere-members>>[0]
    pub emission: RGBColour,
    pub colour: RGBColour,
    pub reflection: Reflection, // ~\~ end
}
// ~\~ end
// ~\~ begin <<lit/index.md|sphere>>[1]
impl Sphere {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        // ~\~ begin <<lit/index.md|sphere-ray-intersect>>[0]
        let q = self.position - ray.origin;
        let b = ray.direction * q;
        let r = self.radius;
        let det = b * b - q * q + r * r;
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
const SPHERES: [Sphere; 9] =
    // ~\~ begin <<lit/index.md|scene-spheres>>[0]
    [
        Sphere {
            radius: 1e5,
            position: vec(1e5 + 1., 40.8, 81.6),
            emission: BLACK,
            colour: rgb(0.75, 0.25, 0.25),
            reflection: Reflection::Diffuse,
        }, // ~\~ end
        // ~\~ begin <<lit/index.md|scene-spheres>>[1]
        Sphere {
            radius: 1e5,
            position: vec(50., 40.8, 1e5),
            emission: BLACK,
            colour: rgb(0.75, 0.75, 0.75),
            reflection: Reflection::Diffuse,
        },
        Sphere {
            radius: 1e5,
            position: vec(50., 40.8, -1e5 + 170.),
            emission: BLACK,
            colour: BLACK,
            reflection: Reflection::Diffuse,
        },
        Sphere {
            radius: 1e5,
            position: vec(50., 1e5, 81.6),
            emission: BLACK,
            colour: rgb(0.75, 0.75, 0.75),
            reflection: Reflection::Diffuse,
        },
        Sphere {
            radius: 1e5,
            position: vec(50., -1e5 + 81.6, 81.6),
            emission: BLACK,
            colour: rgb(0.75, 0.75, 0.75),
            reflection: Reflection::Diffuse,
        }, // ~\~ end
        // ~\~ begin <<lit/index.md|scene-spheres>>[2]
        Sphere {
            radius: 1e5,
            position: vec(-1e5 + 99., 40.8, 81.6),
            emission: BLACK,
            colour: rgb(0.25, 0.25, 0.75),
            reflection: Reflection::Diffuse,
        }, // ~\~ end
        // ~\~ begin <<lit/index.md|scene-spheres>>[3]
        Sphere {
            radius: 16.5,
            position: vec(27., 16.5, 47.),
            emission: BLACK,
            colour: rgb(0.999, 0.999, 0.999),
            reflection: Reflection::Specular,
        },
        Sphere {
            radius: 16.5,
            position: vec(73., 16.5, 78.),
            emission: BLACK,
            colour: rgb(0.999, 0.999, 0.999),
            reflection: Reflection::Refractive,
        }, // ~\~ end
        // ~\~ begin <<lit/index.md|scene-spheres>>[4]
        Sphere {
            radius: 600.,
            position: vec(50., 681.6 - 0.27, 81.6),
            emission: rgb(12.0, 12.0, 12.0),
            colour: BLACK,
            reflection: Reflection::Diffuse,
        },
    ];
// ~\~ end
// ~\~ end
// ~\~ begin <<lit/index.md|scene>>[1]
fn intersect(ray: &Ray) -> Option<(f64, &'static Sphere)> {
    SPHERES
        .iter()
        .fold(None, |result: Option<(f64, &Sphere)>, s| {
            match s.intersect(ray) {
                Some(d) if result.is_none() => Some((d, s)),
                Some(d) if result.unwrap().0 > d => Some((d, s)),
                _ => result,
            }
        })
}
// ~\~ end
// ~\~ begin <<lit/index.md|path-tracing>>[0]
fn radiance(rng: &mut nanorand::tls::TlsWyRand, ray: &Ray, mut depth: u16) -> RGBColour {
    // ~\~ begin <<lit/index.md|do-intersect>>[0]
    let (distance, object) = match intersect(ray) {
        Some(hit) => hit,
        None => return BLACK,
    };
    // //let mut rng = rand::thread_rng();
    // let mut rng = nanorand::tls_rng(); // In effort to mitigate stack loading
    // ~\~ end
    // ~\~ begin <<lit/index.md|russian-roulette-1>>[0]
    let mut f = object.colour;
    let p = f.max();
    depth += 1;
    if depth > 5 {
        if rng.generate::<f64>() < p {
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
            let phi = 2. * PI * rng.generate::<f64>();
            // ~\~ end
            // ~\~ begin <<lit/index.md|diffuse-reflection>>[1]
            let r2: f64 = rng.generate();
            let r = r2.sqrt();
            // ~\~ end
            // ~\~ begin <<lit/index.md|diffuse-reflection>>[2]
            let ncl = if n_refl.x.abs() > 0.1 {
                vec(0., 1., 0.)
            } else {
                vec(1., 0., 0.)
            };
            let u = (ncl % n_refl).normalize();
            let v = n_refl % u;
            // ~\~ end
            // ~\~ begin <<lit/index.md|diffuse-reflection>>[3]
            let d = (u * phi.cos() * r + v * phi.sin() * r + n_refl * (1. - r2).sqrt()).normalize();
            // ~\~ end
            // ~\~ begin <<lit/index.md|diffuse-reflection>>[4]
            radiance(
                rng,
                &Ray {
                    origin: x,
                    direction: d,
                },
                depth,
            )
            // ~\~ end
        }
        Reflection::Specular => {
            // ~\~ begin <<lit/index.md|specular-reflection>>[0]
            let d = ray.direction - n * 2. * (n * ray.direction);
            radiance(
                rng,
                &Ray {
                    origin: x,
                    direction: d,
                },
                depth,
            )
            // ~\~ end
        }
        Reflection::Refractive => {
            // ~\~ begin <<lit/index.md|refractive-reflection>>[0]
            let d = ray.direction - n * 2. * (n * ray.direction);
            let reflected_ray = Ray {
                origin: x,
                direction: d,
            };
            // ~\~ end
            // ~\~ begin <<lit/index.md|refractive-reflection>>[1]
            let into = n * n_refl > 0.;
            // ~\~ end
            // ~\~ begin <<lit/index.md|refractive-reflection>>[2]
            let n_eff = if into {
                N_AIR / N_GLASS
            } else {
                N_GLASS / N_AIR
            };
            // ~\~ end
            // ~\~ begin <<lit/index.md|refractive-reflection>>[3]
            let mu = ray.direction * n_refl;
            let cos2t = 1. - n_eff * n_eff * (1. - mu * mu);
            if cos2t < 0. {
                // ~\~ begin <<lit/index.md|total-internal-reflection>>[0]
                radiance(rng, &reflected_ray, depth)
                // ~\~ end
            } else {
                // ~\~ begin <<lit/index.md|partial-reflection>>[0]
                let tdir =
                    (ray.direction * n_eff - n_refl * (mu * n_eff + cos2t.sqrt())).normalize();
                // ~\~ end
                // ~\~ begin <<lit/index.md|partial-reflection>>[1]
                let c = 1. - (if into { -mu } else { tdir * n });
                let re = R0 + (1. - R0) * c.powf(5.0);
                let tr = 1. - re;
                // ~\~ end
                // ~\~ begin <<lit/index.md|partial-reflection>>[2]
                let p = re.mul_add(0.5, 0.25);
                let rp = re / p;
                let tp = tr / (1. - p);
                if depth > 2 {
                    if rng.generate::<f64>() < p {
                        radiance(rng, &reflected_ray, depth) * rp
                    } else {
                        radiance(
                            rng,
                            &Ray {
                                origin: x,
                                direction: tdir,
                            },
                            depth,
                        ) * tp
                    }
                } else {
                    radiance(rng, &reflected_ray, depth) * re
                        + radiance(
                            rng,
                            &Ray {
                                origin: x,
                                direction: tdir,
                            },
                            depth,
                        ) * tr
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
struct Image {
    width: usize,
    height: usize,
    data: Vec<RGBColour>,
}

impl Default for Image {
    fn default() -> Self {
        let data: Vec<RGBColour> = vec![RGBColour::default(); WIDTH * HEIGHT];
        Image {
            width: WIDTH,
            height: HEIGHT,
            data,
        }
    }
}

impl Image {
    fn new(width: usize, height: usize) -> Image {
        let data: Vec<RGBColour> = vec![RGBColour::default(); width * height];
        Image {
            width,
            height,
            data,
        }
    }

    fn for_each<F>(&mut self, f: F)
    where
        F: Send + Sync + Fn(&mut nanorand::tls::TlsWyRand, usize, usize, &mut RGBColour),
    {
        let w = self.width;
        self.data
            .par_iter_mut()
            .enumerate()
            .for_each_init(nanorand::tls_rng, |rng, (i, c)| {
                let x = i % w;
                let y = i / w;
                f(rng, x, y, c);
            });
    }

    // ~\~ begin <<lit/index.md|print-ppm>>[0]
    fn print_ppm(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;

        let file = File::create(path)?;
        let mut out = std::io::BufWriter::new(file);
        write!(&mut out, "P3\n{} {}\n{}\n", self.width, self.height, 255)?;

        for rgb in &self.data {
            let (r, g, b) = rgb.to_u24();
            write!(&mut out, "{r} {g} {b} ")?;
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

    // ~\~ begin <<lit/index.md|vector-tests>>[0]
    impl Arbitrary for Vec3 {
        fn arbitrary(g: &mut Gen) -> Self {
            let x = f64::arbitrary(g);
            let y = f64::arbitrary(g);
            let z = f64::arbitrary(g);
            vec(x, y, z)
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
            empty_shrinker()
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
        if (a * a) == 0.0 {
            return true;
        }
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
/// Simple tool to sample and plot power consumption, average frequency and cpu die temperatures over time.
pub struct Arghs {
    /// optional sample size
    #[argh(option, short = 's', default = "SAMPLES")]
    samples: usize,

    /// optional thread count
    /// the default (0) will take the systems logical cpu count
    #[argh(option, short = 't', default = "0")]
    threads: usize,

    /// optional stack size in MB per thread
    #[argh(option, short = 'z', default = "8")]
    stack: usize,

    /// optional image size dimensions WxH (1024x768)
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
    let args: Arghs = argh::from_env();
    use rayon::current_thread_index;

    let (w, h) = args.wxh;
    let samps: usize = args.samples;
    let cam = Ray {
        origin: vec(50., 52., 295.6),
        direction: vec(0.0, -0.045, -1.0).normalize(),
    };
    let cx = vec(w as f64 * 0.510 / h as f64, 0., 0.);
    let cy = (cx % cam.direction).normalize() * 0.510;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .stack_size(args.stack * 1024 * 1024)
        .build()
        .unwrap();

    let mut img = Image::new(w, h);

    pool.install(|| {
        img.for_each(|rng, x, y, c| {
            if current_thread_index() == Some(0) {
                print!(
                    "\rRendering ({} spp) {:5.2}%",
                    samps * 4,
                    100. * (y as f64 / (h - 1) as f64)
                );
            }
            for sy in 0..2 {
                for sx in 0..2 {
                    let mut r = BLACK;
                    for _ in 0..samps {
                        let r1 = 2. * rng.generate::<f64>();
                        let dx = if r1 < 1. {
                            r1.sqrt() - 1.
                        } else {
                            1. - (2. - r1).sqrt()
                        };
                        let r2 = 2. * rng.generate::<f64>();
                        let dy = if r2 < 1. {
                            r2.sqrt() - 1.
                        } else {
                            1. - (2. - r2).sqrt()
                        };
                        let d = cx * (((sx as f64 + 0.5 + dx) / 2. + x as f64) / w as f64 - 0.5)
                            + cy * (((sy as f64 + 0.5 + dy) / 2. + (h - y - 1) as f64) / h as f64
                                - 0.5)
                            + cam.direction;
                        r = r + radiance(
                            rng,
                            &Ray {
                                origin: cam.origin + d * 140.,
                                direction: d.normalize(),
                            },
                            0,
                        ) * (1. / samps as f64);
                    }
                    *c = *c + r.clamp() * 0.25;
                }
            }
        });
    });

    println!("\nWriting image.");
    img.print_ppm("image_rust.ppm")
}
// ~\~ end
