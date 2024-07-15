# Main Application

## Argument parsing

```rust #imports
extern crate argh;
use argh::FromArgs;
```

```rust #constants
const SAMPLES: usize = 100;
const WIDTH: usize = 640;
const HEIGHT: usize = 480;
```

```rust #arghs
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
```

## The main file

```rust file=src/main.rs
<<import-quickcheck>>
<<imports>>
extern crate indicatif;
mod vec3;
use vec3::*;

mod colour;
use colour::*;

<<constants>>
<<ray>>
<<material>>
<<sphere>>
<<scene>>
<<path-tracing>>
<<image>>
<<arghs>>


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
```
