# LiteratePT
One of my favourite compute science books is ["Physically Based Rendering" by Matt Pharr, Wenzel Jakob and Greg Humphreys (PBRT)](https://pbrt.org/). For me, this book put the concept of Literate Programming on the map, as well as giving an awesome overview of the technologies that go into graphics rendering. Now PBRT is more than 1200 pages, so I thought what better than to create a tribute of some smaller size?

Over time, there have been developed some ray-tracers of truly miniscule size. It is amazing how much you can do in little code. For me the most clear example is [SmallPT](https://www.kevinbeason.com/smallpt/) by Kevin Beason. SmallPT is a global illumination ray tracer in 100 lines of C++.

![10000 spp rendering](img/image.png)

This is a translation into Rust; not in a 100 lines, but like PBRT, extremely literate. The entirety of this implementation is contained in a single Markdown file. To extract the source code, you may use [Entangled](https://entangled.github.io/), or to render the published version, use [Pandoc](https://pandoc.org/). All the math and equations are explained, and I've tried to explain some concepts in Rust.

## TODO

- [ ] Explain the sub-pixel sampling
- [ ] Explain use of Rayon in `Image::for_each`
- [x] Explain `RGBColour` structure
- [x] Add command-line interface
- [x] Fix performance issues with writing output
- [x] Add proper progress bar
- [x] Simplify recursion pattern

## Getting started with Rust
The easiest way to install the Rust compiler is through the [`rustup` command](https://rustup.rs/). This will install both the Rust compiler `rustc` and its accompanying package manager `cargo`. You'd normally start a new project by running `cargo init`. This command creates the skeleton structure of a Rust project: a `Cargo.toml` and a "Hello World" program in `src/main.rs`. Since my goal is to have everything in a single Markdown file, I include the `Cargo.toml` here:

```toml file=Cargo.toml
[package]
name = "literatept"
version = "0.2.0"
authors = ["Johan Hidding <j.hidding@esciencecenter.nl>"]
edition = "2018"

[profile.release]
opt-level = 3
debug = 0
strip = "debuginfo"

[dependencies]
<<dependencies>>

[dev-dependencies]
<<dev-dependencies>>
```

Now, if I want to introduce some features to this program that require external packages (called *crates* in Rust), I can do so by extending on the `<<dependencies>>` section. For example, I will need a random number generator. This is most commonly available in the `rand` crate:
<!-- > The package description can be extended using [more keys and their definitions](https://doc.rust-lang.org/cargo/reference/manifest.html) -->

```toml #dependencies
rand = "0.8.5"
```

## Outline
Everything about graphics rendering happens in a three-dimensional world, so I will need to explain some of the [vector mathematics](#vectors) that we're using. In SmallPT, the `vec3` type is then doubling up as a type for [colours](#colours). Since we're not trying to be minimal here, I will treat colours entirely separately from vector algebra. After this ground work, we need to implement some [geometry primitives](#geometry): spheres, rays and how they intersect. When we have that, we can describe a [scene](#scene). The SmallPT scene is a modification of the Cornell box, that consists solely of spheres, some so large that they appear as a nearly flat surface.

This all leads up to the core of the matter: [path tracing](#path-tracing). How do we model every possible path that a beam of light can take to arrive in our camera?

The program is not complete before we write a [main function](#main), including code to write the image to a PPM file, and some user interaction: command-line arguments and a friendly progress bar.

```toml #dependencies
rayon = "1.5.3"
indicatif = { version = "0.16.2", features = ["rayon"] }
argh = "0.1.7"
```

```rust #imports
extern crate rayon;

use rayon::prelude::*;
```
