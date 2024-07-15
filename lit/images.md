# Images

```rust #image
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

    <<print-ppm>>
}
```

## Writing to PPM
To write output efficiently, we need a `BufWriter` instance.

```rust #print-ppm
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
```
