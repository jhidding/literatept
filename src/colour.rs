pub(crate) const BLACK: RGBColour = rgb(0.0, 0.0, 0.0);

// ~\~ begin <<lit/index.md|colour>>[0]
pub(crate) trait Colour:
    Sized
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Mul<f64, Output = Self>
{
    fn to_rgb(&self) -> (f64, f64, f64);
    fn clamp(&self) -> Self;

    fn max(&self) -> f64 {
        let (r, g, b) = self.to_rgb();
        if r > g && r > b {
            r
        } else if g > b {
            g
        } else {
            b
        }
    }

    fn to_u24(&self) -> (u8, u8, u8) {
        let to_int = |x: f64| (x.clamp(0., 1.).powf(1. / 2.2) * 255. + 0.5).floor() as u8;
        let (r, g, b) = self.to_rgb();
        (to_int(r), to_int(g), to_int(b))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default)]
pub(crate) struct RGBColour(f64, f64, f64);

pub(crate) const fn rgb(r: f64, g: f64, b: f64) -> RGBColour {
    RGBColour(r, g, b)
}

impl std::ops::Add for RGBColour {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let RGBColour(r1, g1, b1) = self;
        let RGBColour(r2, g2, b2) = other;
        RGBColour(r1 + r2, g1 + g2, b1 + b2)
    }
}

impl std::ops::Mul for RGBColour {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let RGBColour(r1, g1, b1) = self;
        let RGBColour(r2, g2, b2) = other;
        RGBColour(r1 * r2, g1 * g2, b1 * b2)
    }
}

impl std::ops::Mul<f64> for RGBColour {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        let RGBColour(r1, g1, b1) = self;
        RGBColour(r1 * s, g1 * s, b1 * s)
    }
}

impl Colour for RGBColour {
    fn to_rgb(&self) -> (f64, f64, f64) {
        let RGBColour(r, g, b) = self;
        (*r, *g, *b)
    }

    fn clamp(&self) -> Self {
        let RGBColour(r, g, b) = self;
        RGBColour(r.clamp(0., 1.), g.clamp(0., 1.), b.clamp(0., 1.))
    }
}
// ~\~ end
