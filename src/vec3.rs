// ~\~ language=Rust filename=src/vec3.rs
// ~\~ begin <<lit/index.md|vector>>[init]
#[derive(Clone,Copy,Debug)]
pub(crate) struct Vec3
    { pub x: f64, pub y: f64, pub z: f64 }

pub(crate) const fn vec(x: f64, y: f64, z: f64) -> Vec3 {
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
    pub fn abs(self) -> f64 {
        (self * self).sqrt()
    }

    pub fn normalize(self) -> Self {
        self * (1.0 / self.abs())
    }
}
// ~\~ end
