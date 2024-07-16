// ~/~ begin <<lit/vectors.md#src/vec3.rs>>[init]
// ~/~ begin <<lit/vectors.md#vector>>[init]
#[derive(Clone,Copy,Debug)]
pub(crate) struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64
}

pub(crate) const fn vec(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3 { x: x, y: y, z: z }
}
// ~/~ end
// ~/~ begin <<lit/vectors.md#vector>>[1]
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
// ~/~ end
// ~/~ begin <<lit/vectors.md#vector>>[2]
impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self { x: self.x * s
             , y: self.y * s
             , z: self.z * s }
    }
}
// ~/~ end
// ~/~ begin <<lit/vectors.md#vector>>[3]
impl std::ops::Mul<Vec3> for Vec3 {
    type Output = f64;
    fn mul(self, other: Self) -> f64 {
        self.x * other.x +
        self.y * other.y +
        self.z * other.z
    }
}
// ~/~ end
// ~/~ begin <<lit/vectors.md#vector>>[4]
impl std::ops::Rem for Vec3 {
    type Output = Self;
    fn rem(self, other: Self) -> Self {
        Self { x: self.y * other.z - self.z * other.y
             , y: self.z * other.x - self.x * other.z
             , z: self.x * other.y - self.y * other.x }
    }
}
// ~/~ end
// ~/~ begin <<lit/vectors.md#vector>>[5]
impl Vec3 {
    pub fn abs(self) -> f64 {
        (self * self).sqrt()
    }

    pub fn normalize(self) -> Self {
        self * (1.0 / self.abs())
    }
}
// ~/~ end

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::*;

    // ~/~ begin <<lit/vectors.md#vector-tests>>[init]
    impl Arbitrary for Vec3 {
        fn arbitrary(g: &mut Gen) -> Self {
            let x = f64::arbitrary(g);
            let y = f64::arbitrary(g);
            let z = f64::arbitrary(g);
            vec(x, y, z)
        }
    }

    impl Vec3 {
        fn is_finite(&self) -> bool {
            self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
        }

        fn reasonable(&self) -> bool {
            self.is_finite() &&
                self.x.log2().abs() < 16.0 &&
                self.y.log2().abs() < 16.0 &&
                self.z.log2().abs() < 16.0
        }
    }
    // ~/~ end
    // ~/~ begin <<lit/vectors.md#vector-tests>>[1]
    #[quickcheck]
    fn outer_product_orthogonal(a: Vec3, b: Vec3) -> TestResult {
        if !(a.reasonable() && b.reasonable()) { return TestResult::discard(); }
        let c = a % b;
        TestResult::from_bool((a * c).abs() < 1e-6 && (b * c).abs() < 1e-6)
    }
    // ~/~ end
    // ~/~ begin <<lit/vectors.md#vector-tests>>[2]
    #[quickcheck]
    fn normalized_vec_length(a: Vec3) -> TestResult {
        if !a.reasonable() || (a * a) <= 0.0 { return TestResult::discard(); }
        let b = a.normalize();
        TestResult::from_bool((1.0 - b * b).abs() < 1e-6)
    }
    // ~/~ end
    // ~/~ begin <<lit/vectors.md#vector-tests>>[3]
    #[quickcheck]
    fn outer_product_anti_symmetry(a: Vec3, b: Vec3) -> TestResult {
        if !(a.reasonable() && b.reasonable()) { return TestResult::discard(); }
        let c = a % b;
        let d = b % a;
        TestResult::from_bool((c + d).abs() < 1e-6)
    }
    // ~/~ end
}
// ~/~ end