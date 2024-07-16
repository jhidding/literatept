\\[\renewcommand{\vec}[1]{{\bf #1}}\\]

# Vectors
The use of three-component vectors is ubiquitous in this little program.

```rust file=src/vec3.rs
<<vector>>

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::*;

    <<vector-tests>>
}
```

The `Vec3` type has three public members \\(x\\), \\(y\\) and \\(z\\). We define the `struct` and a short-hand helper function `vec`.

```rust #vector
#[derive(Clone,Copy,Debug)]
pub(crate) struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64
}

pub(crate) const fn vec(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3 { x: x, y: y, z: z }
}
```

We derive the `Clone`, `Copy`, and `Debug` traits, meaning that we can print debug statements involving `Vec3` instances, and that we can clone instances usinge the `.clone()` method. The `Copy` trait means that the `Vec3` can be copied implicitly, creating call-by-value semantics on this type.

```admonish info title="Why not a class?"
Rust doesn't have classes. Instead, you define a `struct` with the data elements, and then implement one or more `trait`s on top of that. Data hiding, access patterns, inheritance and what-have-you-not in object-oriented styles of programming can still be achieved using `trait`s. For more information, see [The Rust Book, chapter 17](https://doc.rust-lang.org/book/ch17-00-oop.html).
```

## Operators
Each of the overloaded operators only occupy a single line of code in SmallPt, but this is probably better. Rust has a trait for every standard operator in the language. These operators are syntactic sugar for the relevant function calls in each trait. Here we define `+`, and `-` (both unary and binary forms).

```rust #vector
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
```

SmallPt defines four kinds of multiplication: scaling, point-wise multiplication, dot product and outer product. The point-wise multiplication is only used to manipulate colors, for which we'll use separate structures.

Here's scaling,

```rust #vector
impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self { x: self.x * s
             , y: self.y * s
             , z: self.z * s }
    }
}
```

the dot-product,

\\[\vec{a} \cdot \vec{b} = a_x b_x + a_y b_y + a_z b_z\\]

```rust #vector
impl std::ops::Mul<Vec3> for Vec3 {
    type Output = f64;
    fn mul(self, other: Self) -> f64 {
        self.x * other.x +
        self.y * other.y +
        self.z * other.z
    }
}
```

and outer product for which we abuse the `%` operator,

\\[\vec{a} \wedge \vec{b} = \det \begin{pmatrix}
\hat{x} & \hat{y} & \hat{z}\\\\
a_x & a_y & a_z \\\\
b_x & b_y & b_z
\end{pmatrix}\\]

```rust #vector
impl std::ops::Rem for Vec3 {
    type Output = Self;
    fn rem(self, other: Self) -> Self {
        Self { x: self.y * other.z - self.z * other.y
             , y: self.z * other.x - self.x * other.z
             , z: self.x * other.y - self.y * other.x }
    }
}
```

Vectors can be normalized to a unit-vector.

```rust #vector
impl Vec3 {
    pub fn abs(self) -> f64 {
        (self * self).sqrt()
    }

    pub fn normalize(self) -> Self {
        self * (1.0 / self.abs())
    }
}
```

## Tests
We use the `quickcheck` crate to do some property testing on the `Vec3` type. The idea of property testing is that you define some properties (duh!) on a type that should always hold. Then, if you have a way to generate arbitrary elements of your type, you can see if these properties do indeed hold. In many cases where mathematics or physics is involved, these test are expressed in much cleaner code than the usual unit tests.

```toml #dev-dependencies
quickcheck = "1.0.3"
quickcheck_macros = "1.0.0"
```

```rust #import-quickcheck
#[cfg(test)]
extern crate quickcheck;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;
```

We need to be able to generate `Arbitrary` instances of `Vec`. I'm not sure if this will ever yield a zero-vector, or a sequence of vectors that lie in the same plane. We do want to check our properties on reasonable numbers though.

```rust #vector-tests
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
```

Now we can check that for any vectors \\(\vec{a}\\) and \\(\vec{b}\\), we have,

\\[(\vec{a} \wedge \vec{b}) \cdot \vec{a} = 0,\\]

```rust #vector-tests
#[quickcheck]
fn outer_product_orthogonal(a: Vec3, b: Vec3) -> TestResult {
    if !(a.reasonable() && b.reasonable()) { return TestResult::discard(); }
    let c = a % b;
    TestResult::from_bool((a * c).abs() < 1e-6 && (b * c).abs() < 1e-6)
}
```

that any normalized vector has length 1,

```rust #vector-tests
#[quickcheck]
fn normalized_vec_length(a: Vec3) -> TestResult {
    if !a.reasonable() || (a * a) <= 0.0 { return TestResult::discard(); }
    let b = a.normalize();
    TestResult::from_bool((1.0 - b * b).abs() < 1e-6)
}
```

and that the outer product upholds anti-symmetry,

\\[\vec{a} \wedge \vec{b} = - \vec{b} \wedge \vec{a}.\\]

```rust #vector-tests
#[quickcheck]
fn outer_product_anti_symmetry(a: Vec3, b: Vec3) -> TestResult {
    if !(a.reasonable() && b.reasonable()) { return TestResult::discard(); }
    let c = a % b;
    let d = b % a;
    TestResult::from_bool((c + d).abs() < 1e-6)
}
```
