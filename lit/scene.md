# Scene
The scene in SmallPt is an adaptation of the Cornell box.

```rust #scene
const SPHERES: [Sphere;9] =
    <<scene-spheres>>
```

The scene consists of a red ceiling,

```rust #scene-spheres
[ Sphere { radius:  1e5,  position: vec(1e5+1.,   40.8, 81.6), emission: BLACK
         , colour: rgb(0.75, 0.25, 0.25), reflection: Reflection::Diffuse }
```

four grey walls, one of which is black to emulate photons escaping,

```rust #scene-spheres
, Sphere { radius:  1e5,  position: vec(50., 40.8, 1e5),       emission: BLACK
         , colour: rgb(0.75, 0.75, 0.75), reflection: Reflection::Diffuse }
, Sphere { radius:  1e5,  position: vec(50., 40.8, -1e5+170.),  emission: BLACK
         , colour: BLACK,                 reflection: Reflection::Diffuse }
, Sphere { radius:  1e5,  position: vec(50., 1e5, 81.6),       emission: BLACK
         , colour: rgb(0.75, 0.75, 0.75), reflection: Reflection::Diffuse }
, Sphere { radius:  1e5,  position: vec(50., -1e5+81.6, 81.6), emission: BLACK
         , colour: rgb(0.75, 0.75, 0.75), reflection: Reflection::Diffuse }
```

a blue floor,

```rust #scene-spheres
, Sphere { radius:  1e5,  position: vec(-1e5+99., 40.8, 81.6), emission: BLACK
         , colour: rgb(0.25, 0.25, 0.75), reflection: Reflection::Diffuse }
```

a glass and a metal sphere,

```rust #scene-spheres
, Sphere { radius: 16.5,  position: vec(27., 16.5, 47.), emission: BLACK
         , colour: rgb(0.999, 0.999, 0.999), reflection: Reflection::Specular }
, Sphere { radius: 16.5,  position: vec(73., 16.5, 78.), emission: BLACK
         , colour: rgb(0.999, 0.999, 0.999), reflection: Reflection::Refractive }
```

and a plafonniere

```rust #scene-spheres
, Sphere { radius:  600.,  position: vec(50., 681.6-0.27, 81.6)
         , emission: rgb(12.0, 12.0, 12.0), colour: BLACK
         , reflection: Reflection::Diffuse } ];
```

Given this scene, we can define the function `intersect` which computes the first intersection of a ray with any of the objects in the scene. If the ray intersects, a tuple is returned giving the distance and reference to the obstructing object.

```rust #scene
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
```

It feel like we've done a lot of work here, but we've only arrived at line 48 of SmallPt.
