use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

const ASPECT_RATIO: f64 = 16.0 / 9.0;
const IMAGE_WIDTH: usize = 400;
const IMAGE_HEIGHT: usize = (IMAGE_WIDTH as f64 / ASPECT_RATIO) as usize;
const SAMPLES_PER_PIXEL: usize = 50;
const MAX_DEPTH: usize = 50;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng::with_seed(0x5EED1234CAFEBEEF);

    let world = random_scene(&mut rng);

    let look_from = Vec3::new(13.0, 2.0, 3.0);
    let look_at = Vec3::new(0.0, 0.0, 0.0);
    let vup = Vec3::new(0.0, 1.0, 0.0);
    let dist_to_focus = 10.0;
    let aperture = 0.1;
    let camera = Camera::new(look_from, look_at, vup, 20.0, ASPECT_RATIO, aperture, dist_to_focus);

    let file = File::create("output.ppm")?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "P3")?;
    writeln!(writer, "{} {}", IMAGE_WIDTH, IMAGE_HEIGHT)?;
    writeln!(writer, "255")?;

    for j in (0..IMAGE_HEIGHT).rev() {
        eprintln!("Scanlines remaining: {}", j);
        for i in 0..IMAGE_WIDTH {
            let mut pixel_color = Vec3::zero();
            for _ in 0..SAMPLES_PER_PIXEL {
                let u = (i as f64 + rng.next()) / (IMAGE_WIDTH - 1) as f64;
                let v = (j as f64 + rng.next()) / (IMAGE_HEIGHT - 1) as f64;
                let ray = camera.get_ray(u, v, &mut rng);
                pixel_color += ray_color(&ray, &world, MAX_DEPTH, &mut rng);
            }
            write_color(&mut writer, pixel_color, SAMPLES_PER_PIXEL)?;
        }
    }

    eprintln!("Done.");
    Ok(())
}

fn ray_color(ray: &Ray, world: &World, depth: usize, rng: &mut Rng) -> Vec3 {
    if depth == 0 {
        return Vec3::zero();
    }

    if let Some(hit) = world.hit(ray, 0.001, f64::INFINITY) {
        if let Some((attenuation, scattered)) = hit.material.scatter(ray, &hit, rng) {
            return attenuation * ray_color(&scattered, world, depth - 1, rng);
        }
        return Vec3::zero();
    }

    let unit_direction = ray.direction.unit();
    let t = 0.5 * (unit_direction.y + 1.0);
    Vec3::new(1.0, 1.0, 1.0) * (1.0 - t) + Vec3::new(0.5, 0.7, 1.0) * t
}

fn random_scene(rng: &mut Rng) -> World {
    let mut world = World::new();

    let ground_material = Material::Lambertian { albedo: Vec3::new(0.5, 0.5, 0.5) };
    world.add(Sphere::new(Vec3::new(0.0, -1000.0, 0.0), 1000.0, ground_material));

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = rng.next();
            let center = Vec3::new(
                a as f64 + 0.9 * rng.next(),
                0.2,
                b as f64 + 0.9 * rng.next(),
            );

            if (center - Vec3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                let material = if choose_mat < 0.8 {
                    let albedo = Vec3::random(rng) * Vec3::random(rng);
                    Material::Lambertian { albedo }
                } else if choose_mat < 0.95 {
                    let albedo = Vec3::random_range(rng, 0.5, 1.0);
                    let fuzz = rng.random_range(0.0, 0.5);
                    Material::Metal { albedo, fuzz }
                } else {
                    Material::Dielectric { ref_idx: 1.5 }
                };
                world.add(Sphere::new(center, 0.2, material));
            }
        }
    }

    world.add(Sphere::new(
        Vec3::new(0.0, 1.0, 0.0),
        1.0,
        Material::Dielectric { ref_idx: 1.5 },
    ));
    world.add(Sphere::new(
        Vec3::new(-4.0, 1.0, 0.0),
        1.0,
        Material::Lambertian {
            albedo: Vec3::new(0.4, 0.2, 0.1),
        },
    ));
    world.add(Sphere::new(
        Vec3::new(4.0, 1.0, 0.0),
        1.0,
        Material::Metal {
            albedo: Vec3::new(0.7, 0.6, 0.5),
            fuzz: 0.0,
        },
    ));

    world
}

fn write_color<W: Write>(writer: &mut W, pixel_color: Vec3, samples_per_pixel: usize) -> std::io::Result<()> {
    let scale = 1.0 / samples_per_pixel as f64;
    let r = (pixel_color.x * scale).sqrt();
    let g = (pixel_color.y * scale).sqrt();
    let b = (pixel_color.z * scale).sqrt();

    let ir = (256.0 * clamp(r, 0.0, 0.999)) as i32;
    let ig = (256.0 * clamp(g, 0.0, 0.999)) as i32;
    let ib = (256.0 * clamp(b, 0.0, 0.999)) as i32;

    writeln!(writer, "{} {} {}", ir, ig, ib)
}

fn clamp(x: f64, min: f64, max: f64) -> f64 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

#[derive(Clone, Copy, Debug)]
struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    fn at(&self, t: f64) -> Vec3 {
        self.origin + self.direction * t
    }
}

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    fn length_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(&self, other: &Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    fn unit(&self) -> Self {
        *self / self.length()
    }

    fn near_zero(&self) -> bool {
        let s = 1e-8;
        self.x.abs() < s && self.y.abs() < s && self.z.abs() < s
    }

    fn random(rng: &mut Rng) -> Self {
        Self::new(rng.next(), rng.next(), rng.next())
    }

    fn random_range(rng: &mut Rng, min: f64, max: f64) -> Self {
        Self::new(
            rng.random_range(min, max),
            rng.random_range(min, max),
            rng.random_range(min, max),
        )
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl Mul for Vec3 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

impl MulAssign for Vec3 {
    fn mul_assign(&mut self, other: Self) {
        self.x *= other.x;
        self.y *= other.y;
        self.z *= other.z;
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, t: f64) -> Self {
        Self::new(self.x * t, self.y * t, self.z * t)
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, t: f64) -> Self {
        self * (1.0 / t)
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[derive(Clone)]
struct Sphere {
    center: Vec3,
    radius: f64,
    material: Material,
}

impl Sphere {
    fn new(center: Vec3, radius: f64, material: Material) -> Self {
        Self {
            center,
            radius,
            material,
        }
    }

    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let a = ray.direction.length_squared();
        let half_b = oc.dot(&ray.direction);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrtd = discriminant.sqrt();

        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || t_max < root {
            root = (-half_b + sqrtd) / a;
            if root < t_min || t_max < root {
                return None;
            }
        }

        let point = ray.at(root);
        let outward_normal = (point - self.center) / self.radius;
        let mut record = HitRecord::new(point, outward_normal, root, ray);
        record.material = self.material.clone();
        Some(record)
    }
}

#[derive(Clone)]
struct World {
    objects: Vec<Sphere>,
}

impl World {
    fn new() -> Self {
        Self { objects: Vec::new() }
    }

    fn add(&mut self, object: Sphere) {
        self.objects.push(object);
    }

    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut temp_record = None;
        let mut closest_so_far = t_max;

        for object in &self.objects {
            if let Some(hit) = object.hit(ray, t_min, closest_so_far) {
                closest_so_far = hit.t;
                temp_record = Some(hit);
            }
        }

        temp_record
    }
}

#[derive(Clone)]
struct HitRecord {
    point: Vec3,
    normal: Vec3,
    t: f64,
    front_face: bool,
    material: Material,
}

impl HitRecord {
    fn new(point: Vec3, outward_normal: Vec3, t: f64, ray: &Ray) -> Self {
        let front_face = ray.direction.dot(&outward_normal) < 0.0;
        let normal = if front_face {
            outward_normal
        } else {
            -outward_normal
        };
        Self {
            point,
            normal,
            t,
            front_face,
            material: Material::Lambertian {
                albedo: Vec3::new(0.5, 0.5, 0.5),
            },
        }
    }
}

#[derive(Clone)]
enum Material {
    Lambertian { albedo: Vec3 },
    Metal { albedo: Vec3, fuzz: f64 },
    Dielectric { ref_idx: f64 },
}

impl Material {
    fn scatter(&self, _ray_in: &Ray, hit: &HitRecord, rng: &mut Rng) -> Option<(Vec3, Ray)> {
        match self {
            Material::Lambertian { albedo } => {
                let mut scatter_direction = hit.normal + random_unit_vector(rng);
                if scatter_direction.near_zero() {
                    scatter_direction = hit.normal;
                }
                Some((*albedo, Ray::new(hit.point, scatter_direction)))
            }
            Material::Metal { albedo, fuzz } => {
                let reflected = reflect(_ray_in.direction.unit(), hit.normal);
                let scattered = Ray::new(
                    hit.point,
                    reflected + *fuzz * random_in_unit_sphere(rng),
                );
                if scattered.direction.dot(&hit.normal) > 0.0 {
                    Some((*albedo, scattered))
                } else {
                    None
                }
            }
            Material::Dielectric { ref_idx } => {
                let attenuation = Vec3::new(1.0, 1.0, 1.0);
                let refraction_ratio = if hit.front_face {
                    1.0 / ref_idx
                } else {
                    *ref_idx
                };

                let unit_direction = _ray_in.direction.unit();
                let cos_theta = (-unit_direction).dot(&hit.normal).min(1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

                let cannot_refract = refraction_ratio * sin_theta > 1.0;
                let direction = if cannot_refract
                    || reflectance(cos_theta, refraction_ratio) > rng.next()
                {
                    reflect(unit_direction, hit.normal)
                } else {
                    refract(unit_direction, hit.normal, refraction_ratio)
                };

                Some((attenuation, Ray::new(hit.point, direction)))
            }
        }
    }
}

fn random_unit_vector(rng: &mut Rng) -> Vec3 {
    let a = rng.random_range(0.0, 2.0 * PI);
    let z = rng.random_range(-1.0, 1.0);
    let r = (1.0 - z * z).sqrt();
    Vec3::new(r * a.cos(), r * a.sin(), z)
}

fn random_in_unit_sphere(rng: &mut Rng) -> Vec3 {
    loop {
        let p = Vec3::random_range(rng, -1.0, 1.0);
        if p.length_squared() < 1.0 {
            return p;
        }
    }
}

fn random_in_unit_disk(rng: &mut Rng) -> Vec3 {
    loop {
        let p = Vec3::new(
            rng.random_range(-1.0, 1.0),
            rng.random_range(-1.0, 1.0),
            0.0,
        );
        if p.length_squared() < 1.0 {
            return p;
        }
    }
}

fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - 2.0 * v.dot(&n) * n
}

fn refract(uv: Vec3, n: Vec3, etai_over_etat: f64) -> Vec3 {
    let cos_theta = (-uv).dot(&n).min(1.0);
    let r_out_perp = etai_over_etat * (uv + cos_theta * n);
    let r_out_parallel = -(1.0 - r_out_perp.length_squared()).abs().sqrt() * n;
    r_out_perp + r_out_parallel
}

fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
    // Schlick's approximation for reflectance
    let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    lens_radius: f64,
}

impl Camera {
    fn new(
        look_from: Vec3,
        look_at: Vec3,
        vup: Vec3,
        vfov: f64,
        aspect_ratio: f64,
        aperture: f64,
        focus_dist: f64,
    ) -> Self {
        let theta = degrees_to_radians(vfov);
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (look_from - look_at).unit();
        let u = vup.cross(&w).unit();
        let v = w.cross(&u);

        let origin = look_from;
        let horizontal = focus_dist * viewport_width * u;
        let vertical = focus_dist * viewport_height * v;
        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - focus_dist * w;

        Self {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
            u,
            v,
            lens_radius: aperture / 2.0,
        }
    }

    fn get_ray(&self, s: f64, t: f64, rng: &mut Rng) -> Ray {
        let rd = self.lens_radius * random_in_unit_disk(rng);
        let offset = self.u * rd.x + self.v * rd.y;

        Ray::new(
            self.origin + offset,
            self.lower_left_corner + s * self.horizontal + t * self.vertical - self.origin - offset,
        )
    }
}

fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

struct Rng {
    state: u64,
}

impl Rng {
    fn with_seed(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    fn random_range(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next()
    }
}
