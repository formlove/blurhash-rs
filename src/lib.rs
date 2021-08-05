mod ac;
mod base83;
mod dc;
// mod error;
mod simd;
mod util;

// pub use error::Error;

use base83::encode_no_alloc;
use simd::find_abs_max;
use std::f32::consts::PI;
pub use util::{linear_to_srgb, pre_compute_data_linear, srgb_to_linear};

/// Calculates the blurhash for an image using the given x and y component counts.
pub fn encode(
    components_x: u32,
    components_y: u32,
    width: u32,
    height: u32,
    rgba_image: &[u8],
) -> String {
    if !(1..=9).contains(&components_x) || !(1..=9).contains(&components_y) {
        panic!("BlurHash must have between 1 and 9 components");
    }

    let pre_computed = pre_compute_data_linear(rgba_image);
    let factors: Vec<[f32; 4]> = (0..components_y)
        .into_iter()
        .map(|y| {
            (0..components_x)
                .into_iter()
                .map(|x| (x, y))
                .collect::<Vec<(u32, u32)>>()
        })
        .flatten()
        .map(|(x, y)| multiply_basis_function(x, y, width, height, &pre_computed))
        .collect();

    let dc = &factors[0];
    let ac = &factors[1..];

    let mut blurhash = String::new();

    let size_flag = (components_x - 1) + (components_y - 1) * 9;
    blurhash.push_str(&base83::encode(size_flag, 1));

    let maximum_value: f32;
    if !ac.is_empty() {
        let actual_maximum_value = unsafe { find_abs_max(ac) };
        let quantised_maximum_value = f32::max(
            0.0,
            f32::min(82., f32::floor(actual_maximum_value * 166. - 0.5)),
        ) as u32;

        maximum_value = (quantised_maximum_value + 1) as f32 / 166.;
        encode_no_alloc(quantised_maximum_value, 1, &mut blurhash);
    } else {
        maximum_value = 1.;
        encode_no_alloc(0, 1, &mut blurhash);
    }

    encode_no_alloc(dc::encode(dc), 4, &mut blurhash);

    for i in 0..components_y * components_x - 1 {
        encode_no_alloc(ac::encode(&ac[i as usize], maximum_value), 2, &mut blurhash);
    }

    blurhash
}

fn multiply_basis_function(
    component_x: u32,
    component_y: u32,
    width: u32,
    height: u32,
    computed: &Vec<f32>,
) -> [f32; 4] {
    let mut r = 0.;
    let mut g = 0.;
    let mut b = 0.;
    let normalisation = match (component_x, component_y) {
        (0, 0) => 1.,
        _ => 2.,
    };

    let inverse_width = 1.0 / width as f32;
    let inverse_height = 1.0 / height as f32;

    let x_lut: Vec<f32> = (0..width)
        .map(|x| f32::cos(PI * component_x as f32 * x as f32 * inverse_width))
        .collect();

    let y_lut: Vec<f32> = (0..height)
        .map(|y| f32::cos(PI * component_y as f32 * y as f32 * inverse_height))
        .collect();

    let mut x = 0;
    let mut y = 0;

    for rgba in computed.chunks_exact(4) {
        let basis = unsafe { *x_lut.get_unchecked(x) * *y_lut.get_unchecked(y) };

        r += basis * rgba[0];
        g += basis * rgba[1];
        b += basis * rgba[2];

        x += 1;
        if x >= width as usize {
            x = 0;
            y += 1;
        }
    }

    let scale = normalisation / (width * height) as f32;

    [r * scale, g * scale, b * scale, 0.]
}

/// Decodes the given blurhash to an image of the specified size.
///
/// The punch parameter can be used to de- or increase the contrast of the
/// resulting image.
pub fn decode(blurhash: &str, width: u32, height: u32, punch: f32) -> Vec<u8> {
    let (num_x, num_y) = components(blurhash);

    let quantised_maximum_value = base83::decode(&blurhash[1..2]);
    let maximum_value = (quantised_maximum_value + 1) as f32 / 166.;

    let mut colors = vec![[0.; 3]; num_x * num_y];

    for i in 0..colors.len() {
        if i == 0 {
            let value = base83::decode(&blurhash[2..6]);
            colors[i as usize] = dc::decode(value as u32);
        } else {
            let value = base83::decode(&blurhash[4 + i * 2..6 + i * 2]);
            colors[i as usize] = ac::decode(value as u32, maximum_value * punch);
        }
    }

    let bytes_per_row = width * 4;
    let mut pixels = vec![0; (bytes_per_row * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let mut pixel = [0.; 3];

            for j in 0..num_y {
                for i in 0..num_x {
                    let basis = f32::cos((PI * x as f32 * i as f32) / width as f32)
                        * f32::cos((PI * y as f32 * j as f32) / height as f32);
                    let color = &colors[i + j * num_x as usize];

                    pixel[0] += color[0] * basis;
                    pixel[1] += color[1] * basis;
                    pixel[2] += color[2] * basis;
                }
            }

            let int_r = linear_to_srgb(pixel[0]);
            let int_g = linear_to_srgb(pixel[1]);
            let int_b = linear_to_srgb(pixel[2]);

            pixels[(4 * x + y * bytes_per_row) as usize] = int_r as u8;
            pixels[(4 * x + 1 + y * bytes_per_row) as usize] = int_g as u8;
            pixels[(4 * x + 2 + y * bytes_per_row) as usize] = int_b as u8;
            pixels[(4 * x + 3 + y * bytes_per_row) as usize] = 255u8;
        }
    }
    pixels
}

fn components(blurhash: &str) -> (usize, usize) {
    if blurhash.len() < 6 {
        panic!("The blurhash string must be at least 6 characters");
    }

    let size_flag = base83::decode(&blurhash[0..1]);
    let num_y = (f32::floor(size_flag as f32 / 9.) + 1.) as usize;
    let num_x = (size_flag % 9) + 1;

    let expected = 4 + 2 * num_x * num_y;
    if blurhash.len() != expected {
        panic!(
            "blurhash length mismatch: length is {} but it should be {}",
            blurhash.len(),
            (4 + 2 * num_x * num_y)
        );
    }

    (num_x, num_y)
}

#[cfg(test)]
mod tests {
    use super::{decode, encode};
    use image::{save_buffer, ColorType::Rgba8};
    use image::{EncodableLayout, GenericImageView};
    use std::time::Instant;

    #[test]
    fn decode_blurhash() {
        let img = image::open("octocat.png").unwrap();
        let (origwidth, origheight) = img.dimensions();

        let start = Instant::now();
        // let img = img.thumbnail(32, 32);
        let (width, height) = img.dimensions();

        let blurhash = encode(2, 4, width, height, img.to_rgba8().as_bytes()).unwrap();
        println!("This took {:?} sting is {}", start.elapsed(), blurhash);
        let img = decode(&blurhash, origwidth, origheight, 1.0);
        save_buffer("out4.png", &img, origwidth, origheight, RGBA(8)).unwrap();

        // assert_eq!(img[0..5], [45, 1, 56, 255, 45]);
    }
}
