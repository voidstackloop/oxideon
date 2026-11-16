//! Classical image processing on grayscale [`Image`]s (row-major `f64`, nominally `[0, 1]`): [`convolve2d`],
//! [`gaussian_blur`], [`sobel`], [`canny`], [`otsu_threshold`], morphology ([`erode`], [`dilate`]),
//! [`connected_components`], [`integral_image`], [`equalize_histogram`], [`harris_response`],
//! [`distance_transform`], and bilinear [`resize`].

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f64>,
}

impl Image {
    pub fn new(w: usize, h: usize, data: Vec<f64>) -> Result<Self> {
        if w == 0 || h == 0 || data.len() != w * h {
            return Err(Error::Msg(format!("Image: {w}x{h} needs {} pixels, got {}", w * h, data.len())));
        }
        Ok(Self { w, h, data })
    }

    pub fn zeros(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![0.0; w * h] }
    }

    pub fn from_fn(w: usize, h: usize, f: impl Fn(usize, usize) -> f64) -> Self {
        Self { w, h, data: (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| f(x, y)).collect() }
    }

    pub fn at(&self, x: usize, y: usize) -> f64 {
        self.data[y * self.w + x]
    }

    /// Pixel with reflect-101 padding for out-of-range coordinates.
    fn get_reflect(&self, x: isize, y: isize) -> f64 {
        let r = |v: isize, n: usize| -> usize {
            let n = n as isize;
            let mut v = v;
            if n == 1 {
                return 0;
            }
            while v < 0 || v >= n {
                v = if v < 0 { -v } else { 2 * (n - 1) - v };
            }
            v as usize
        };
        self.at(r(x, self.w), r(y, self.h))
    }
}

/// 2-D correlation with an odd-sized `kernel` (rows of equal length), reflect padding.
pub fn convolve2d(img: &Image, kernel: &[Vec<f64>]) -> Result<Image> {
    let (kh, kw) = (kernel.len(), kernel.first().map_or(0, Vec::len));
    if kh % 2 == 0 || kw % 2 == 0 || kernel.iter().any(|r| r.len() != kw) {
        return Err(Error::Msg("convolve2d: the kernel must be rectangular with odd dimensions".into()));
    }
    let (rh, rw) = ((kh / 2) as isize, (kw / 2) as isize);
    Ok(Image::from_fn(img.w, img.h, |x, y| {
        let mut s = 0.0;
        for (j, row) in kernel.iter().enumerate() {
            for (i, k) in row.iter().enumerate() {
                s += k * img.get_reflect(x as isize + i as isize - rw, y as isize + j as isize - rh);
            }
        }
        s
    }))
}

/// Separable Gaussian blur with standard deviation `sigma` (kernel radius `ceil(3 sigma)`).
pub fn gaussian_blur(img: &Image, sigma: f64) -> Result<Image> {
    if sigma <= 0.0 {
        return Err(Error::Msg("gaussian_blur: sigma must be positive".into()));
    }
    let r = (3.0 * sigma).ceil() as isize;
    let k: Vec<f64> = (-r..=r).map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f64 = k.iter().sum();
    let k: Vec<f64> = k.iter().map(|v| v / s).collect();
    let row = convolve2d(img, std::slice::from_ref(&k))?;
    convolve2d(&row, &k.iter().map(|&v| vec![v]).collect::<Vec<_>>())
}

/// Sobel gradients: `(gx, gy)`.
pub fn sobel(img: &Image) -> Result<(Image, Image)> {
    let kx = vec![vec![-1.0, 0.0, 1.0], vec![-2.0, 0.0, 2.0], vec![-1.0, 0.0, 1.0]];
    let ky = vec![vec![-1.0, -2.0, -1.0], vec![0.0, 0.0, 0.0], vec![1.0, 2.0, 1.0]];
    Ok((convolve2d(img, &kx)?, convolve2d(img, &ky)?))
}

/// Canny edge detector: Gaussian smoothing, Sobel gradients, non-maximum suppression and hysteresis thresholding
/// (`low` / `high` apply to the gradient magnitude). Returns a binary edge map (1.0 = edge).
pub fn canny(img: &Image, sigma: f64, low: f64, high: f64) -> Result<Image> {
    if low > high {
        return Err(Error::Msg("canny: low threshold must not exceed the high threshold".into()));
    }
    let (gx, gy) = sobel(&gaussian_blur(img, sigma)?)?;
    let mag = Image::from_fn(img.w, img.h, |x, y| gx.at(x, y).hypot(gy.at(x, y)));
    let mut nms = Image::zeros(img.w, img.h);
    for y in 0..img.h {
        for x in 0..img.w {
            let m = mag.at(x, y);
            // quantize the gradient direction to 0/45/90/135 degrees
            let ang = gy.at(x, y).atan2(gx.at(x, y)).to_degrees().rem_euclid(180.0);
            let (dx, dy): (isize, isize) = if !(22.5..157.5).contains(&ang) { (1, 0) } else if ang < 67.5 { (1, 1) } else if ang < 112.5 { (0, 1) } else { (-1, 1) };
            let n1 = mag.get_reflect(x as isize + dx, y as isize + dy);
            let n2 = mag.get_reflect(x as isize - dx, y as isize - dy);
            if m >= n1 && m >= n2 {
                nms.data[y * img.w + x] = m;
            }
        }
    }
    let mut edge = Image::zeros(img.w, img.h);
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for y in 0..img.h {
        for x in 0..img.w {
            if nms.at(x, y) >= high && high > 0.0 {
                edge.data[y * img.w + x] = 1.0;
                stack.push((x, y));
            }
        }
    }
    while let Some((x, y)) = stack.pop() {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if nx >= 0 && ny >= 0 && (nx as usize) < img.w && (ny as usize) < img.h {
                    let (nx, ny) = (nx as usize, ny as usize);
                    if edge.at(nx, ny) == 0.0 && nms.at(nx, ny) >= low && nms.at(nx, ny) > 0.0 {
                        edge.data[ny * img.w + nx] = 1.0;
                        stack.push((nx, ny));
                    }
                }
            }
        }
    }
    Ok(edge)
}

/// Otsu's threshold of an image with values in `[0, 1]` (256-bin histogram): pixels `>` the result are foreground.
pub fn otsu_threshold(img: &Image) -> f64 {
    let mut hist = [0.0f64; 256];
    img.data.iter().for_each(|&v| hist[((v.clamp(0.0, 1.0) * 255.0).round()) as usize] += 1.0);
    let total = img.data.len() as f64;
    let sum_all: f64 = hist.iter().enumerate().map(|(i, h)| i as f64 * h).sum();
    let (mut w0, mut sum0, mut best, mut thr) = (0.0, 0.0, -1.0, 0usize);
    for (t, &h) in hist.iter().enumerate() {
        w0 += h;
        if w0 == 0.0 {
            continue;
        }
        let w1 = total - w0;
        if w1 == 0.0 {
            break;
        }
        sum0 += t as f64 * h;
        let (m0, m1) = (sum0 / w0, (sum_all - sum0) / w1);
        let between = w0 * w1 * (m0 - m1).powi(2);
        if between > best {
            best = between;
            thr = t;
        }
    }
    thr as f64 / 255.0
}

fn morph(img: &Image, radius: usize, take_max: bool) -> Image {
    let r = radius as isize;
    Image::from_fn(img.w, img.h, |x, y| {
        let mut acc = if take_max { f64::NEG_INFINITY } else { f64::INFINITY };
        for dy in -r..=r {
            for dx in -r..=r {
                let (nx, ny) = (x as isize + dx, y as isize + dy);
                let v = if nx < 0 || ny < 0 || nx >= img.w as isize || ny >= img.h as isize { if take_max { f64::NEG_INFINITY } else { f64::INFINITY } } else { img.at(nx as usize, ny as usize) };
                acc = if take_max { acc.max(v) } else { acc.min(v) };
            }
        }
        acc
    })
}

/// Grayscale / binary erosion with a square structuring element of the given radius (min filter).
pub fn erode(img: &Image, radius: usize) -> Image {
    morph(img, radius, false)
}

/// Grayscale / binary dilation (max filter).
pub fn dilate(img: &Image, radius: usize) -> Image {
    morph(img, radius, true)
}

/// Labels connected foreground (`> 0.5`) regions; returns `(labels, count)` with background label 0.
pub fn connected_components(img: &Image, eight_connected: bool) -> (Vec<usize>, usize) {
    let mut labels = vec![0usize; img.w * img.h];
    let mut count = 0;
    for start in 0..img.w * img.h {
        if img.data[start] <= 0.5 || labels[start] != 0 {
            continue;
        }
        count += 1;
        labels[start] = count;
        let mut stack = vec![start];
        while let Some(p) = stack.pop() {
            let (x, y) = ((p % img.w) as isize, (p / img.w) as isize);
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if (dx == 0 && dy == 0) || (!eight_connected && dx != 0 && dy != 0) {
                        continue;
                    }
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && ny >= 0 && nx < img.w as isize && ny < img.h as isize {
                        let q = ny as usize * img.w + nx as usize;
                        if img.data[q] > 0.5 && labels[q] == 0 {
                            labels[q] = count;
                            stack.push(q);
                        }
                    }
                }
            }
        }
    }
    (labels, count)
}

/// Summed-area table with a zero border: `(h + 1) x (w + 1)`, so any box sum is four lookups ([`box_sum`]).
pub fn integral_image(img: &Image) -> Vec<Vec<f64>> {
    let mut t = vec![vec![0.0; img.w + 1]; img.h + 1];
    for y in 0..img.h {
        for x in 0..img.w {
            t[y + 1][x + 1] = img.at(x, y) + t[y][x + 1] + t[y + 1][x] - t[y][x];
        }
    }
    t
}

/// Sum over the half-open box `[x0, x1) x [y0, y1)` from an integral image.
pub fn box_sum(t: &[Vec<f64>], x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
    t[y1][x1] - t[y0][x1] - t[y1][x0] + t[y0][x0]
}

/// Histogram equalization over 256 levels.
pub fn equalize_histogram(img: &Image) -> Image {
    let mut hist = [0usize; 256];
    let level = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as usize;
    img.data.iter().for_each(|&v| hist[level(v)] += 1);
    let mut cdf = [0usize; 256];
    let mut run = 0;
    for i in 0..256 {
        run += hist[i];
        cdf[i] = run;
    }
    let cdf_min = cdf.iter().cloned().find(|&c| c > 0).unwrap_or(0);
    let denom = (img.data.len() - cdf_min).max(1) as f64;
    Image { w: img.w, h: img.h, data: img.data.iter().map(|&v| (cdf[level(v)] - cdf_min) as f64 / denom).collect() }
}

/// Harris corner response `det(M) - k trace(M)^2` with a Gaussian window of standard deviation `sigma`.
pub fn harris_response(img: &Image, sigma: f64, k: f64) -> Result<Image> {
    let (gx, gy) = sobel(img)?;
    let prod = |a: &Image, b: &Image| Image { w: img.w, h: img.h, data: a.data.iter().zip(&b.data).map(|(p, q)| p * q).collect() };
    let (sxx, syy, sxy) = (gaussian_blur(&prod(&gx, &gx), sigma)?, gaussian_blur(&prod(&gy, &gy), sigma)?, gaussian_blur(&prod(&gx, &gy), sigma)?);
    Ok(Image::from_fn(img.w, img.h, |x, y| {
        let (a, b, c) = (sxx.at(x, y), syy.at(x, y), sxy.at(x, y));
        (a * b - c * c) - k * (a + b) * (a + b)
    }))
}

/// Euclidean distance (two-pass chamfer approximation with 3-4 weights scaled to pixels) from every pixel to the
/// nearest foreground (`> 0.5`) pixel.
pub fn distance_transform(img: &Image) -> Image {
    let inf = 1e9;
    let mut d = Image::from_fn(img.w, img.h, |x, y| if img.at(x, y) > 0.5 { 0.0 } else { inf });
    let (a, b) = (1.0, std::f64::consts::SQRT_2);
    for y in 0..img.h {
        for x in 0..img.w {
            let mut v = d.at(x, y);
            if x > 0 { v = v.min(d.at(x - 1, y) + a); }
            if y > 0 { v = v.min(d.at(x, y - 1) + a); }
            if x > 0 && y > 0 { v = v.min(d.at(x - 1, y - 1) + b); }
            if x + 1 < img.w && y > 0 { v = v.min(d.at(x + 1, y - 1) + b); }
            d.data[y * img.w + x] = v;
        }
    }
    for y in (0..img.h).rev() {
        for x in (0..img.w).rev() {
            let mut v = d.at(x, y);
            if x + 1 < img.w { v = v.min(d.at(x + 1, y) + a); }
            if y + 1 < img.h { v = v.min(d.at(x, y + 1) + a); }
            if x + 1 < img.w && y + 1 < img.h { v = v.min(d.at(x + 1, y + 1) + b); }
            if x > 0 && y + 1 < img.h { v = v.min(d.at(x - 1, y + 1) + b); }
            d.data[y * img.w + x] = v;
        }
    }
    d
}

/// Bilinear resize (pixel-centre alignment).
pub fn resize(img: &Image, w: usize, h: usize) -> Result<Image> {
    if w == 0 || h == 0 {
        return Err(Error::Msg("resize: target size must be positive".into()));
    }
    Ok(Image::from_fn(w, h, |x, y| {
        let fx = ((x as f64 + 0.5) * img.w as f64 / w as f64 - 0.5).clamp(0.0, img.w as f64 - 1.0);
        let fy = ((y as f64 + 0.5) * img.h as f64 / h as f64 - 0.5).clamp(0.0, img.h as f64 - 1.0);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(img.w - 1), (y0 + 1).min(img.h - 1));
        let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
        img.at(x0, y0) * (1.0 - tx) * (1.0 - ty) + img.at(x1, y0) * tx * (1.0 - ty) + img.at(x0, y1) * (1.0 - tx) * ty + img.at(x1, y1) * tx * ty
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(w: usize, h: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Image {
        Image::from_fn(w, h, |x, y| f64::from(u8::from(x >= x0 && x < x1 && y >= y0 && y < y1)))
    }

    #[test]
    fn convolution_and_blur() {
        let id = vec![vec![0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0]];
        let img = Image::from_fn(5, 4, |x, y| (x * 7 + y * 3) as f64);
        assert_eq!(convolve2d(&img, &id).unwrap(), img);
        let ones: Vec<Vec<f64>> = vec![vec![1.0 / 9.0; 3]; 3];
        let flat = Image::new(4, 4, vec![2.0; 16]).unwrap();
        assert!(convolve2d(&flat, &ones).unwrap().data.iter().all(|v| (v - 2.0).abs() < 1e-12));
        // blur preserves the mean of a constant image and lowers the peak of an impulse
        let impulse = Image::from_fn(15, 15, |x, y| f64::from(u8::from(x == 7 && y == 7)));
        let b = gaussian_blur(&impulse, 1.5).unwrap();
        assert!((b.data.iter().sum::<f64>() - 1.0).abs() < 1e-9 && b.at(7, 7) < 0.1 && b.at(7, 7) > b.at(7, 9));
        assert!(convolve2d(&flat, &[vec![1.0, 1.0]]).is_err() && gaussian_blur(&flat, 0.0).is_err() && Image::new(2, 2, vec![0.0]).is_err());
    }

    #[test]
    fn edges_thresholds_and_structure() {
        let img = square(24, 24, 6, 6, 18, 18);
        let (gx, _) = sobel(&img).unwrap();
        assert!(gx.at(6, 12) > 0.0 && gx.at(17, 12) < 0.0 && gx.at(12, 12) == 0.0, "left edge rises, right edge falls");
        let edges = canny(&img, 1.0, 0.5, 1.5).unwrap();
        let n = edges.data.iter().filter(|&&v| v > 0.0).count();
        assert!(n > 30 && n < 140, "thin outline of the square: {n} pixels");
        assert_eq!(edges.at(12, 12), 0.0, "interior is not an edge");
        assert_eq!(edges.at(0, 0), 0.0);
        assert!(canny(&img, 1.0, 2.0, 1.0).is_err());
        // Otsu separates two intensity populations
        let two = Image::from_fn(10, 10, |x, _| if x < 5 { 0.2 } else { 0.8 });
        let t = otsu_threshold(&two);
        assert!((0.2..0.8).contains(&t), "{t}");
        // morphology: erosion shrinks, dilation grows, opening removes an isolated speck
        let mut speck = square(20, 20, 5, 5, 15, 15);
        speck.data[2 * 20 + 2] = 1.0;
        let opened = dilate(&erode(&speck, 1), 1);
        assert_eq!(opened.at(2, 2), 0.0);
        assert_eq!(opened.at(10, 10), 1.0);
        assert!(erode(&speck, 1).data.iter().sum::<f64>() < speck.data.iter().sum::<f64>() && dilate(&speck, 1).data.iter().sum::<f64>() > speck.data.iter().sum::<f64>());
        // components: two squares + one diagonal-touching pixel
        let mut two_sq = square(12, 6, 0, 0, 3, 3);
        for (x, y) in [(6, 0), (7, 0), (6, 1), (7, 1), (3, 3)] {
            two_sq.data[y * 12 + x] = 1.0;
        }
        assert_eq!(connected_components(&two_sq, false).1, 3);
        assert_eq!(connected_components(&two_sq, true).1, 2, "(3,3) touches the first square diagonally");
    }

    #[test]
    fn integral_equalize_corners_distance_resize() {
        let img = Image::from_fn(6, 5, |x, y| (x + 2 * y) as f64);
        let t = integral_image(&img);
        let brute: f64 = (1..4).flat_map(|y| (2..5).map(move |x| (x, y))).map(|(x, y)| img.at(x, y)).sum();
        assert_eq!(box_sum(&t, 2, 1, 5, 4), brute);
        assert_eq!(box_sum(&t, 0, 0, 6, 5), img.data.iter().sum::<f64>());
        // equalization stretches a low-contrast image to the full range
        let dull = Image::from_fn(16, 16, |x, y| 0.4 + 0.1 * ((x + y) % 4) as f64 / 3.0);
        let eq = equalize_histogram(&dull);
        let (lo, hi) = eq.data.iter().fold((1.0f64, 0.0f64), |a, &v| (a.0.min(v), a.1.max(v)));
        assert!(lo == 0.0 && hi == 1.0);
        // Harris: the response at a square's corner exceeds that on an edge or in flat regions
        let sq = square(30, 30, 8, 8, 22, 22);
        let hr = harris_response(&sq, 1.5, 0.05).unwrap();
        assert!(hr.at(8, 8) > 0.0 && hr.at(8, 8) > hr.at(15, 8).max(hr.at(15, 15)) && hr.at(15, 8) <= 0.0 + 1e-9);
        // distance transform from a single seed
        let mut seed = Image::zeros(9, 9);
        seed.data[4 * 9 + 4] = 1.0;
        let d = distance_transform(&seed);
        assert_eq!(d.at(4, 4), 0.0);
        assert_eq!(d.at(8, 4), 4.0);
        assert!((d.at(7, 7) - 3.0 * std::f64::consts::SQRT_2).abs() < 1e-12);
        // resize: constant images stay constant, linear ramps stay linear
        let ramp = Image::from_fn(4, 4, |x, _| x as f64);
        let up = resize(&ramp, 8, 8).unwrap();
        assert!(up.data.iter().all(|v| (0.0..=3.0).contains(v)) && up.at(0, 0) == 0.0 && up.at(7, 3) == 3.0);
        assert!((up.at(3, 0) - 1.25).abs() < 1e-12);
        assert!(resize(&ramp, 0, 3).is_err());
    }
}
