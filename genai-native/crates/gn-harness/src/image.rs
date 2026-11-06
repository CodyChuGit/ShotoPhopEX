//! Pictures and masks as straight RGBA8, read from and written to PNG (any PNG colour type and
//! depth is normalised to 8-bit RGBA on the way in).

use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Straight-alpha RGBA8, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Image {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, data: vec![0; width as usize * height as usize * 4] }
    }

    pub fn read(path: &Path) -> crate::Result<Self> {
        let f = std::fs::File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let mut decoder = png::Decoder::new(BufReader::new(f));
        decoder.set_transformations(png::Transformations::normalize_to_color8());
        let mut reader = decoder.read_info().map_err(|e| format!("{} is not a PNG: {e}", path.display()))?;
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
        let n = info.width as usize * info.height as usize;
        let px: &[u8] = buf.get(..info.buffer_size()).ok_or("short PNG frame")?;
        let mut data = Vec::with_capacity(n * 4);
        match info.color_type {
            png::ColorType::Rgba => data.extend_from_slice(px),
            png::ColorType::Rgb => {
                for [r, g, b] in px.as_chunks::<3>().0 {
                    data.extend_from_slice(&[*r, *g, *b, 255]);
                }
            }
            png::ColorType::Grayscale => {
                for g in px {
                    data.extend_from_slice(&[*g, *g, *g, 255]);
                }
            }
            png::ColorType::GrayscaleAlpha => {
                for [g, a] in px.as_chunks::<2>().0 {
                    data.extend_from_slice(&[*g, *g, *g, *a]);
                }
            }
            png::ColorType::Indexed => return Err(format!("{}: indexed PNG without expansion", path.display())),
        }
        if data.len() != n * 4 {
            return Err(format!("{}: decoded {} bytes for {n} pixels", path.display(), data.len()));
        }
        Ok(Self { width: info.width, height: info.height, data })
    }

    pub fn write(&self, path: &Path) -> crate::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        }
        let f = std::fs::File::create(path).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        let mut enc = png::Encoder::new(BufWriter::new(f), self.width, self.height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().map_err(|e| format!("{}: {e}", path.display()))?;
        w.write_image_data(&self.data).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Luma in `0..=1` per pixel (Rec. 709 weights on the straight colour).
    pub fn luma(&self) -> Vec<f32> {
        self.data.as_chunks::<4>().0.iter().map(|[r, g, b, _]| (0.2126 * f32::from(*r) + 0.7152 * f32::from(*g) + 0.0722 * f32::from(*b)) / 255.0).collect()
    }

    /// Alpha as coverage in `0..=1` (a mask PNG's shape).
    pub fn coverage(&self) -> Vec<f32> {
        self.data.as_chunks::<4>().0.iter().map(|[_, _, _, a]| f32::from(*a) / 255.0).collect()
    }

    /// Red channel as coverage in `0..=1` (a mask saved as grey).
    pub fn grey_coverage(&self) -> Vec<f32> {
        self.data.as_chunks::<4>().0.iter().map(|[r, _, _, _]| f32::from(*r) / 255.0).collect()
    }

    pub fn same_size(&self, o: &Self) -> bool {
        self.width == o.width && self.height == o.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_round_trips_and_normalises() {
        let d = std::env::temp_dir().join(format!("gn-image-{}", std::process::id()));
        let mut im = Image::new(3, 2);
        for (i, p) in im.data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            *p = [i as u8 * 40, 255 - i as u8 * 40, 7, 200];
        }
        let p = d.join("a.png");
        im.write(&p).unwrap_or_else(|e| panic!("{e}"));
        let back = Image::read(&p).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(back, im);
        assert_eq!(back.luma().len(), 6);
        assert!((back.coverage()[0] - 200.0 / 255.0).abs() < 1e-6);
        let _ = std::fs::remove_dir_all(d);
    }
}
