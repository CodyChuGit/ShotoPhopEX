//! PNG in and out, through `photocraft-codecs`. Images cross the server boundary as PNG because
//! used for saved results and thumbnails.

use photocraft_codecs::{ChannelLayout, EncodeOptions, Format, Image, PngCompression, SampleType};

use crate::{Error, Gray8, Result, Rgba8};

fn enc(e: photocraft_codecs::CodecError) -> Error {
    Error::Image(e.to_string())
}

/// Uploads cross a loopback socket and are read once: the fastest deflate level wins (lossless
/// either way; a megapixel RGBA encodes in a few milliseconds instead of tens).
fn upload_options() -> EncodeOptions {
    EncodeOptions { png_compression: PngCompression::Fast, embed_icc: false, embed_metadata: false, ..EncodeOptions::default() }
}

/// Encode straight-alpha RGBA8 as PNG.
pub fn encode_rgba8(img: &Rgba8) -> Result<Vec<u8>> {
    let image = Image::from_u8(img.width, img.height, ChannelLayout::Rgba, img.data.clone()).map_err(enc)?;
    photocraft_codecs::encode(&image, Format::Png, &upload_options()).map_err(enc)
}

/// Encode an 8-bit mask as a grayscale PNG.
pub fn encode_gray8(mask: &Gray8) -> Result<Vec<u8>> {
    let image = Image::from_u8(mask.width, mask.height, ChannelLayout::Gray, mask.data.clone()).map_err(enc)?;
    photocraft_codecs::encode(&image, Format::Png, &upload_options()).map_err(enc)
}

/// Decode any image the codecs know into RGBA8 (16-bit and float results are quantised, gray is
/// expanded; CMYK is refused because a generative server never produces it).
pub fn decode_rgba8(bytes: &[u8]) -> Result<Rgba8> {
    let image = photocraft_codecs::decode(bytes).map_err(enc)?;
    if image.layout().is_cmyk() {
        return Err(Error::Image("the server returned a CMYK image".into()));
    }
    let rgba = image.converted(ChannelLayout::Rgba, SampleType::U8);
    Rgba8::new(rgba.width(), rgba.height(), rgba.data().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_round_trips_through_png() {
        let mut data = Vec::new();
        for i in 0..(5 * 3 * 4) {
            data.push((i * 7 % 256) as u8);
        }
        let img = Rgba8::new(5, 3, data).unwrap_or_else(|e| panic!("{e}"));
        let png = encode_rgba8(&img).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(&png[1..4], b"PNG");
        let back = decode_rgba8(&png).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(back, img);
    }

    #[test]
    fn gray_masks_decode_as_rgba() {
        let mask = Gray8::new(4, 2, vec![0, 64, 128, 255, 255, 128, 64, 0]).unwrap_or_else(|e| panic!("{e}"));
        let png = encode_gray8(&mask).unwrap_or_else(|e| panic!("{e}"));
        let back = decode_rgba8(&png).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(back.get(3, 0), Some([255, 255, 255, 255]));
        assert_eq!(back.get(0, 0), Some([0, 0, 0, 255]));
    }

    #[test]
    fn garbage_is_an_error_not_a_panic() {
        assert!(decode_rgba8(b"not a png").is_err());
        assert!(decode_rgba8(&[]).is_err());
    }
}
