//! Live preview while generating (`integrations.previewWhileGenerating`): while a generative job
//! runs on a document, the latest picture the model made of its result
//! ([`photocraft_engine::jobs::JobPreview`], from `Session::job_previews`: one, or one per tile
//! of a batched Upscale request) is drawn over the place the result will take, in document space with the canvas's own mapping, until the result lands.
//! The picture is usually the model's latent grid (an eighth or a sixteenth of the result), so it
//! is drawn with linear filtering and scaled up smoothly. A texture is uploaded only when the
//! job shows a new picture ([`JobPreview::version`]) and is dropped once the job ends.

use std::time::Duration;

use egui::epaint::{Mesh, Vertex};
use egui::{Color32, Pos2, TextureHandle, TextureOptions, pos2};
use photocraft_doc::DocId;
use photocraft_engine::jobs::{JobId, JobPreview};

use crate::PhotocraftApp;
use crate::canvas::ViewXform;

/// How often the canvas looks for a newer picture while one is shown.
const POLL: Duration = Duration::from_millis(100);

/// A cached upload: the picture's version and its texture.
type Cached = (u64, TextureHandle);

/// Where the outputs drawn on `doc`'s canvas last frame are listed.
fn canvas_key(doc: DocId) -> egui::Id {
    egui::Id::new(("pc-gen-preview", doc.0))
}

/// The upload of output `output`'s picture on `doc`'s canvas.
fn output_key(doc: DocId, output: u32) -> egui::Id {
    egui::Id::new(("pc-gen-preview", doc.0, output))
}

fn row_key(record: u64) -> egui::Id {
    egui::Id::new(("pc-gen-preview-row", record))
}

/// The texture of `p`, uploaded once per picture and kept under `key` between frames. `None`
/// when the bytes do not match the size (never panics on a bad picture).
pub(crate) fn texture(ctx: &egui::Context, key: egui::Id, p: &JobPreview) -> Option<TextureHandle> {
    let cached = ctx.data(|d| d.get_temp::<Cached>(key));
    if let Some((version, tex)) = &cached
        && *version == p.version
    {
        return Some(tex.clone());
    }
    let (w, h) = (p.width as usize, p.height as usize);
    if w == 0 || h == 0 || w.checked_mul(h).and_then(|n| n.checked_mul(4)) != Some(p.rgba.len()) {
        return None;
    }
    let image = egui::ColorImage::from_rgba_unmultiplied([w, h], &p.rgba);
    let tex = match cached {
        Some((_, mut tex)) => {
            tex.set(image, TextureOptions::LINEAR);
            tex
        }
        None => ctx.load_texture("generation-preview", image, TextureOptions::LINEAR),
    };
    ctx.data_mut(|d| d.insert_temp(key, (p.version, tex.clone())));
    Some(tex)
}

/// Forget the upload under `key` (its texture is freed with the last handle).
fn drop_texture(ctx: &egui::Context, key: egui::Id) {
    ctx.data_mut(|d| d.remove::<Cached>(key));
}

/// The four screen corners of document rectangle `r` (clockwise from its top-left), through the
/// view's zoom, rotation, flip and pixel aspect.
pub(crate) fn corners(xf: &ViewXform, r: photocraft_geom::Rect) -> [Pos2; 4] {
    let (x0, y0, x1, y1) = (r.x0 as f32, r.y0 as f32, r.x1 as f32, r.y1 as f32);
    [xf.to_screen(x0, y0), xf.to_screen(x1, y0), xf.to_screen(x1, y1), xf.to_screen(x0, y1)]
}

/// Draw the pictures of the generation running on `doc`, if it shows any, each over its place
/// (one, or one per tile of a batched Upscale request). Called by the canvas right after the
/// document itself, so selection outlines and tool overlays stay on top.
pub(crate) fn paint(app: &PhotocraftApp, ctx: &egui::Context, painter: &egui::Painter, xf: &ViewXform, doc: DocId) {
    let shown = app.session.job_previews(doc);
    let list = canvas_key(doc);
    let before = ctx.data(|d| d.get_temp::<Vec<u32>>(list)).unwrap_or_default();
    let now: Vec<u32> = shown.iter().map(|p| p.output).collect();
    // Outputs no longer shown (the job ended, or moved on to other tiles) give their textures back.
    for gone in before.iter().filter(|o| !now.contains(o)) {
        drop_texture(ctx, output_key(doc, *gone));
    }
    if now.is_empty() {
        if !before.is_empty() {
            ctx.data_mut(|d| d.remove::<Vec<u32>>(list));
        }
        return;
    }
    ctx.data_mut(|d| d.insert_temp(list, now));
    let uvs = [pos2(0.0, 0.0), pos2(1.0, 0.0), pos2(1.0, 1.0), pos2(0.0, 1.0)];
    for p in &shown {
        let Some(tex) = texture(ctx, output_key(doc, p.output), p) else { continue };
        let mut mesh = Mesh::with_texture(tex.id());
        for (pos, uv) in corners(xf, p.rect).into_iter().zip(uvs) {
            mesh.vertices.push(Vertex { pos, uv, color: Color32::WHITE });
        }
        mesh.add_triangle(0, 1, 2);
        mesh.add_triangle(0, 2, 3);
        painter.add(egui::Shape::mesh(mesh));
    }
    ctx.request_repaint_after(POLL);
}

/// The latest picture of generation record `record` while its job runs (the Generations
/// panel's row thumbnail), else `None` and its upload is dropped.
pub(crate) fn row_texture(app: &PhotocraftApp, ctx: &egui::Context, record: u64) -> Option<TextureHandle> {
    let key = row_key(record);
    let shown = app.session.generations.job_for_record(record).and_then(|j: JobId| app.session.job_preview_of(j));
    match shown {
        Some(p) => {
            ctx.request_repaint_after(POLL);
            texture(ctx, key, &p)
        }
        None => {
            drop_texture(ctx, key);
            None
        }
    }
}

/// Drop the row thumbnail of a generation that is no longer running.
pub(crate) fn forget_row(ctx: &egui::Context, record: u64) {
    drop_texture(ctx, row_key(record));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn picture(version: u64, level: u8) -> JobPreview {
        JobPreview {
            rect: photocraft_geom::Rect::new(10, 20, 74, 52),
            width: 8,
            height: 4,
            rgba: Arc::new([level, level, level, 255].repeat(32)),
            step: 1,
            steps: 4,
            output: 0,
            version,
        }
    }

    /// One upload per picture: the same version reuses the texture, a new one replaces its
    /// pixels in place, a malformed one is refused.
    #[test]
    fn a_picture_is_uploaded_once_per_version() {
        let ctx = egui::Context::default();
        let key = egui::Id::new("t");
        let a = texture(&ctx, key, &picture(1, 10)).unwrap();
        let again = texture(&ctx, key, &picture(1, 99)).unwrap();
        assert_eq!(a.id(), again.id());
        let next = texture(&ctx, key, &picture(2, 50)).unwrap();
        assert_eq!(next.id(), a.id(), "the same texture, new pixels");
        assert_eq!(ctx.data(|d| d.get_temp::<Cached>(key)).map(|c| c.0), Some(2));
        let bad = JobPreview { rgba: Arc::new(vec![0; 3]), ..picture(3, 0) };
        assert!(texture(&ctx, egui::Id::new("u"), &bad).is_none());
        drop_texture(&ctx, key);
        assert!(ctx.data(|d| d.get_temp::<Cached>(key)).is_none());
    }

    /// The picture spans its document rectangle on screen, under the view's mapping.
    #[test]
    fn the_picture_covers_its_place_on_screen() {
        let xf = ViewXform {
            rect: egui::Rect::from_min_size(Pos2::ZERO, egui::vec2(200.0, 100.0)),
            zoom: 2.0,
            center: [50.0, 25.0],
            flip: false,
            rotation: 0.0,
            aspect: 1.0,
        };
        let c = corners(&xf, photocraft_geom::Rect::new(10, 20, 74, 52));
        assert_eq!(c[0], pos2(20.0, 40.0));
        assert_eq!(c[2], pos2(148.0, 104.0));
        let flipped = ViewXform { flip: true, ..xf };
        let f = corners(&flipped, photocraft_geom::Rect::new(10, 20, 74, 52));
        assert_eq!(f[0].x, 180.0, "a flipped view mirrors the place");
    }
}
