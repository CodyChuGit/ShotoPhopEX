//! Edit › Preferences › App Icon… (fork): the app's own icon, made the user's. Any image becomes
//! the title-bar mark and the window/taskbar icon: one chosen from disk, pasted, or downloaded
//! from an address. Each one joins a small gallery kept in the preferences (`appIcon`, as square
//! PNGs of at most 256 px), so switching between them, or back to the built-in icon, is one
//! click. Nothing ships in the repository; the executable's own file icon (Explorer, a pinned
//! shortcut) stays the built-in one, since it is baked in at build time.
//!
//! A click on the title bar's mark opens the window too. `edit.appIcon` takes `use`, `remove` and
//! `add` (`{name, data}`, the image's bytes in base64) for automation.

use std::sync::{Arc, Mutex};

use base64::Engine;
use egui::{Color32, Rect, Sense, Stroke, TextureHandle, TextureOptions, pos2, vec2};
use photocraft_codecs::{ChannelLayout, EncodeOptions, Format, Image, SampleType};
use photocraft_engine::prefs::CustomIcon;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;

pub const COMMAND: &str = "edit.appIcon";
/// The largest side an icon is kept at (Windows' biggest shell icon).
pub const MAX_SIDE: u32 = 256;
/// How many custom icons the gallery keeps (they live in the preferences file).
pub const MAX_ICONS: usize = 12;
/// The largest image file taken in.
const MAX_INPUT_BYTES: usize = 40 * 1024 * 1024;
/// The built-in icon, for going back to it while the app runs (the app sets it at launch).
const BUILT_IN_256: &[u8] = include_bytes!("../../../assets/app-icon/hicolor/256x256/apps/ai.storyteller.photocraft.png");

/// A download's outcome: the bytes, or why it failed.
type Fetched = Result<Vec<u8>, String>;

/// A download running on its own thread: the slot it fills in.
#[derive(Clone, Debug, Default)]
pub struct Pending(Arc<Mutex<Option<Fetched>>>);

impl PartialEq for Pending {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// The window's state in `UiState::app_icon`. Never saved.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AppIconUi {
    pub open: bool,
    /// The address typed for Download.
    pub url: String,
    /// The download in flight, and the name its icon will take.
    pub pending: Option<(Pending, String)>,
    /// The last thing that went wrong, shown in the window.
    pub error: Option<String>,
    /// The icon applied to the window and the title bar: (name, PNG length); `None` until the
    /// first frame looks.
    pub applied: Option<(String, usize)>,
}

/// The custom icon in use, if the preference names one that exists.
fn active(app: &PhotocraftApp) -> Option<&CustomIcon> {
    let p = &app.session.prefs().app_icon;
    p.icons.iter().find(|i| !p.active.is_empty() && i.name == p.active)
}

/// RGBA8 pixels of a PNG (or any image the codecs read).
fn rgba(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let img = photocraft_codecs::decode(bytes).map_err(|e| e.to_string())?.convert(ChannelLayout::Rgba, SampleType::U8);
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 || img.data().len() != w as usize * h as usize * 4 {
        return Err("the image is empty".into());
    }
    Ok((w, h, img.data().to_vec()))
}

/// Fit an image into a transparent square of at most [`MAX_SIDE`], centred, never enlarged;
/// a box filter (alpha-weighted) shrinks it.
pub fn fit_square(w: u32, h: u32, px: &[u8]) -> (u32, Vec<u8>) {
    let side = w.max(h).clamp(1, MAX_SIDE);
    let big = u64::from(w.max(h).max(1));
    let cw = ((u64::from(w) * u64::from(side)).div_ceil(big) as u32).clamp(1, side);
    let ch = ((u64::from(h) * u64::from(side)).div_ceil(big) as u32).clamp(1, side);
    let (ox, oy) = ((side - cw) / 2, (side - ch) / 2);
    let mut out = vec![0u8; side as usize * side as usize * 4];
    for dy in 0..ch {
        let y0 = (u64::from(dy) * u64::from(h) / u64::from(ch)) as u32;
        let y1 = ((u64::from(dy + 1) * u64::from(h) / u64::from(ch)) as u32).clamp(y0 + 1, h.max(1));
        for dx in 0..cw {
            let x0 = (u64::from(dx) * u64::from(w) / u64::from(cw)) as u32;
            let x1 = ((u64::from(dx + 1) * u64::from(w) / u64::from(cw)) as u32).clamp(x0 + 1, w.max(1));
            let (mut r, mut g, mut b, mut a, mut n) = (0u64, 0u64, 0u64, 0u64, 0u64);
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y as usize * w as usize + x as usize) * 4;
                    let Some(p) = px.get(i..i + 4) else { continue };
                    let pa = u64::from(p[3]);
                    r += u64::from(p[0]) * pa;
                    g += u64::from(p[1]) * pa;
                    b += u64::from(p[2]) * pa;
                    a += pa;
                    n += 1;
                }
            }
            let o = ((dy + oy) as usize * side as usize + (dx + ox) as usize) * 4;
            if a > 0 && n > 0 {
                out[o] = (r / a) as u8;
                out[o + 1] = (g / a) as u8;
                out[o + 2] = (b / a) as u8;
                out[o + 3] = (a / n) as u8;
            }
        }
    }
    (side, out)
}

/// Any image's bytes as an icon: a square PNG of at most [`MAX_SIDE`].
pub fn to_icon_png(bytes: &[u8]) -> Result<Vec<u8>, String> {
    if bytes.len() > MAX_INPUT_BYTES {
        return Err(format!("the image is larger than {} MB", MAX_INPUT_BYTES / (1024 * 1024)));
    }
    let (w, h, px) = rgba(bytes).map_err(|e| format!("not an image this app reads ({e})"))?;
    rgba_to_icon_png(w, h, &px)
}

fn rgba_to_icon_png(w: u32, h: u32, px: &[u8]) -> Result<Vec<u8>, String> {
    let (side, sq) = fit_square(w, h, px);
    let img = Image::from_u8(side, side, ChannelLayout::Rgba, sq).map_err(|e| e.to_string())?;
    photocraft_codecs::encode(&img, Format::Png, &EncodeOptions::default()).map_err(|e| e.to_string())
}

/// A name for a new icon from a file name or address: its stem, unique in the gallery.
fn unique_name(app: &PhotocraftApp, source: &str) -> String {
    let last = source.trim_end_matches(['/', '\\']).rsplit(['/', '\\']).next().unwrap_or("");
    let last = last.split(['?', '#']).next().unwrap_or("");
    let stem = last.rsplit_once('.').map_or(last, |(s, _)| s).trim();
    let base: String = if stem.is_empty() { "Icon".to_string() } else { stem.chars().take(40).collect() };
    let taken = |n: &str| app.session.prefs().app_icon.icons.iter().any(|i| i.name == n);
    if !taken(&base) {
        return base;
    }
    (2..).map(|k| format!("{base} {k}")).find(|n| !taken(n)).unwrap_or(base)
}

/// Add an icon to the gallery from an icon PNG and use it.
fn add_png(app: &mut PhotocraftApp, name: String, png: &[u8]) -> Result<Value, String> {
    if app.session.prefs().app_icon.icons.len() >= MAX_ICONS {
        return Err(format!("the gallery holds {MAX_ICONS} icons; remove one first (right-click it)"));
    }
    let data = base64::engine::general_purpose::STANDARD.encode(png);
    app.session.edit_prefs(|p| {
        p.app_icon.icons.push(CustomIcon { name: name.clone(), png: data });
        p.app_icon.active = name.clone();
    });
    app.ui.app_icon.applied = None;
    app.ui.app_icon.error = None;
    Ok(json!({"added": name}))
}

/// Add any image's bytes, named after `source` (a file name or address).
pub fn add_image(app: &mut PhotocraftApp, source: &str, bytes: &[u8]) -> Result<Value, String> {
    let png = to_icon_png(bytes)?;
    let name = unique_name(app, source);
    add_png(app, name, &png)
}

/// Use a gallery icon by name ("" is the built-in one).
pub fn use_icon(app: &mut PhotocraftApp, name: &str) -> Result<Value, String> {
    if !name.is_empty() && !app.session.prefs().app_icon.icons.iter().any(|i| i.name == name) {
        return Err(format!("there is no app icon named `{name}`"));
    }
    let name = name.to_string();
    app.session.edit_prefs(|p| p.app_icon.active = name.clone());
    Ok(json!({"active": name}))
}

/// Remove a gallery icon; the built-in one returns if it was in use.
pub fn remove(app: &mut PhotocraftApp, name: &str) -> Result<Value, String> {
    if !app.session.prefs().app_icon.icons.iter().any(|i| i.name == name) {
        return Err(format!("there is no app icon named `{name}`"));
    }
    let name = name.to_string();
    app.session.edit_prefs(|p| {
        p.app_icon.icons.retain(|i| i.name != name);
        if p.app_icon.active == name {
            p.app_icon.active.clear();
        }
    });
    Ok(json!({"removed": name}))
}

/// Choose Image…: any file the codecs read.
fn choose(app: &mut PhotocraftApp) -> Result<Value, String> {
    let (name, bytes) = app.pick_file_bytes().ok_or_else(|| "cancelled".to_string())??;
    add_image(app, &name, &bytes)
}

/// Paste Image: the clipboard's image.
fn paste(app: &mut PhotocraftApp) -> Result<Value, String> {
    let get = app.services.clipboard_get_image.as_mut().ok_or("no clipboard here")?;
    let (w, h, px) = get().ok_or("the clipboard holds no image")?;
    if w == 0 || h == 0 || px.len() != w as usize * h as usize * 4 {
        return Err("the clipboard's image is empty".into());
    }
    let png = rgba_to_icon_png(w, h, &px)?;
    let name = unique_name(app, "Pasted");
    add_png(app, name, &png)
}

/// Download: fetch the typed address on a thread; [`poll`] adds the icon when it arrives.
fn start_download(app: &mut PhotocraftApp) -> Result<(), String> {
    let url = app.ui.app_icon.url.trim().to_string();
    if url.is_empty() {
        return Err("type an image's address first".into());
    }
    let fetch = app.services.download.ok_or("downloads aren't available here")?;
    let slot = Pending::default();
    #[cfg(not(target_arch = "wasm32"))]
    {
        let fill = slot.0.clone();
        let u = url.clone();
        std::thread::Builder::new()
            .name("app-icon-download".into())
            .spawn(move || {
                let r = fetch(&u);
                if let Ok(mut s) = fill.lock() {
                    *s = Some(r);
                }
            })
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_arch = "wasm32")]
    {
        let r = fetch(&url);
        if let Ok(mut s) = slot.0.lock() {
            *s = Some(r);
        }
    }
    app.ui.app_icon.pending = Some((slot, url));
    app.ui.app_icon.error = None;
    Ok(())
}

/// Take a finished download in.
fn poll(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let Some((slot, url)) = app.ui.app_icon.pending.clone() else { return };
    let done = slot.0.lock().ok().and_then(|mut s| s.take());
    match done {
        None => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
        Some(r) => {
            app.ui.app_icon.pending = None;
            if let Err(e) = r.and_then(|bytes| add_image(app, &url, &bytes)) {
                app.ui.app_icon.error = Some(e);
            } else {
                app.ui.app_icon.url.clear();
            }
        }
    }
}

/// Apply the icon in use to the title bar's mark and the window when it changed.
fn apply(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let want = active(app).map_or((String::new(), 0), |i| (i.name.clone(), i.png.len()));
    if app.ui.app_icon.applied.as_ref() == Some(&want) {
        return;
    }
    // At launch the built-in icon is already in place.
    let first = app.ui.app_icon.applied.is_none();
    app.ui.app_icon.applied = Some(want.clone());
    let custom = active(app).and_then(|i| base64::engine::general_purpose::STANDARD.decode(&i.png).ok()).and_then(|png| rgba(&png).ok());
    match custom {
        Some((w, h, px)) => {
            let image = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &px);
            let options = TextureOptions { mipmap_mode: Some(egui::TextureFilter::Linear), ..TextureOptions::LINEAR };
            let tex = ctx.load_texture("photocraft-brand-mark-custom", image, options);
            ctx.data_mut(|d| d.insert_temp(crate::brand::custom_id(), tex));
            ctx.send_viewport_cmd(egui::ViewportCommand::Icon(Some(Arc::new(egui::IconData { rgba: px, width: w, height: h }))));
        }
        None => {
            ctx.data_mut(|d| d.remove::<TextureHandle>(crate::brand::custom_id()));
            if !first && let Ok((w, h, px)) = rgba(BUILT_IN_256) {
                ctx.send_viewport_cmd(egui::ViewportCommand::Icon(Some(Arc::new(egui::IconData { rgba: px, width: w, height: h }))));
            }
        }
    }
}

/// The menu item and the title bar's mark open the window; `use`, `remove` and `add` change the
/// icon without it.
pub fn menu(app: &mut PhotocraftApp, ctx: &egui::Context, id: &str, params: &Value) -> Option<Result<Value, String>> {
    if id != COMMAND {
        return None;
    }
    let o = params.as_object().cloned().unwrap_or_default();
    if let Some(k) = o.keys().find(|k| !["use", "remove", "add"].contains(&k.as_str())) {
        return Some(Err(format!("edit.appIcon has no parameter `{k}`")));
    }
    let r = if let Some(v) = o.get("use") {
        v.as_str().ok_or_else(|| "use must be an icon's name (\"\" for the built-in one)".to_string()).and_then(|n| use_icon(app, n))
    } else if let Some(v) = o.get("remove") {
        v.as_str().ok_or_else(|| "remove must be an icon's name".to_string()).and_then(|n| remove(app, n))
    } else if let Some(v) = o.get("add") {
        let name = v.get("name").and_then(Value::as_str).unwrap_or("Icon");
        match v.get("data").and_then(Value::as_str).map(|d| base64::engine::general_purpose::STANDARD.decode(d.trim())) {
            Some(Ok(bytes)) => add_image(app, name, &bytes),
            Some(Err(e)) => Err(format!("add.data is not base64 ({e})")),
            None => Err("add needs {name, data} (data: the image's bytes in base64)".into()),
        }
    } else {
        app.ui.app_icon.open = true;
        Ok(json!({"appIcon": true}))
    };
    apply(app, ctx);
    Some(r)
}

/// One gallery tile; returns its response.
fn tile(ui: &mut egui::Ui, t: &Tokens, tex: &TextureHandle, selected: bool, name: &str) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(vec2(56.0, 56.0), Sense::click());
    let p = ui.painter();
    let bg = if resp.hovered() { t.field.gamma_multiply(1.4) } else { t.field };
    p.rect_filled(rect, t.radius_sm, bg);
    let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
    p.image(tex.id(), rect.shrink(8.0), uv, Color32::WHITE);
    if selected {
        p.rect_stroke(rect, t.radius_sm, Stroke::new(2.0, t.accent), egui::StrokeKind::Inside);
    }
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, true, selected, name));
    resp.on_hover_text(name)
}

/// The gallery's thumbnails, by (name, PNG length), uploaded once.
fn thumbnail(ctx: &egui::Context, key: &str, png: &[u8]) -> Option<TextureHandle> {
    let id = egui::Id::new(("app-icon-thumb", key, png.len()));
    if let Some(tex) = ctx.data(|d| d.get_temp::<TextureHandle>(id)) {
        return Some(tex);
    }
    let (w, h, px) = rgba(png).ok()?;
    let image = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &px);
    let options = TextureOptions { mipmap_mode: Some(egui::TextureFilter::Linear), ..TextureOptions::LINEAR };
    let tex = ctx.load_texture(format!("app-icon-{key}"), image, options);
    ctx.data_mut(|d| d.insert_temp(id, tex.clone()));
    Some(tex)
}

/// What a click in the window asked for, run once the window is drawn.
enum Action {
    Use(String),
    Remove(String),
    Choose,
    Paste,
    Download,
}

/// Every frame: keep the icon applied, take downloads in, and draw the window while open.
pub fn window(app: &mut PhotocraftApp, ctx: &egui::Context) {
    poll(app, ctx);
    apply(app, ctx);
    if !app.ui.app_icon.open {
        return;
    }
    let t = Tokens::get(ctx);
    let prefs = app.session.prefs().app_icon.clone();
    let active_name = active(app).map(|i| i.name.clone()).unwrap_or_default();
    let can_paste = app.services.clipboard_get_image.is_some();
    let can_download = app.services.download.is_some();
    let downloading = app.ui.app_icon.pending.is_some();
    let builtin = thumbnail(ctx, "", BUILT_IN_256);
    let mut url = app.ui.app_icon.url.clone();
    let mut action = None;
    let mut open = true;
    egui::Window::new(tl!("App Icon")).id(egui::Id::new("app-icon")).open(&mut open).resizable(false).default_width(300.0).show(ctx, |ui| {
        ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
        ui.horizontal_wrapped(|ui| {
            if let Some(tex) = &builtin
                && tile(ui, &t, tex, active_name.is_empty(), tl!("Built-in")).clicked()
            {
                action = Some(Action::Use(String::new()));
            }
            for icon in &prefs.icons {
                let Some(tex) = base64::engine::general_purpose::STANDARD.decode(&icon.png).ok().and_then(|png| thumbnail(ctx, &icon.name, &png)) else {
                    continue;
                };
                let resp = tile(ui, &t, &tex, icon.name == active_name, &icon.name);
                if resp.clicked() {
                    action = Some(Action::Use(icon.name.clone()));
                }
                resp.context_menu(|ui| {
                    if ui.button(tl!("Remove")).clicked() {
                        action = Some(Action::Remove(icon.name.clone()));
                        ui.close();
                    }
                });
            }
        });
        ui.horizontal(|ui| {
            if crate::widgets::secondary_button(ui, tl!("Choose Image…"), 120.0).clicked() {
                action = Some(Action::Choose);
            }
            if can_paste && crate::widgets::secondary_button(ui, tl!("Paste Image"), 100.0).clicked() {
                action = Some(Action::Paste);
            }
        });
        if can_download {
            ui.horizontal(|ui| {
                let edit = egui::TextEdit::singleline(&mut url).hint_text(tl!("Image address (https://…)")).desired_width(190.0);
                let resp = ui.add_enabled(!downloading, edit);
                let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let label = if downloading { tl!("Downloading…") } else { tl!("Download") };
                if ui.add_enabled_ui(!downloading, |ui| crate::widgets::secondary_button(ui, label, 90.0)).inner.clicked() || (enter && !downloading) {
                    action = Some(Action::Download);
                }
            });
        }
        ui.label(egui::RichText::new(tl!("Any picture works; it is fitted into a square. Right-click an icon to remove it.")).small().color(t.text_dim));
        if let Some(e) = &app.ui.app_icon.error {
            ui.label(egui::RichText::new(e).small().color(t.warning));
        }
    });
    app.ui.app_icon.url = url;
    app.ui.app_icon.open = open && app.ui.app_icon.open;
    let r = match action {
        None => return,
        Some(Action::Use(n)) => use_icon(app, &n),
        Some(Action::Remove(n)) => remove(app, &n),
        Some(Action::Choose) => choose(app),
        Some(Action::Paste) => paste(app),
        Some(Action::Download) => start_download(app).map(|()| Value::Null),
    };
    match r {
        Err(e) if e == "cancelled" => {}
        Err(e) => app.ui.app_icon.error = Some(e),
        Ok(_) => {
            if app.ui.app_icon.pending.is_none() {
                app.ui.app_icon.error = None;
            }
        }
    }
    apply(app, ctx);
}

/// `ui.inspect.appIcon`.
pub fn inspect(app: &PhotocraftApp) -> Value {
    let p = &app.session.prefs().app_icon;
    json!({
        "open": app.ui.app_icon.open,
        "active": active(app).map(|i| i.name.clone()).unwrap_or_default(),
        "icons": p.icons.iter().map(|i| i.name.clone()).collect::<Vec<_>>(),
        "downloading": app.ui.app_icon.pending.is_some(),
        "error": app.ui.app_icon.error,
    })
}

#[cfg(test)]
#[path = "app_icon_ui_tests.rs"]
mod tests;
