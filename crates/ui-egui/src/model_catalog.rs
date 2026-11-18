//! The model picker's rows: a page's templates grouped into one row per model, and the version
//! of a model that runs, picked from Generate › Speed and whether reference pictures are
//! attached (ShotoPhopEX).
//!
//! A template belongs to a model and runs at one or more speeds: Full is the model without its
//! turbo LoRA, Balanced the turbo or Lightning version at a good step count, Fast the turbo
//! version at the fewest steps that still give an okay picture. A model that lacks a speed runs
//! at the nearest one it has. Recipes the app picks from context (with references) or that
//! belong to another tool (Remove's adapter, Expand's Outpaint LoRA on the Edit page) are not
//! rows of their own. The data is plain: nothing here talks to the server.

use crate::model_picker::Item;

/// Generate › Speed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Speed {
    Fast,
    Balanced,
    Full,
}

impl Speed {
    pub const ALL: [Speed; 3] = [Speed::Fast, Speed::Balanced, Speed::Full];

    /// From the engine's step tier (`generate_cmds::speed`: `fast`, `balanced`, `best`).
    pub fn from_tier(tier: &str) -> Speed {
        match tier {
            "fast" => Speed::Fast,
            "balanced" => Speed::Balanced,
            _ => Speed::Full,
        }
    }

    /// The value `integrations.speed` stores.
    pub fn pref(self) -> &'static str {
        match self {
            Speed::Fast => "fast",
            Speed::Balanced => "balanced",
            Speed::Full => "full",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Speed::Fast => tl!("Fast"),
            Speed::Balanced => tl!("Balanced"),
            Speed::Full => tl!("Full"),
        }
    }

    /// The speeds to try, this one first, then the nearest.
    fn nearest(self) -> [Speed; 3] {
        match self {
            Speed::Fast => [Speed::Fast, Speed::Balanced, Speed::Full],
            Speed::Balanced => [Speed::Balanced, Speed::Fast, Speed::Full],
            Speed::Full => [Speed::Full, Speed::Balanced, Speed::Fast],
        }
    }
}

const FB: &[Speed] = &[Speed::Fast, Speed::Balanced];
const ALL: &[Speed] = &Speed::ALL;
const FAST: &[Speed] = &[Speed::Fast];
const BALANCED: &[Speed] = &[Speed::Balanced];
const FULL: &[Speed] = &[Speed::Full];

/// What a template is to the picker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The model's version at these speeds.
    Runs(&'static [Speed]),
    /// The same, used when reference pictures are attached (and the only version when the model
    /// has no other on the page: then its row shows only with references).
    WithRefs(&'static [Speed]),
    /// Not a row: another tool's recipe or one kept for scripts.
    Hidden,
}

/// A model: its row in the picker and its preview.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Model {
    pub key: &'static str,
    pub name: &'static str,
    /// Two letters on the glyph.
    pub mark: &'static str,
    pub color: u32,
    /// What it is good at, in a line.
    pub good: &'static str,
    pub tags: &'static [&'static str],
    /// Detail, 1 to 5 (our own read of the 2026-11 tests).
    pub detail: u8,
    /// The Get AI Models pack that brings its files (`models.packs`' extras), when there is one.
    pub pack: Option<&'static str>,
}

const MODELS: &[Model] = &[
    Model {
        key: "krea2",
        name: "Krea 2",
        mark: "Kr",
        color: 0xf97316,
        good: "Most photographic: skin, light and texture",
        tags: &["Photo", "People", "Light"],
        detail: 5,
        pack: Some("standard"),
    },
    Model {
        key: "moss",
        name: "MOSS",
        mark: "Mo",
        color: 0x0d9488,
        good: "All-rounder with clean text and layouts; runs on 12 GB and Macs",
        tags: &["Text in image", "Layouts", "Low memory"],
        detail: 3,
        pack: Some("low-vram"),
    },
    Model {
        key: "qwen-2.1",
        name: "Qwen Image 2.1",
        mark: "Q2",
        color: 0xa855f7,
        good: "Best at edits; up to 9 reference pictures",
        tags: &["Edits", "References", "Transparent"],
        detail: 4,
        pack: Some("standard"),
    },
    Model {
        key: "qwen-edit-2511",
        name: "Qwen Image Edit",
        mark: "QE",
        color: 0x7c3aed,
        good: "Faithful edits and fills; up to 2 references",
        tags: &["Edits", "Fill", "References"],
        detail: 4,
        pack: Some("qwen-edit-2511"),
    },
    Model {
        key: "flux3-klein",
        name: "FLUX 3 Klein",
        mark: "Kl",
        color: 0x0ea5e9,
        good: "Small and quick to load; fits 8 GB",
        tags: &["Low memory", "Quick load"],
        detail: 3,
        pack: Some("flux3-klein"),
    },
    Model {
        key: "p-h3-image",
        name: "H3 Image",
        mark: "H3",
        color: 0xe11d48,
        good: "Very large model, detailed pictures up to 8 MP",
        tags: &["Detail", "Big pictures"],
        detail: 5,
        pack: None,
    },
    Model {
        key: "minimax-h3-image",
        name: "MiniMax H3",
        mark: "MX",
        color: 0x9f1239,
        good: "Largest open model, up to 16 MP",
        tags: &["Up to 16 MP", "References"],
        detail: 5,
        pack: None,
    },
    Model {
        key: "sam3.1",
        name: "SAM 3.1",
        mark: "S",
        color: 0x10b981,
        good: "Selects by text, box or click",
        tags: &["Selection"],
        detail: 4,
        pack: Some("standard"),
    },
    Model {
        key: "vitmatte",
        name: "ViTMatte",
        mark: "V",
        color: 0x06b6d4,
        good: "Refines hair and soft edges",
        tags: &["Edges"],
        detail: 4,
        pack: Some("standard"),
    },
    Model {
        key: "CorridorKey",
        name: "CorridorKey",
        mark: "CK",
        color: 0x22c55e,
        good: "Keys green and blue screens",
        tags: &["Keying"],
        detail: 4,
        pack: None,
    },
    Model {
        key: "qwen3-vl",
        name: "Qwen3-VL",
        mark: "VL",
        color: 0x64748b,
        good: "Writes and checks prompts",
        tags: &["Prompts"],
        detail: 3,
        pack: Some("standard"),
    },
];

/// The model a template belongs to (its key), from the template id, else its family.
pub fn model_key(id: &str, family: &str) -> &'static str {
    let head = id.split('/').next().unwrap_or("");
    let key = match head {
        "krea2-turbo" | "krea2" | "krea2-raw" => "krea2",
        "moss-image" => "moss",
        "qwen-2.1" => "qwen-2.1",
        "qwen-edit-2511" => "qwen-edit-2511",
        h if h.starts_with("flux3-klein") => "flux3-klein",
        "p-h3-image" => "p-h3-image",
        "minimax-h3-image" => "minimax-h3-image",
        h if h.starts_with("sam3") => "sam3.1",
        _ => "",
    };
    if !key.is_empty() {
        return key;
    }
    match family {
        "krea2" => "krea2",
        "moss-image" => "moss",
        "qwen-image-2.1" => "qwen-2.1",
        "qwen-image-edit-2511" => "qwen-edit-2511",
        f if f.starts_with("flux") => "flux3-klein",
        "p-h3-image" => "p-h3-image",
        "minimax-h3-image" => "minimax-h3-image",
        "sam3.1" => "sam3.1",
        "vitmatte" => "vitmatte",
        "CorridorKey" => "CorridorKey",
        "qwen3-vl" => "qwen3-vl",
        _ => "other",
    }
}

/// A model's row data; an unknown key gets a plain one named after the template.
pub fn model(key: &str) -> Model {
    MODELS.iter().find(|m| m.key == key).copied().unwrap_or(Model {
        key: "other",
        name: "Other",
        mark: "•",
        color: 0x64748b,
        good: "",
        tags: &[],
        detail: 3,
        pack: None,
    })
}

/// What template `id` is to the picker on `page`.
pub fn role(id: &str, page: &str) -> Role {
    let tail = id.split_once('/').map_or("", |(_, t)| t);
    match model_key(id, "") {
        "krea2" => match tail {
            "image-style" => Role::WithRefs(FB),
            // Krea 2 Raw at 30 to 40 steps, the version without the turbo LoRA (to come).
            t if t.starts_with("raw") || id.starts_with("krea2-raw") => Role::Runs(FULL),
            _ => Role::Runs(FB),
        },
        "moss" => match tail {
            "turbo" | "edit-turbo" => Role::Runs(FB),
            "image" | "edit" => Role::Runs(FULL),
            // The base before fine-tuning: kept for scripts.
            _ => Role::Hidden,
        },
        "qwen-edit-2511" => match tail {
            "fill-lightning-4" => Role::Runs(FAST),
            "fill-lightning-8" => Role::Runs(BALANCED),
            "edit-lightning-8" | "expand-lightning-8" => Role::Runs(FB),
            "fill" | "edit" | "expand" => Role::Runs(FULL),
            "fill-ref-lightning-8" => Role::WithRefs(FB),
            "fill-ref" => Role::WithRefs(FULL),
            "image-ref" => Role::WithRefs(ALL),
            // Fill's mask-guided recipe and Remove's adapter belong to those tools.
            "fill-guided" | "remove-lightning-8" => Role::Hidden,
            _ => Role::Runs(ALL),
        },
        "qwen-2.1" => match tail {
            "image-ref" => Role::WithRefs(ALL),
            // The Outpaint LoRA's 6-step recipe: Expand's Fast; on the Edit page it is Expand's.
            "outpaint-turbo" if page == "expand" => Role::Runs(FAST),
            "outpaint-turbo" => Role::Hidden,
            // The image template follows Speed's step tiers in the engine (25 or 40 steps).
            "image" => Role::Runs(ALL),
            _ => Role::Runs(FULL),
        },
        "flux3-klein" => Role::Runs(BALANCED),
        "p-h3-image" | "minimax-h3-image" => Role::Runs(FULL),
        _ => Role::Runs(ALL),
    }
}

/// The speeds a template runs at, whatever its role (none when hidden).
fn speeds(role: Role) -> &'static [Speed] {
    match role {
        Role::Runs(s) | Role::WithRefs(s) => s,
        Role::Hidden => &[],
    }
}

/// One row of the picker: a model and the indices of its templates in the items.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub model: Model,
    /// What the row says: the model's name, or on a tool's page with several templates of one
    /// model (Select's text, box and click recipes) the template's short name.
    pub label: String,
    pub templates: Vec<usize>,
}

/// The Generate pages, where a row is a model and Speed picks its version. On the other pages
/// (a tool's own dialog) each template is its own row.
pub fn per_model(page: &str) -> bool {
    matches!(page, "image" | "fill" | "edit" | "expand")
}

/// Where a model sorts on `page`: the picks for that kind of work first (Krea 2 and MOSS make
/// new pictures, Qwen 2.1 edits best: Cody, 2026-11-17).
fn rank(key: &str, page: &str) -> usize {
    const MAKE: &[&str] = &["krea2", "moss", "flux3-klein", "qwen-2.1", "qwen-edit-2511", "p-h3-image", "minimax-h3-image"];
    const CHANGE: &[&str] = &["qwen-2.1", "qwen-edit-2511", "krea2", "flux3-klein", "moss", "p-h3-image", "minimax-h3-image"];
    let order = if page == "image" { MAKE } else { CHANGE };
    order.iter().position(|k| *k == key).unwrap_or(order.len())
}

/// The picker's rows for `items` (one page's templates): one per model, its hidden recipes left
/// out, a model whose only templates take references listed only when `refs`.
pub fn rows(items: &[Item], page: &str, refs: bool) -> Vec<Row> {
    let mut out: Vec<Row> = Vec::new();
    if !per_model(page) {
        for (i, it) in items.iter().enumerate() {
            let m = model(model_key(&it.id, &it.family));
            let shared = items.iter().filter(|o| model_key(&o.id, &o.family) == m.key).count() > 1;
            let label = if shared || m.key == "other" { crate::model_picker::short_name(&it.name).to_string() } else { m.name.to_string() };
            out.push(Row { model: m, label, templates: vec![i] });
        }
        out.sort_by_key(|row| rank(row.model.key, page));
        return out;
    }
    for (i, it) in items.iter().enumerate() {
        let r = role(&it.id, page);
        if r == Role::Hidden {
            continue;
        }
        let key = model_key(&it.id, &it.family);
        match out.iter_mut().find(|row| row.model.key == model(key).key) {
            Some(row) => row.templates.push(i),
            None => out.push(Row { model: model(key), label: model(key).name.to_string(), templates: vec![i] }),
        }
    }
    out.retain(|row| refs || row.templates.iter().any(|&i| matches!(role(&items[i].id, page), Role::Runs(_))));
    out.sort_by_key(|row| rank(row.model.key, page));
    out
}

/// The template of `row` that runs at `speed` (or the nearest speed it has), the references
/// version when `refs` and the model has one.
pub fn resolve<'a>(items: &'a [Item], row: &Row, page: &str, speed: Speed, refs: bool) -> Option<&'a Item> {
    if !per_model(page) {
        return row.templates.first().map(|&i| &items[i]);
    }
    let pick = |want_refs: bool| {
        speed.nearest().into_iter().find_map(|s| {
            row.templates.iter().map(|&i| &items[i]).find(|it| {
                let r = role(&it.id, page);
                matches!(r, Role::WithRefs(_)) == want_refs && speeds(r).contains(&s)
            })
        })
    };
    if refs && let Some(t) = pick(true) {
        return Some(t);
    }
    pick(false).or_else(|| pick(true))
}

/// The speeds `row` offers (without references), for the dots and Speed's hint.
pub fn row_speeds(items: &[Item], row: &Row, page: &str) -> Vec<Speed> {
    Speed::ALL.into_iter().filter(|s| row.templates.iter().any(|&i| matches!(role(&items[i].id, page), Role::Runs(v) if v.contains(s)))).collect()
}

/// The template that runs when `chosen` (a template id the picker or a script stored) is the
/// choice on `page` at `speed`: its model's version for that speed and the references. A
/// template that is not a picker row (a hidden recipe a script named) runs as named.
pub fn remap(items: &[Item], page: &str, chosen: &str, speed: Speed, refs: bool) -> Option<String> {
    let it = items.iter().find(|i| i.id == chosen)?;
    if !per_model(page) || role(chosen, page) == Role::Hidden {
        return Some(chosen.to_string());
    }
    let key = model_key(&it.id, &it.family);
    let row = rows(items, page, true).into_iter().find(|r| r.model.key == key)?;
    resolve(items, &row, page, speed, refs).map(|t| t.id.clone())
}

#[cfg(test)]
#[path = "model_catalog_tests.rs"]
mod tests;
