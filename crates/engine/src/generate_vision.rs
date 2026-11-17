//! Asking the vision-language model: prompt enhancement, captions, naming things and the
//! yes/no judges that check a result.

use std::collections::BTreeMap;

use photocraft_genai::{GenerativeBackend, Gray8, Request, Rgba8};
use photocraft_geom::Rect;
use serde_json::Value;

use super::{JobProgress, MAX_SEED, check_rewrite, is_rewrite, request_size_within, resize_rgba8, rgba8_at, side_by_side};

pub(crate) const ENHANCE: &str = "generate.enhancePrompt";

pub(super) const ENHANCE_EDIT_TEMPLATE: &str = "enhance/edit";

pub(super) const ENHANCE_IMAGE_TEMPLATE: &str = "enhance/image";

/// The system turn before an edit or fill instruction. Short on purpose: the 4B model echoes
/// or rambles under a long system prompt. The rules are the ones that recur in the
/// Qwen-Image-Edit guides: lead with the verb, name things instead of "this"/"it", say what the
/// result looks like, say what stays; plus the case the first user session hit, a change of
/// medium on a doodle, where "keep the lines" would defeat the instruction.
pub(super) const ENHANCE_EDIT_SYSTEM: &str = "You rewrite instructions for an image-editing AI. You can see the image.\n\
Rewrite the user's instruction as one or two clear sentences, at most 70 words, that:\n\
- start with the action verb (Change, Replace, Add, Remove, Turn, Make, Give),\n\
- name things by what they are in the image instead of \"this\", \"it\" or \"here\" (\"the hand-drawn smiley face\", \"the white lighthouse\", \"the woman in the red coat\"),\n\
- say concretely what the result looks like: for something added, what it is, its colour and material, where it sits and how big, lit like the rest of the scene; for something removed, what shows where it was; for something replaced, what replaces it and that it keeps the old one's place and size,\n\
- for a person, say that their face, identity, hair and pose stay the same unless the user asks to change them,\n\
- end by naming the things you see in this image that must stay as they are, not a stock phrase; when the change is to the whole picture (night, a season, a painting style), say that the layout, objects and framing stay, not that any object stays unchanged.\n\
Words to be written in the image go in double quotes, exactly as the user gave them (Replace the text \"SALE\" with \"OPEN\").\n\
When the instruction changes the style, medium or realism of the picture (\"make this realistic\", \"turn it into a painting\"), name the real-world subject the result shows and the visible traits of the new style, and say that the drawing's strokes are replaced (a doodle of a face made realistic becomes \"a photograph of a real human face with the same expression and pose, replacing the black outline strokes with real skin, hair and facial features\"); never ask to keep the drawing's lines.\n\
Keep the user's intent and every thing, colour and number they named. Add nothing they did not ask for.\n\
Example of the form only (never reuse its words): \"give the statue a scarf\" on a photo of a bronze statue in a park becomes \"Wrap a knitted red wool scarf around the neck of the bronze statue, draped over its shoulders and lit by the same daylight, keeping the statue's pose, the plinth and the park behind it as they are.\"\n\
Write in English even when the user wrote in another language, and keep words to be written in the image as given.\n\
Reply with the rewritten instruction only: no explanation, no quotes around it, no markdown.";

/// Added to the system turn for a fill: where the selection is and what the instruction means
/// there.
pub(super) const ENHANCE_FILL_NOTE: &str = "The user selected the {place} of the image; the instruction says what should be generated inside that selection so that it blends with its surroundings. Describe what appears there.";

/// The system turn's note for a removal, whose picture is two views side by side: the
/// surroundings on the left, a close-up of the painted area on the right ([`side_by_side`]).
/// The rewriter has no other view of what was painted: with the fill note and one crop it
/// named what lay under the stroke ("the window" for a stroke on a car, "the dog's head",
/// "the hand" for a ring) or the most salient thing in view ("the pendant" for a stroke on
/// the chain); with the painted area outlined in green it talked about the outline ("the
/// green circle", "the green-painted face"). Measured 2026-10-21 on the robustness set.
/// A selection as big as the close-up's frame left both views the same picture and the namer
/// took the most salient thing (a label for a woman selected whole, 2026-11-16), so the close-up
/// darkens everything outside the selection ([`dim_outside`]), which changes no colour of the object.
pub(super) const ENHANCE_REMOVE_NOTE: &str = "You can see two views of the same photo side by side: on the right, a close-up in which the object to remove (or part of it) keeps its colours and everything else is darkened; on the left, the same with its surroundings, nothing darkened. The object is the thing that is not darkened on the right. Reply with exactly one sentence of the form \"Remove the <object>.\" The object is the whole thing, not a part of it (the whole car, not its window; the whole dog, not its head; the whole necklace, not its pendant; the whole cup, not the coffee in it), named by what it is and its colour in at most five words; never call it an area, region, part or selection. It is the object itself, not the person, animal or larger thing that wears, holds or carries it (the glasses, not the man wearing them; the mask, not the face; the ring, not the hand; the umbrella, not the person under it). Say nothing else: not what is behind it, not the two views, not other things in the picture.";

/// The system turn for judging two results of a removal side by side ([`judge_pair`]): the
/// edge-match score preferred a smeared skin-coloured blob over a man's eyes to the eyes
/// themselves (its colour and smoothness matched the cheeks), where a look at the two tells
/// them apart at once (measured 2026-10-21).
pub(super) const ENHANCE_JUDGE_SYSTEM: &str = "You can see two versions of the same part of a photo side by side, left and right, after an attempt to remove something from it; the question names what was to be removed. A version where that thing is still there, whole or in part, loses. A version with a flaw where it was also loses: a smear, a blur, a blocky or pixelated patch, a flat panel, an area of flat red or other paint, a wrong texture, a duplicated line or object, or something that does not belong. Pick the version where the thing is gone and the scene looks like a natural, untouched photograph.\n\
Reply with exactly one sentence: \"The left one.\" or \"The right one.\" Nothing else.";

/// What the judge is asked, with the object's name when the rewriter found one ("left"
/// before "right", so a parroted question picks the left). Asked only which looked natural,
/// the judge kept a crow on its wire and a boat on its lake: both look natural (measured
/// 2026-10-22).
pub(super) fn judge_question(object: Option<&str>) -> String {
    match object {
        Some(name) => format!("The {name} was to be removed. In which one is the {name} gone and the scene untouched, left or right?"),
        None => "Something was to be removed. In which one is it gone and the scene untouched, left or right?".to_string(),
    }
}

/// The system turn for asking whether the removed thing is still in one result
/// ([`still_there`]): a yes or no on one picture, asked of each result before the two are
/// judged side by side. The side-by-side judge kept transparent-framed glasses it was told
/// to remove, where the direct question on the one picture answers plainly (2026-10-22).
pub(super) const ENHANCE_PRESENCE_SYSTEM: &str = "You can see one photo. The question names a thing. Say whether that thing, or any part of it, is visible in the photo: look carefully for thin, faint, transparent or partly hidden ones.\n\
Reply with exactly one word: Yes or No.";

/// The system turn for a fill from a reference image. The rewriter sees the picture (its
/// selection flattened to grey) and the reference side by side, as AusBoss's Krea 2 clothing
/// workflow shows its prompt writer the pair, and names what to take exactly as the reference
/// shows it; the template's own wrapper then says where it goes and what stays.
pub(super) const ENHANCE_FILL_REF_SYSTEM: &str = "You can see two photos side by side. The left photo is being edited: its flat grey area will be repainted. The right photo is a reference. The user names what to take from the reference.\n\
Reply with one noun phrase of at most 50 words naming that thing and describing it exactly as the right photo shows it: for clothing its garment type, colour, material, knit or weave, pattern, collar, buttons, length and fit; for anything else its shape, colour, material and details. Describe it as it will look on the left photo's subject, in that pose. Keep the user's words where they are specific.\n\
Never mention grey, blank, masked, left, right, photo, image or reference. Reply with the phrase only: no sentence, no quotes, no markdown.";

/// The system turn before a text-to-image idea (the shape of the official Krea 2 template's
/// expander and of Qwen's rewriter, in a few lines).
pub(super) const ENHANCE_IMAGE_SYSTEM: &str = "You are an expert prompt writer for a text-to-image model. Expand the user's idea into one paragraph of 60 to 120 words that the model can draw: the main subject and what it is doing first, then what can be seen of it (shape, materials, surfaces, clothing, expression), its pose, how it is framed (close-up, medium shot, wide shot, from which angle), the setting around it, and the light (where it comes from, its colour, the shadows it casts). End with the medium.\n\
Faithfulness first: keep every subject, count, action, colour, position and spatial relation the user gave, and any framing they named; a portrait is a head-and-shoulders close-up, so describe only the face, hair, neck and shoulders and what is behind them, never a chair, hands, a held object or the body below the shoulders. If the user named a medium or style (oil painting, watercolour, sketch, 3D render, anime), end with that medium in their words and never call it a photograph; only if they named none, end with \"A photograph\" and a few words on its look. Never add a person, animal or main object the user did not ask for; small details that belong in the setting are fine. Write things that can be seen instead of mood words and praise. Put any words to be written in the image in double quotes, exactly as given, and say where they sit. If the idea is already detailed, polish it lightly and keep its wording. Write in English even when the user wrote in another language, translating their words faithfully (a breed, a place or a dish keeps its name, such as Shiba Inu), and keep words to be written in the image as given.\n\
Reply with the paragraph only: no title, no bullets, no quotes around it, no markdown, no notes, nothing about these rules.";

/// The chat's settings for one try: a retry after a refused rewrite samples warmer (0.7, then
/// 1.0), since at the worker's 0.3 the next seed often wrote the refused answer again word for
/// word and the third try was wasted (round 5 on the 5090, 2026-11-17).
fn retry_params(params: &BTreeMap<String, Value>, rewrite: bool, attempt: u64) -> BTreeMap<String, Value> {
    let mut p = params.clone();
    if rewrite && attempt > 0 {
        p.insert("temperature".to_string(), Value::from(if attempt == 1 { 0.7 } else { 1.0 }));
    }
    p
}

/// Whether a prompt asks for a medium other than a photograph (a painting, a drawing, a
/// render…): Best's photo rewriter turned "an oil painting of a lighthouse" into a photograph
/// of one (round 3, 2026-11-12), so such prompts get the plain rewriter.
pub fn names_other_medium(prompt: &str) -> bool {
    const MEDIA: &[&str] = &[
        "painting",
        "painted",
        "watercolor",
        "watercolour",
        "gouache",
        "acrylic",
        "oil on canvas",
        "illustration",
        "illustrated",
        "drawing",
        "drawn",
        "sketch",
        "cartoon",
        "anime",
        "manga",
        "comic",
        "3d render",
        "render of",
        "rendered",
        "cgi",
        "pixel art",
        "vector",
        "line art",
        "concept art",
        "digital art",
        "low poly",
        "claymation",
        "woodcut",
        "linocut",
        "engraving",
        "pastel",
        "charcoal",
        "pencil",
        "ink wash",
        "mural",
        "fresco",
        "stained glass",
        "poster",
        "logo",
        "icon",
        "clip art",
        "sticker",
        "isometric",
        "cel shaded",
        "cel-shaded",
        "voxel",
    ];
    let p = prompt.to_lowercase();
    MEDIA.iter().any(|m| p.contains(m))
}

/// The photographic rewriter ([`EnhanceKind::ImagePhoto`], `enhanceStyle: "photo"`): how the most
/// liked photoreal Krea 2 / MOSS pictures on Civitai are prompted (their generation data,
/// 2026-11-12): plain prose that opens with the kind of shot and what took it, one named light
/// source, and the small flaws of a real photo, without "photorealistic", "8k" or camera brands.
pub(super) const ENHANCE_IMAGE_PHOTO_SYSTEM: &str = "You are an expert prompt writer for a photorealistic text-to-image model. Rewrite the user's idea as one paragraph of 60 to 110 words that reads like the caption of a real, unretouched photograph.\n\
Always open with the kind of shot, chosen for the scene: a portrait on an 85mm lens for a person, a close-up on a macro lens for a small object, a wide shot on a 24mm lens for a room or landscape, a candid photo on a 35mm lens for people busy with something, a snapshot on a 50mm lens for a pet. If the user names a lens, film or camera setting, keep it word for word. Then the subject and what it does, the materials and surfaces, the setting the user named, and one concrete light source (window light, overcast daylight, a lamp, a softbox, on-camera flash) with how it falls.\n\
Name two or three small imperfections a real photo has, each on something the user named that really has it: pores and fine lines on a person's face; fur on an animal; scratches, chips or dust on an object; creases on fabric; crumbs on food. Shallow focus and soft grain apply to the whole frame.\n\
Describe only what the user asked for. Keep every subject, action, colour and spatial relation they gave. Add no new people, animals, objects, plants, weather, accessories or clothing; where the user names no clothing, say nothing about clothing. Put any words to be rendered in the image in double quotes. Never write photorealistic, hyperrealistic, 8k, 4k, masterpiece, best quality, DSLR or a camera brand.\n\
Write in English even when the user wrote in another language, translating their words faithfully (a breed, a place or a dish keeps its name, such as Shiba Inu), and keep words to be written in the image as given. Never mention these instructions or anything that is absent from the scene. Reply with the paragraph only: no title, no bullets, no quotes around it, no markdown.";

/// The longest text-to-image rewrite, in tokens (about 160 words).
pub(super) const ENHANCE_IMAGE_MAX_TOKENS: u64 = 220;

/// The rewriter of an Ultra picture ([`EnhanceKind::ImageUltra`]): the 16 MP model is trained
/// on long, exact captions and has the pixels for small things, so the caption plans every
/// part of the frame, and a designed page (a poster, a sheet, a diagram) gets its whole layout
/// and every word spelled out.
pub(super) const ENHANCE_IMAGE_ULTRA_SYSTEM: &str = "You write the caption a very high resolution text-to-image model draws from. Rewrite the user's idea as one fluent English paragraph of 150 to 300 words in which every part of the picture is planned and can be inspected up close.\n\
First decide what the picture is. A scene or photograph: say who or what is in it and how many, the pose and which way each subject faces the camera, what touches what, the scale, the place, the light and where it comes from, the lens and distance, then the fine detail a close look would find: skin, hair, fabric weave, wear, dust, reflections. A designed page such as a poster, cover, label, character sheet, product board, infographic, diagram or screen: lay it out like a designer, naming each area of the page, what it shows, and how the areas are ordered, aligned and separated; leave no area blank unless the user asks for empty space.\n\
Keep every fact the user gave: subjects, counts, actions, colours, positions, style and medium. Do not add people, animals or main objects they did not ask for. Write every word that must appear in the picture exactly, in double quotes, and say where it sits and how it is lettered; never invent names of venues, people, brands or sponsors, dates, prices, addresses or contact details; where a designed page needs more wording than the user gave, repeat or rearrange the user's own words. Replace mood words and praise with things that can be seen. Never write masterpiece, best quality, 8k, 4k or hyperrealistic.\n\
Reply with the paragraph only: no title, no bullets, no JSON, no quotes around it, no markdown.";

/// The longest Ultra rewrite, in tokens (about 450 words).
pub(super) const ENHANCE_ULTRA_MAX_TOKENS: u64 = 640;

/// Which rewriter an `enhance` runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EnhanceKind {
    /// A fill: the picture, the selection's place, and what to generate there.
    Fill,
    /// A fill from a reference: the picture (its selection flattened to grey) and the reference
    /// side by side, and what to take from the reference.
    FillRef,
    /// A removal: the surroundings and a close-up of the painted part side by side; name the
    /// whole object the part belongs to.
    Remove,
    /// Two results of a removal side by side: which looks like an untouched photograph.
    Judge,
    /// One result of a removal: is the named thing still in it.
    Presence,
    /// An edit: the picture and an instruction about it.
    Edit,
    /// Text to image: no picture.
    Image,
    /// A text-to-image idea written the way real photos are captioned ([`ENHANCE_IMAGE_PHOTO_SYSTEM`]).
    ImagePhoto,
    /// A text-to-image idea planned in full for the 16 MP model ([`ENHANCE_IMAGE_ULTRA_SYSTEM`]).
    ImageUltra,
    /// A picture: name its main subject (Remove Background with no prompt).
    Subject,
    /// A picture: describe its background alone (Generative Expand with no prompt).
    Backdrop,
    /// A picture: which of its edges cut its main subject off (Generative Expand with no prompt).
    CutEdges,
    /// A picture: a caption of all of it, subject and setting (Generative Expand through the
    /// Outpaint LoRA with no prompt: its `Scene:` line).
    Scene,
    /// A fill on Krea 2 AnyPaint: the crop with the selection flattened to grey; describe the
    /// whole finished picture with the user's words in the grey area.
    FillScene,
    /// A picture: list the separate things in its foreground that are not people or animals
    /// (Split into Layers with no prompt, after the people).
    Things,
}

/// The worker's chat `encoder` that answers on Krea 2's Qwen3-VL-4B (genai-native
/// `CHAT_ON_KREA2`).
pub(super) const KREA2_CHAT_ENCODER: &str = "krea2";

/// A chat's `encoder` that names a MOSS template answers on that template's own Qwen3.5-9B.
pub(super) const MOSS_CHAT_PREFIX: &str = "moss-image/";

/// The same for FLUX 3 Klein, whose encoder is Qwen3-VL-4B (Klein + rewrite went from 16.4 s
/// to the picture's own 5 s plus the chat, 2026-11-12).
pub(super) const KLEIN_CHAT_PREFIX: &str = "flux3-klein/";

/// The same for MiniMax-H3-Image, whose prompt writer is Qwen3.5-9B.
pub(super) const MINIMAX_CHAT_PREFIX: &str = "minimax-h3-image/";

/// The system turn of [`EnhanceKind::FillScene`]: AusBoss's "Write the prompt" step of the
/// Krea 2 Inpaint Masked workflow (v1.3), which the AnyPaint LoRA follows best: it was trained
/// on descriptions of the complete output picture, not of the patch.
pub(super) const ENHANCE_FILL_SCENE_SYSTEM: &str = "You see a picture with a flat mid-grey area that will be painted. Write one text-to-image prompt describing this picture as it will look when finished: one paragraph of 40 to 80 words covering the subject, setting, materials, lighting and camera style, with the grey area showing what the user asks for. Never call it grey, blank, masked, selected or painted, and describe it as part of the finished scene. Keep the user's words where they are specific. Reply with the prompt only: no title, no quotes, no markdown.";

/// The system turn of [`EnhanceKind::Subject`]: a short noun phrase the matte can follow.
pub(super) const ENHANCE_SUBJECT_SYSTEM: &str = "You see a picture. Reply with one noun phrase of three to eight words naming its main subject as the picture shows it, for example \"the man in a white t-shirt\" or \"the red vintage car\". Reply with the phrase only: no sentence, no quotes, no full stop.";

/// The user turn of [`EnhanceKind::Subject`].
pub(super) const ENHANCE_SUBJECT_REQUEST: &str = "Name the main subject of this picture.";

/// The system turn of [`EnhanceKind::Backdrop`]: what Generative Expand continues into the new
/// canvas. Told only to "continue the scene", Qwen-Image-2.1 and 2511 both painted another
/// copy of the person into the added area; named as a backdrop with no people, 2511 continued
/// it cleanly (Cody's portrait, in the app, 2026-11-03).
pub(super) const ENHANCE_BACKDROP_SYSTEM: &str = "You see a picture. Reply with one noun phrase of four to twelve words describing only its background and setting, leaving out every person, animal and main subject, for example \"a plain grey studio wall\" or \"an orange curtain backdrop above a red carpet\". Reply with the phrase only: no sentence, no quotes, no full stop.";

/// The system turn of [`EnhanceKind::CutEdges`].
pub(super) const ENHANCE_CUT_SYSTEM: &str = "You see a picture. Say which of its edges cut off its main subject, a person, animal or object whose body runs past the frame (for example a portrait cropped at the chest is cut off at the bottom). Reply in the form \"cut off at: bottom, left\" naming the edges among top, bottom, left and right, or reply \"not cut off at all\" when the whole subject is inside the picture. Reply with that line only.";

/// The user turn of [`EnhanceKind::CutEdges`].
pub(super) const ENHANCE_CUT_REQUEST: &str = "Which edges of this picture cut off its main subject?";

/// The system turn of [`EnhanceKind::Scene`]: the Outpaint LoRA was trained with a Qwen3-VL-8B
/// description of the whole picture after its instruction (AusBoss's model card, read
/// 2026-11-03), so the caption names the subject as well as the setting; never who a person
/// is (it named Cody's sitter, which goes into the prompt and the layer's info).
/// The system turn of [`EnhanceKind::Things`]; `{k}` is the most things asked for.
pub(super) const ENHANCE_THINGS_SYSTEM: &str = "You see a picture. List the separate physical things in its foreground that a designer would put on their own layers and that are not people or animals (for example a table, a cup, a bicycle, a vase of flowers), nearest to the camera first, as short noun phrases of one to three words each, separated by commas, at most {k} of them. Leave out walls, floors, the sky, the ground, and anything a person wears or holds. Reply with the list only: no sentence, no numbers, no full stop.";

pub(super) const ENHANCE_SCENE_SYSTEM: &str = "You see a picture. Describe all of it in one or two plain sentences, as a caption: the main subject (who or what, pose, clothing, how much of them the frame shows) and the setting, background, lighting and style. Never name or identify a person, even a famous one: describe people only by how they look. Reply with the description only.";

/// The user turn of [`EnhanceKind::Scene`].
pub(super) const ENHANCE_SCENE_REQUEST: &str = "Describe this picture.";

/// The seed of an Expand's scene caption: fixed, so the same picture gets the same caption
/// whatever the expand's own seed, and a re-roll reuses the encoded prompt and references
/// instead of paying a new encode (the in-app benchmark, 2026-11-03: every new seed missed).
pub(super) const EXPAND_CAPTION_SEED: u64 = 0x5EED_CA95;

/// The last Expand caption, keyed on the picture it describes: a re-roll of the same picture
/// skips the chat (about 4 s) and the text encoder's reload for it (about 2.8 s), which the
/// fixed [`EXPAND_CAPTION_SEED`] would give back word for word anyway.
pub(super) static EXPAND_CAPTION: std::sync::Mutex<Option<(u64, String)>> = std::sync::Mutex::new(None);

/// The key of [`EXPAND_CAPTION`]: the described picture's size and pixels.
pub(super) fn caption_key(picture: &Rgba8) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (picture.width, picture.height).hash(&mut h);
    picture.data.hash(&mut h);
    h.finish()
}

/// The user turn of [`EnhanceKind::Backdrop`].
pub(super) const ENHANCE_BACKDROP_REQUEST: &str = "Describe the background of this picture.";

/// Where a region sits on a canvas, in words ("upper-left part", "centre", "right part").
pub(crate) fn place_in_words(region: Rect, canvas: Rect) -> String {
    let (cw, ch) = (f64::from(canvas.width().max(1)), f64::from(canvas.height().max(1)));
    let cx = (f64::from(region.x0 + region.x1) / 2.0 - f64::from(canvas.x0)) / cw;
    let cy = (f64::from(region.y0 + region.y1) / 2.0 - f64::from(canvas.y0)) / ch;
    let big = !canvas.is_empty() && f64::from(region.width()) * f64::from(region.height()) > 0.6 * cw * ch;
    if big {
        return "whole".to_string();
    }
    let v = if cy < 1.0 / 3.0 {
        "upper"
    } else if cy > 2.0 / 3.0 {
        "lower"
    } else {
        ""
    };
    let h = if cx < 1.0 / 3.0 {
        "left"
    } else if cx > 2.0 / 3.0 {
        "right"
    } else {
        ""
    };
    match (v, h) {
        ("", "") => "centre".to_string(),
        ("", h) => format!("{h} part"),
        (v, "") => format!("{v} part"),
        (v, h) => format!("{v}-{h} part"),
    }
}

/// The rewriter's answer, cleaned: the first non-empty text without the chat template's
/// leading "assistant" echo, surrounding quotes or markdown emphasis; `None` when the model
/// said nothing usable.
#[cfg(test)]
pub(crate) fn clean_enhanced(texts: &[String]) -> Option<String> {
    clean_answer(texts, 3)
}

/// [`clean_enhanced`] for an answer of at least `min_words` words: a Yes or No, or one thing's
/// name, is one word, where a rewritten prompt of fewer than three is a failed one.
pub(super) fn clean_answer(texts: &[String], min_words: usize) -> Option<String> {
    for t in texts {
        let mut s = t.trim();
        for prefix in ["assistant:", "Assistant:", "assistant", "Assistant", "**Final Prompt:**", "Final Prompt:", "Rewritten:", "Instruction:"] {
            if let Some(rest) = s.strip_prefix(prefix) {
                s = rest.trim();
            }
        }
        let s = s.trim_matches(|c: char| c == '"' || c == '*' || c == '`' || c == '\u{201c}' || c == '\u{201d}').trim();
        if s.split_whitespace().count() >= min_words.max(1) {
            return Some(s.to_string());
        }
    }
    None
}

/// Rewrite `prompt` with the vision-language model: `Ok(Some)` the rewritten prompt, `Ok(None)`
/// when the model said nothing usable in two tries (consecutive seeds), `Err` when the server
/// could not run it (no encoder file, down). An edit or fill is rewritten with the picture in
/// view when one is given, blind otherwise. The rewriter goes straight to the backend: its
/// small encoder is not a model switch worth a purge, and it is the file Krea 2 loads anyway.
/// The things the vision-language model lists in `picture`'s foreground (at most `k`, nearest
/// first, [`EnhanceKind::Things`]), as short phrases with no leading article; empty when it
/// could not run or said nothing usable.
pub(crate) fn list_things(backend: &dyn GenerativeBackend, picture: &Rgba8, k: usize, seed: u64, progress: &JobProgress) -> Vec<String> {
    let k = k.clamp(1, 12);
    let said = enhance_prompt(backend, EnhanceKind::Things, "List the things.", Some(picture), Some(&k.to_string()), seed, progress).ok().flatten();
    let Some(said) = said else { return Vec::new() };
    let mut out: Vec<String> = Vec::new();
    for raw in said.split([',', '\n', ';']) {
        let t = raw.trim().trim_matches(|c: char| c == '.' || c == '"' || c == '-' || c == '*').trim();
        let t = t.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')' || c == ' ');
        let t = t.strip_prefix("the ").or_else(|| t.strip_prefix("a ")).or_else(|| t.strip_prefix("an ")).unwrap_or(t).trim();
        let words = t.split_whitespace().count();
        // "None": nothing to list (one-word answers count since 2026-11-12).
        let nothing = ["none", "nothing", "n/a", "no", "unknown"].iter().any(|n| t.eq_ignore_ascii_case(n));
        if t.is_empty() || nothing || words > 4 || t.chars().count() > 40 || out.iter().any(|o| o.eq_ignore_ascii_case(t)) {
            continue;
        }
        out.push(t.to_lowercase());
        if out.len() >= k {
            break;
        }
    }
    out
}

/// What a prompt writer that reads text only says when shown a picture (genai-native's
/// gn-diffusion on a Mac: MOSS's Qwen3.5-9B on MLX, its vision tower not ported).
pub(super) const TEXT_ONLY: &str = "reads text only";

/// How a Krea 2 AnyPaint fill's words reach its rewriter ([`EnhanceKind::FillScene`]).
pub(crate) const FILL_SCENE_LEAD: &str = "What the grey area shows:";

pub(crate) fn enhance_prompt(
    backend: &dyn GenerativeBackend,
    kind: EnhanceKind,
    prompt: &str,
    image: Option<&Rgba8>,
    place: Option<&str>,
    seed: u64,
    progress: &JobProgress,
) -> std::result::Result<Option<String>, photocraft_genai::Error> {
    match enhance_once(backend, kind, prompt, image, place, seed, progress) {
        // Refused for its picture: an edit or a fill is rewritten from its words alone; the
        // kinds that must see the picture stay refused.
        Err(photocraft_genai::Error::Server(why)) if why.contains(TEXT_ONLY) && image.is_some() && matches!(kind, EnhanceKind::Edit | EnhanceKind::Fill) => {
            enhance_once(backend, kind, prompt, None, place, seed, progress)
        }
        r => r,
    }
}

pub(super) fn enhance_once(
    backend: &dyn GenerativeBackend,
    kind: EnhanceKind,
    prompt: &str,
    image: Option<&Rgba8>,
    place: Option<&str>,
    seed: u64,
    progress: &JobProgress,
) -> std::result::Result<Option<String>, photocraft_genai::Error> {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return Ok(None);
    }
    let image = match kind {
        EnhanceKind::Image | EnhanceKind::ImagePhoto | EnhanceKind::ImageUltra => None,
        _ => image.cloned(),
    };
    // The engine's instructions are the chat's system turn, the user's prompt its user turn.
    let edit_system = if image.is_some() { ENHANCE_EDIT_SYSTEM.to_string() } else { ENHANCE_EDIT_SYSTEM.replace(" You can see the image.", "") };
    let template = if image.is_some() { ENHANCE_EDIT_TEMPLATE } else { ENHANCE_IMAGE_TEMPLATE };
    let system = match kind {
        EnhanceKind::Image => ENHANCE_IMAGE_SYSTEM.to_string(),
        EnhanceKind::ImagePhoto => ENHANCE_IMAGE_PHOTO_SYSTEM.to_string(),
        EnhanceKind::ImageUltra => ENHANCE_IMAGE_ULTRA_SYSTEM.to_string(),
        EnhanceKind::Edit => edit_system,
        EnhanceKind::Fill => format!("{edit_system}\n{}", ENHANCE_FILL_NOTE.replace("{place}", place.unwrap_or("selected part"))),
        EnhanceKind::Remove => format!("{edit_system}\n{ENHANCE_REMOVE_NOTE}"),
        EnhanceKind::Judge => ENHANCE_JUDGE_SYSTEM.to_string(),
        EnhanceKind::Presence => ENHANCE_PRESENCE_SYSTEM.to_string(),
        EnhanceKind::FillRef => ENHANCE_FILL_REF_SYSTEM.to_string(),
        EnhanceKind::Subject => ENHANCE_SUBJECT_SYSTEM.to_string(),
        EnhanceKind::Backdrop => ENHANCE_BACKDROP_SYSTEM.to_string(),
        EnhanceKind::CutEdges => ENHANCE_CUT_SYSTEM.to_string(),
        EnhanceKind::Scene => ENHANCE_SCENE_SYSTEM.to_string(),
        EnhanceKind::FillScene => ENHANCE_FILL_SCENE_SYSTEM.to_string(),
        EnhanceKind::Things => ENHANCE_THINGS_SYSTEM.replace("{k}", place.unwrap_or("6")),
    };
    let mut params = BTreeMap::new();
    params.insert("system".to_string(), Value::String(system));
    // AnyPaint's prompt is written by Krea 2's own Qwen3-VL-4B, as in AusBoss's workflow: the
    // fill that follows finds Krea 2 still resident instead of loading it again after
    // Qwen-Image-2.1's 8B answered (21.9-24.3 s a fill with enhance before, 2.4-4.1 s after).
    if kind == EnhanceKind::FillScene {
        params.insert("encoder".to_string(), Value::String(KREA2_CHAT_ENCODER.to_string()));
    }
    if matches!(kind, EnhanceKind::Image | EnhanceKind::ImagePhoto | EnhanceKind::ImageUltra) {
        // A text-to-image rewrite is 60 to 120 words: past the default 120 tokens it was cut
        // mid-sentence (2026-11-12). `place` names the encoder for these kinds: Krea 2's own
        // when the picture runs on Krea 2, a MOSS template's own Qwen3.5-9B when it runs on
        // MOSS, so the model stays resident (no 8B load and swap).
        let most = if kind == EnhanceKind::ImageUltra { ENHANCE_ULTRA_MAX_TOKENS } else { ENHANCE_IMAGE_MAX_TOKENS };
        params.insert("max_tokens".to_string(), Value::from(most));
        let own = |p: &str| p == KREA2_CHAT_ENCODER || [MOSS_CHAT_PREFIX, KLEIN_CHAT_PREFIX, MINIMAX_CHAT_PREFIX].iter().any(|x| p.starts_with(x));
        if let Some(enc) = place.filter(|p| own(p)) {
            params.insert("encoder".to_string(), Value::String(enc.to_string()));
        }
    }
    // A rewrite that replaces the user's words is checked ([`check_rewrite`]) and asked again
    // with the next seed when refused: three tries in all, then the words stay as typed.
    let rewrite = is_rewrite(kind);
    let user_words = prompt.strip_prefix(FILL_SCENE_LEAD).unwrap_or(prompt).trim();
    let tries = if rewrite { 3u64 } else { 2 };
    for attempt in 0..tries {
        let req = Request {
            template: template.to_string(),
            prompt: prompt.to_string(),
            negative: String::new(),
            seed: seed.wrapping_add(attempt) % MAX_SEED,
            steps: 0,
            guidance: 0.0,
            image: image.clone(),
            mask: None,
            models: Vec::new(),
            size: None,
            params: retry_params(&params, rewrite, attempt),
            references: Vec::new(),
        };
        let resp = backend.run(&req, progress)?;
        // A Yes or No and a list of things can be a single word ("No", "pear"): before
        // 2026-11-12 both needed three, so a one-thing list and every presence answer came back
        // as nothing.
        let min_words = if matches!(kind, EnhanceKind::Presence | EnhanceKind::Things) { 1 } else { 3 };
        let Some(said) = clean_answer(&resp.texts, min_words) else { continue };
        if !rewrite {
            return Ok(Some(said));
        }
        match check_rewrite(kind, user_words, &said, params.get("system").and_then(Value::as_str).unwrap_or_default()) {
            Ok(s) => return Ok(Some(s)),
            Err(why) => eprintln!("enhance: try {} refused ({}): {}", attempt + 1, why.words(), said),
        }
    }
    Ok(None)
}

/// Whether `left` looks more natural than `right` to the vision-language model
/// ([`ENHANCE_JUDGE_SYSTEM`]): `Some(true)` left, `Some(false)` right, `None` when it could
/// not be asked or did not answer (the caller falls back to [`edge_mismatch`]).
pub(super) fn judge_pair(
    backend: &dyn GenerativeBackend,
    left: &Rgba8,
    right: &Rgba8,
    object: Option<&str>,
    seed: u64,
    progress: &JobProgress,
) -> Option<bool> {
    let seen = side_by_side(&judge_view(left), &judge_view(right)).ok()?;
    let (rw, rh) = request_size_within(seen.width, seen.height, JUDGE_PAIR_MAX_PIXELS);
    let seen = rgba8_at(seen, rw, rh).ok()?;
    let answer = enhance_prompt(backend, EnhanceKind::Judge, &judge_question(object), Some(&seen), None, seed, progress).ok()??.to_lowercase();
    match (answer.find("left"), answer.find("right")) {
        (Some(l), Some(r)) => Some(l < r),
        (Some(_), None) => Some(true),
        (None, Some(_)) => Some(false),
        (None, None) => None,
    }
}

/// Whether each of a removal's results is asked on its own if the object is still there
/// ([`still_there`]) before the two are judged side by side. Off: the two extra loads of the
/// vision-language model per stroke, on top of the rewriter's and the judge's, had the GPU
/// driver reset twice in two runs (nvlddmkm event 153, the server killed mid-answer,
/// 2026-10-23), where the runs without it ran for hours; the enlarged judge view
/// ([`judge_view`]) is what let the judge see thin transparent frames, and stays.
pub(super) const PRESENCE_CHECK: bool = false;

/// Whether the vision-language model still sees `object` in `result`'s judged crop
/// ([`ENHANCE_PRESENCE_SYSTEM`]): `Some(true)` yes, `Some(false)` no, `None` when it could
/// not be asked or did not answer.
pub(super) fn still_there(backend: &dyn GenerativeBackend, result: &Rgba8, object: &str, seed: u64, progress: &JobProgress) -> Option<bool> {
    let answer = presence_answer(backend, result, &format!("Is there a {object}, or any part of one, in this photo?"), seed, progress)?;
    let first = answer.trim().split(|c: char| !c.is_alphanumeric()).next().unwrap_or("");
    match first {
        "yes" => Some(true),
        "no" => Some(false),
        _ => match (answer.find("yes"), answer.find("no")) {
            (Some(y), Some(n)) => Some(y < n),
            (Some(_), None) => Some(true),
            (None, Some(_)) => Some(false),
            (None, None) => None,
        },
    }
}

/// The vision-language model's Yes or No to `question` about `picture` (enlarged as for the
/// judge, [`judge_view`]), read from its first word only: `None` when it could not be asked or
/// its answer does not start with Yes or No.
pub(crate) fn sees(backend: &dyn GenerativeBackend, picture: &Rgba8, question: &str, seed: u64, progress: &JobProgress) -> Option<bool> {
    let answer = presence_answer(backend, picture, question, seed, progress)?;
    match answer.trim().split(|c: char| !c.is_alphanumeric()).next().unwrap_or("") {
        "yes" => Some(true),
        "no" => Some(false),
        _ => None,
    }
}

/// The model's answer (lower case) to a [`EnhanceKind::Presence`] question about `picture`.
pub(super) fn presence_answer(backend: &dyn GenerativeBackend, picture: &Rgba8, question: &str, seed: u64, progress: &JobProgress) -> Option<String> {
    let seen = judge_view(picture);
    Some(enhance_prompt(backend, EnhanceKind::Presence, question, Some(&seen), None, seed, progress).ok()??.to_lowercase())
}

/// A judged crop enlarged until its shorter side is at least [`JUDGE_VIEW_MIN_SIDE`], its
/// longer side at most [`JUDGE_VIEW_MAX_SIDE`] either way: thin things in a small crop were
/// lost on the judge, and a pair of crops at 1024 px each (a 1 MP picture for the
/// vision-language model with the 20 GB fill model resident) is what ran when the GPU driver
/// reset four times in a night (2026-10-23); the pair stays under half a megapixel now.
pub(super) fn judge_view(crop: &Rgba8) -> Rgba8 {
    let (w, h) = (crop.width, crop.height);
    let (shorter, longer) = (w.min(h).max(1), w.max(h).max(1));
    let up = f64::from(JUDGE_VIEW_MIN_SIDE) / f64::from(shorter);
    let cap = f64::from(JUDGE_VIEW_MAX_SIDE) / f64::from(longer);
    let scale = if longer > JUDGE_VIEW_MAX_SIDE { cap } else { up.min(cap).max(1.0) };
    if (scale - 1.0).abs() < 1e-6 {
        return crop.clone();
    }
    let (nw, nh) = (((f64::from(w) * scale).round() as u32).max(1), ((f64::from(h) * scale).round() as u32).max(1));
    resize_rgba8(crop, nw, nh).unwrap_or_else(|_| crop.clone())
}

pub(super) const JUDGE_VIEW_MIN_SIDE: u32 = 320;

pub(super) const JUDGE_VIEW_MAX_SIDE: u32 = 512;

/// The judge's side-by-side picture is sent at most this large (see [`judge_view`]).
pub(super) const JUDGE_PAIR_MAX_PIXELS: u64 = 512 * 1024;

/// The box of the pixels of `mask` at or above `level`, widened by a quarter of its larger
/// side: what the judge of a removal looks at.
pub(super) fn judged_rect(mask: &Gray8, level: u8) -> Rect {
    let w = mask.width as usize;
    let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for (i, m) in mask.data.iter().enumerate() {
        if *m >= level && w > 0 {
            let (x, y) = ((i % w) as i32, (i / w) as i32);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
        }
    }
    if x1 <= x0 || y1 <= y0 {
        return Rect::new(0, 0, mask.width as i32, mask.height as i32);
    }
    let m = ((x1 - x0).max(y1 - y0) / 4).max(8);
    Rect::new(x0 - m, y0 - m, x1 + m, y1 + m).intersect(&Rect::new(0, 0, mask.width as i32, mask.height as i32))
}
