//! Checking a rewritten prompt before it replaces the user's: the rewriter is a 4B to 9B
//! vision-language model, and before 2026-11-16 any answer of three words or more was taken.
//! A garbled answer (the release builds' broken quantized kernels on RTX 50 cards wrote
//! strings of junk tokens, and Enhance put them in the prompt box as "improved"), a refusal or
//! a note about the task, an answer cut off mid-sentence, and an answer that dropped what the
//! user asked for (their subject, a colour, a count, words in quotes) are all refused here; the
//! caller asks again with the next seed, and in the end keeps the prompt as typed.

use super::EnhanceKind;

/// Why a rewrite was refused (for the log and the status bar).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// Junk tokens: letters of another script, broken words, one word over and over.
    Garbled,
    /// "Sure, here is…", "I can't…", "As an AI…": talk about the task, not a prompt.
    Chatter,
    /// It says what the user typed and nothing more.
    Unchanged,
    /// It lost part of what the user asked for: their subject, a colour, a count, quoted words.
    LostIntent,
    /// It called the medium the user named (an oil painting, a 3D render) a photograph.
    PhotoForMedium,
    /// It chose a painted, drawn or rendered medium the user never named.
    InventedMedium,
}

impl Refusal {
    pub(crate) fn words(self) -> &'static str {
        match self {
            Refusal::Garbled => "the model's answer was garbled",
            Refusal::Chatter => "the model talked about the task or copied its instructions",
            Refusal::Unchanged => "the model gave the prompt back unchanged",
            Refusal::LostIntent => "the model's rewrite dropped part of what you asked for",
            Refusal::PhotoForMedium => "the model's rewrite turned the medium you named into a photograph",
            Refusal::InventedMedium => "the model's rewrite chose a painted or drawn look you didn't ask for",
        }
    }
}

/// The kinds whose answer is a prompt that replaces the user's words.
pub(crate) fn is_rewrite(kind: EnhanceKind) -> bool {
    matches!(
        kind,
        EnhanceKind::Fill
            | EnhanceKind::FillRef
            | EnhanceKind::FillScene
            | EnhanceKind::Edit
            | EnhanceKind::Image
            | EnhanceKind::ImagePhoto
            | EnhanceKind::ImageUltra
    )
}

/// The rewriter's answer to `typed`, checked and tidied (a "Here is the prompt:" lead-in
/// dropped, a text-to-image paragraph cut off mid-sentence ended at its last full sentence),
/// or why it is refused.
pub(crate) fn check_rewrite(kind: EnhanceKind, typed: &str, said: &str, system: &str) -> Result<String, Refusal> {
    let said = drop_lead_in(said.trim());
    // A medium the user named stays theirs. The 4B writers kept "an isometric 3d render" or "an
    // anime girl" through the paragraph and then closed with "A photograph with crisp 3d
    // rendering textures" (28 of 29 refusals in round 5, 2026-11-17): that closing sentence is
    // dropped rather than the whole rewrite refused. Before that, Krea 2's writer ended "an oil
    // painting" idea with "A photograph with visible brushwork" (round 2).
    let image = matches!(kind, EnhanceKind::Image | EnhanceKind::ImagePhoto | EnhanceKind::ImageUltra);
    let named_medium = image && super::names_other_medium(typed) && !typed.to_lowercase().contains("photo");
    let without_photo;
    let said = if named_medium && said.to_lowercase().contains("photograph") {
        without_photo = drop_sentences_with(said, "photograph");
        if without_photo.to_lowercase().contains("photograph") || without_photo.len() * 3 < said.len() * 2 {
            return Err(Refusal::PhotoForMedium);
        }
        without_photo.as_str()
    } else {
        said
    };
    if garbled(typed, said) {
        return Err(Refusal::Garbled);
    }
    // The photo and 16 MP captions are told to open with set phrases ("a portrait on an 85mm
    // lens"), so only the other rewriters are held to not copying their instructions.
    let copied = !matches!(kind, EnhanceKind::ImagePhoto | EnhanceKind::ImageUltra | EnhanceKind::FillScene) && copies_instructions(typed, said, system);
    if chatter(said) || copied {
        return Err(Refusal::Chatter);
    }
    // And no medium is chosen when none was named: MOSS's writer ended "portrait of an old
    // fisherman" with "a realistic oil painting" (A/B round 3), Krea 2's turned a kitchen into a
    // watercolour (round 1).
    if image && !super::names_other_medium(typed) && invents_medium(said) {
        return Err(Refusal::InventedMedium);
    }
    if normalised(said) == normalised(typed) {
        return Err(Refusal::Unchanged);
    }
    // A reference fill's answer describes the reference, not the typed words.
    if kind != EnhanceKind::FillRef && !keeps_intent(typed, said) {
        return Err(Refusal::LostIntent);
    }
    let said = if matches!(kind, EnhanceKind::Image | EnhanceKind::ImagePhoto | EnhanceKind::ImageUltra | EnhanceKind::FillScene) {
        end_at_full_sentence(said)
    } else {
        said
    };
    Ok(said.to_string())
}

/// "Here is the enhanced prompt:" / "Rewritten prompt:" on a line or before a colon, dropped.
fn drop_lead_in(s: &str) -> &str {
    let lower = s.to_lowercase();
    let leads = [
        "here is",
        "here's",
        "sure",
        "certainly",
        "okay",
        "ok,",
        "rewritten prompt",
        "enhanced prompt",
        "improved prompt",
        "prompt:",
        "final prompt",
        "output:",
    ];
    if !leads.iter().any(|l| lower.starts_with(l)) {
        return s;
    }
    // The lead-in ends at the first colon or line break within its first 15 words.
    let cut = s.find([':', '\n']).filter(|&i| s[..i].split_whitespace().count() <= 15);
    match cut {
        Some(i) => s[i + 1..].trim().trim_matches(|c: char| c == '"' || c == '*' || c == '\u{201c}' || c == '\u{201d}').trim(),
        None => s,
    }
}

fn normalised(s: &str) -> String {
    s.to_lowercase().chars().filter(|c| c.is_alphanumeric() || c.is_whitespace()).collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Latin script: ASCII and the accented letters of the Latin-1 and Latin Extended blocks.
fn latin(c: char) -> bool {
    c.is_ascii() || ('\u{00C0}'..='\u{024F}').contains(&c) || ('\u{2010}'..='\u{2027}').contains(&c) || c == '\u{00B0}' || c == '\u{00D7}'
}

/// Junk: another script the user did not write in, control or replacement characters, broken
/// words, or one word or phrase over and over.
pub(crate) fn garbled(typed: &str, said: &str) -> bool {
    let chars: Vec<char> = said.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.is_empty() {
        return true;
    }
    if said.chars().any(|c| c == '\u{FFFD}' || (c.is_control() && c != '\n' && c != '\t')) {
        return true;
    }
    // A prompt typed in another script may be answered in it.
    let typed_latin = typed.chars().filter(|c| !c.is_whitespace()).all(latin);
    if typed_latin {
        let foreign = chars.iter().filter(|c| !latin(**c)).count();
        if foreign * 50 > chars.len() {
            return true;
        }
    }
    let words: Vec<String> = said.split_whitespace().map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase()).filter(|w| !w.is_empty()).collect();
    if words.is_empty() {
        return true;
    }
    if typed_latin {
        // Broken words: very long, or long with no vowel ("xkqzt"), or letters and digits mixed
        // other than as sizes and lenses ("85mm", "4k", "3d", "1920s").
        let broken = words
            .iter()
            .filter(|w| {
                let n = w.chars().count();
                let letters = w.chars().filter(|c| c.is_alphabetic()).count();
                let digits = w.chars().filter(|c| c.is_ascii_digit()).count();
                let vowel = w.chars().any(|c| "aeiouyàáâäèéêëìíîïòóôöùúûü".contains(c));
                n > 22 || (letters >= 5 && !vowel) || (digits > 0 && letters > 3 && !w.chars().next().is_some_and(|c| c.is_ascii_digit()))
            })
            .count();
        if broken * 6 > words.len() {
            return true;
        }
    }
    // One word four times running.
    if words.windows(4).any(|w| w.iter().all(|x| x == &w[0])) {
        return true;
    }
    // A phrase of three words said four times or more.
    if words.len() >= 12 {
        let mut seen: std::collections::HashMap<&[String], usize> = std::collections::HashMap::new();
        for w in words.windows(3) {
            let n = seen.entry(w).or_default();
            *n += 1;
            if *n >= 4 {
                return true;
            }
        }
    }
    // Long answers with few distinct words.
    if words.len() >= 24 {
        let distinct: std::collections::HashSet<&String> = words.iter().collect();
        if distinct.len() * 10 < words.len() * 3 {
            return true;
        }
    }
    false
}

/// A refusal, an apology or talk about the task.
fn chatter(said: &str) -> bool {
    let l = said.to_lowercase();
    const MARKS: &[&str] = &[
        "as an ai",
        "i cannot",
        "i can't",
        "i can not",
        "i'm sorry",
        "i am sorry",
        "i apologize",
        "i'm unable",
        "i am unable",
        "the user's instruction",
        "the user wants",
        "the user asked",
        "the user's prompt",
        "the user's idea",
        "your instruction",
        "your prompt",
        "rewritten prompt",
        "enhanced prompt",
        "<think>",
        "</think>",
    ];
    MARKS.iter().any(|m| l.contains(m))
}

/// Words that say nothing about what the picture shows: articles, pronouns, the verbs of an
/// instruction, filler.
const STOP: &[&str] = &[
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "these",
    "those",
    "into",
    "onto",
    "from",
    "over",
    "under",
    "make",
    "makes",
    "making",
    "turn",
    "change",
    "add",
    "remove",
    "replace",
    "put",
    "give",
    "let",
    "please",
    "want",
    "like",
    "more",
    "less",
    "some",
    "very",
    "really",
    "its",
    "his",
    "her",
    "their",
    "them",
    "him",
    "she",
    "they",
    "you",
    "your",
    "our",
    "has",
    "have",
    "had",
    "was",
    "were",
    "are",
    "is",
    "be",
    "been",
    "being",
    "can",
    "could",
    "would",
    "should",
    "will",
    "just",
    "also",
    "too",
    "here",
    "there",
    "then",
    "than",
    "but",
    "not",
    "all",
    "any",
    "each",
    "out",
    "off",
    "about",
    "image",
    "picture",
    "photo",
    "photograph",
    "pic",
    "look",
    "looks",
    "looking",
    "lot",
    "bit",
    "thing",
    "things",
    "one",
    "get",
    "got",
    "use",
    "using",
    "show",
    "showing",
    "what",
    "which",
    "who",
    "how",
    "where",
    "when",
    "it's",
    "while",
    "keep",
    "stay",
    "same",
    "new",
    "different",
    "other",
    "another",
    "something",
    "anything",
    "everything",
    "whole",
    "entire",
    "area",
    "part",
    "selection",
    "selected",
    "grey",
    "gray",
    "shows",
];

const COLOURS: &[&str] = &[
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "violet",
    "pink",
    "brown",
    "black",
    "white",
    "grey",
    "gray",
    "gold",
    "golden",
    "silver",
    "teal",
    "cyan",
    "magenta",
    "beige",
    "navy",
    "maroon",
    "turquoise",
    "crimson",
    "scarlet",
    "lavender",
    "indigo",
    "olive",
    "ivory",
    "tan",
    "khaki",
    "bronze",
    "copper",
    "emerald",
    "coral",
    "mint",
];

const COUNTS: &[&str] = &["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "dozen", "pair", "twins", "single"];

fn tokens(s: &str) -> Vec<String> {
    s.split(|c: char| !(c.is_alphanumeric() || c == '\'')).map(|w| w.trim_matches('\'').to_lowercase()).filter(|w| !w.is_empty()).collect()
}

/// Text in double quotes (straight or curly): words to be drawn, kept exactly.
fn quoted(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let s = s.replace(['\u{201c}', '\u{201d}'], "\"");
    let mut parts = s.split('"');
    parts.next();
    while let (Some(inside), _) = (parts.next(), parts.next()) {
        if !inside.trim().is_empty() {
            out.push(inside.trim().to_string());
        }
    }
    out
}

/// Whether a word of the answer stands for `w` of the prompt: the same word, or one that shares
/// its first five letters (realistic / realism, snowy / snow), or its singular.
fn found(w: &str, said: &[String]) -> bool {
    let stem: String = w.chars().take(5).collect();
    let short = w.chars().count() < 5;
    let singular = w.strip_suffix("es").filter(|s| s.len() >= 3).or_else(|| w.strip_suffix('s').filter(|s| s.len() >= 3));
    said.iter().any(|s| {
        s == w || (!short && s.starts_with(&stem)) || singular.is_some_and(|x| s == x || s.starts_with(x)) || (short && s.strip_suffix('s') == Some(w))
    })
}

/// Whether the answer keeps what the user asked for: every quoted text exactly, every colour
/// and count, and most of the other words that name something (all of one or two, three in
/// five).
pub(crate) fn keeps_intent(typed: &str, said: &str) -> bool {
    for q in quoted(typed) {
        if !said.replace(['\u{201c}', '\u{201d}'], "\"").contains(&q) {
            return false;
        }
    }
    // Words are only counted in a Latin script: Chinese or Japanese runs without spaces.
    if !typed.chars().filter(|c| !c.is_whitespace()).all(latin) {
        return true;
    }
    let said_words = tokens(said);
    let typed_words = tokens(typed);
    let hard = typed_words.iter().filter(|w| COLOURS.contains(&w.as_str()) || COUNTS.contains(&w.as_str()) || w.chars().all(|c| c.is_ascii_digit()));
    for w in hard {
        if !found(w, &said_words) {
            return false;
        }
    }
    let mut content: Vec<&String> = typed_words.iter().filter(|w| w.chars().count() >= 3 && !STOP.contains(&w.as_str())).collect();
    content.dedup();
    let n = content.len();
    if n == 0 {
        return true;
    }
    let kept = content.iter().filter(|w| found(w, &said_words)).count();
    let need = match n {
        1 | 2 => 1.max(n - 1),
        _ => (n * 3).div_ceil(5),
    };
    kept >= need
}

/// `s` without its sentences that contain `word` (any case), a sentence ending at a full stop,
/// question or exclamation mark, or at the end of the text (a closing "A photograph" cut off by
/// the token limit).
fn drop_sentences_with(s: &str, word: &str) -> String {
    let mut out = String::new();
    let mut start = 0;
    let bytes: Vec<(usize, char)> = s.char_indices().collect();
    for (k, &(i, c)) in bytes.iter().enumerate() {
        let last = k + 1 == bytes.len();
        if matches!(c, '.' | '!' | '?') || last {
            let end = i + c.len_utf8();
            let sentence = &s[start..end];
            if !sentence.to_lowercase().contains(word) {
                out.push_str(sentence);
            }
            start = end;
        }
    }
    out.trim().to_string()
}

/// A paragraph cut off mid-sentence (the token limit) ends at its last full stop instead, when
/// that keeps at least two thirds of it.
fn end_at_full_sentence(s: &str) -> &str {
    let t = s.trim_end();
    if t.ends_with(['.', '!', '?', '"', '\u{201d}', ')']) {
        return t;
    }
    match t.rfind(['.', '!', '?']) {
        Some(i) if i + 1 >= t.len() * 2 / 3 => &t[..=i],
        _ => t,
    }
}

/// Whether the answer copies its instructions: a run of six words from the system turn that the
/// user did not type ("keeping the yellow note, the fruit bowl and the cabinets unchanged" from
/// the edit example in every edit, "without adding extra people or animals" from a rule; A/B
/// round 2, 2026-11-16).
pub(crate) fn copies_instructions(typed: &str, said: &str, system: &str) -> bool {
    const RUN: usize = 6;
    let sys = tokens(system);
    if sys.len() < RUN {
        return false;
    }
    let runs: std::collections::HashSet<&[String]> = sys.windows(RUN).collect();
    let typed = tokens(typed).join(" ");
    tokens(said).windows(RUN).any(|w| runs.contains(w) && !typed.contains(&w.join(" ")))
}

/// Whether a rewrite names a painted, drawn or rendered medium. Narrower than
/// `names_other_medium` on purpose: "rendered as a photograph", "pastel tones" and "a poster on
/// the wall" are not a change of medium.
fn invents_medium(said: &str) -> bool {
    const MEDIA: &[&str] = &[
        "painting",
        "watercolor",
        "watercolour",
        "gouache",
        "oil on canvas",
        "impasto",
        "brushstroke",
        "brush stroke",
        "illustration",
        "drawing",
        "sketch",
        "cartoon",
        "anime",
        "manga",
        "comic",
        "3d render",
        "cgi",
        "pixel art",
        "line art",
        "concept art",
        "digital art",
        "low poly",
        "claymation",
        "woodcut",
        "linocut",
        "engraving",
        "charcoal",
        "pencil",
        "ink wash",
        "cel shaded",
        "cel-shaded",
        "voxel",
    ];
    let l = said.to_lowercase();
    MEDIA.iter().any(|m| l.contains(m))
}
