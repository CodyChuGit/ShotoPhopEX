//! Split into Layers' checks on a fill: whether the subject is gone, whether someone new was
//! painted (tile by tile, pictures of people and fog set aside), and blank or ghosted holes.

use std::collections::VecDeque;

use photocraft_geom::Rect;

use crate::jobs::JobCtx;
use crate::select_ml_cmds::{Instance, SubjectFinder};
use crate::{EngineError, Result};

use super::{
    ANIMAL, BLANK_CELL, BLANK_CELLS, BLANK_SHARE, COMPLETION_MAX, CONTINUES, FIGURE_TEXTURE, FRAGMENT_SHARE, GHOST_CLUSTER_CELLS, GHOST_CLUSTER_SHARE,
    GHOST_LUMA, IN_PICTURE, JOIN_REACH, JUDGE_MIN_PIXELS, KEPT_BLOCK, KEPT_LOOKS, KEPT_TOLERANCE, LEFT_IN_HOLE, LEFT_MIN_SHARE, NEW_AT_MOST, NEW_INSIDE,
    NEW_MIN_PIXELS, NEW_MIN_SHARE, PART_GAP, PART_SIDE, PICTURE_MOST, PICTURE_OVER, PICTURE_TERMS, PICTURES_AT_MOST, PLAIN_TEXTURE, RING_WIDTH, Subject,
    TEXTURE_PIXELS, THING_IN_HOLE, Value, WHOLE_FRAME, ZOOM_ACROSS, ZOOM_MARGIN, ZOOM_MIN_PIXELS, ZOOM_MIN_SIDE, ZOOM_TILE, bare, json,
};

/// Whether a detection sits mostly in the hole and covers enough of it: `share` of the
/// detection inside the hole and at least [`LEFT_MIN_SHARE`] of the hole (a person beside the
/// hole overlaps it little; a speck is nothing). A detection over nearly the whole frame is the
/// detector's quirk, not a subject (it flagged one clean backdrop fill whole, 2026-11-03).
pub(super) fn mostly_in_hole(cov: &[f32], hole: &[f32], share: f32) -> bool {
    let holed = hole.iter().filter(|v| **v >= 0.5).count();
    if holed == 0 {
        return false;
    }
    let (mut on, mut inside) = (0usize, 0usize);
    for (c, h) in cov.iter().zip(hole) {
        if *c >= 0.5 {
            on += 1;
            if *h >= 0.5 {
                inside += 1;
            }
        }
    }
    if on as f32 > WHOLE_FRAME * cov.len() as f32 {
        return false;
    }
    on > 0 && inside as f32 >= share * on as f32 && inside as f32 >= LEFT_MIN_SHARE * holed as f32
}

/// How many pixels two masks share.
pub(super) fn overlap(a: &[f32], b: &[f32]) -> u64 {
    a.iter().zip(b).filter(|(x, y)| **x >= 0.5 && **y >= 0.5).count() as u64
}

/// Whether a detection is a subject seen again (its completed part included): it holds most of
/// the subject's mask.
pub(super) fn continues(detection: &[f32], subject: &[f32], subject_pixels: u64) -> bool {
    subject_pixels > 0 && overlap(detection, subject) as f32 >= CONTINUES * subject_pixels as f32
}

/// Whether a detection (`on` pixels) is someone who was in the picture before any fill
/// (`already`): it holds half of them; or it continues them ([`continues`]) and adds at most
/// [`COMPLETION_MAX`] times their visible pixels in `touched`; or it continues them where they
/// stood against the cut subject (`cut_cov`, within [`JOIN_REACH`]), whose removal shows the
/// rest of them, however much that is (the man in a framed photograph, mostly behind the
/// waiter, refused the waiter's removal in four runs of round 9 for adding more than twice his
/// visible sliver; measured in the app 2026-11-04).
pub(super) fn was_there(cov: &[f32], on: u64, touched: &[f32], cut_cov: &[f32], already: &[Figure], w: usize) -> bool {
    already.iter().any(|(a, an)| {
        overlap(cov, a) * 2 >= on || (continues(cov, a, *an) && (added_in(cov, touched, a) <= *an * COMPLETION_MAX || near(a, cut_cov, w, JOIN_REACH as usize)))
    })
}

/// Whether a detection of `term` is the completed part of a subject behind of that kind, seen
/// on its own: joined to them (within [`JOIN_REACH`]) in `region` (where their hidden part was),
/// their own pixels and that joined part making [`FRAGMENT_SHARE`] of it, the joined part at
/// most [`COMPLETION_MAX`] times their visible pixels. The detector returned what the fill
/// completed of a person waiting their turn as a person of its own, holding none of their mask,
/// and the first strips behind the rightmost man were refused for it in two runs of round 9
/// (measured in the app 2026-11-04). The bound holds a second copy of someone painted beside
/// them to their size: a copy of the woman in blue is more than twice the sliver of the man at
/// the frame's edge (round 4).
pub(super) fn fragment_of<'a>(term: &str, cov: &[f32], region: &[f32], behind: impl IntoIterator<Item = (&'a str, &'a [f32], u64)>, w: usize) -> bool {
    if w == 0 || !cov.len().is_multiple_of(w) {
        return false;
    }
    let h = cov.len() / w;
    let on = cov.iter().filter(|c| **c >= 0.5).count() as u64;
    on > 0
        && behind.into_iter().any(|(t, bc, bn)| {
            if t != term || bn == 0 || !near(cov, bc, w, JOIN_REACH as usize) {
                return false;
            }
            let wanted: Vec<bool> = cov.iter().zip(region).zip(bc).map(|((c, r), b)| *c >= 0.5 && *r >= 0.5 && *b < 0.5).collect();
            let joined = joined_to(&wanted, bc, w, h).iter().filter(|j| **j).count() as u64;
            (overlap(cov, bc) + joined) as f32 >= FRAGMENT_SHARE * on as f32 && joined <= bn * COMPLETION_MAX
        })
}

/// The subjects behind as borrowed shapes (kind, mask, pixels).
pub(super) fn borrowed(behind: &[(String, Vec<f32>, u64)]) -> impl Iterator<Item = (&str, &[f32], u64)> {
    behind.iter().map(|(t, bc, bn)| (t.as_str(), bc.as_slice(), *bn))
}

/// Why a detection on a fill's result is the fill's own work (the subject painted back, or
/// someone new), or `None` when it was there before. It sits mostly in what the fill repainted
/// (a person: the hole widened by the grow; a thing: its own spot, as a saucer beside a cup
/// overlaps the grow and not the cup), and continues no subject behind by at most
/// [`COMPLETION_MAX`] times that subject's visible pixels (its hidden part completed). A second
/// copy of the woman in blue, painted beside her, covered the sliver of the man at the frame's
/// edge and passed as his completion (measured in the app 2026-11-04). A completion that still
/// `looks` like what was cut ([`KEPT_LOOKS`] of its pixels in the hole as they were) is the
/// subject painted back over whoever stands behind: the front-left man of the café photo came
/// back almost pixel for pixel and passed (measured in the app 2026-11-04). A person's or an
/// animal's completed part seen on its own, joined to a subject behind of its kind
/// ([`fragment_of`]), is that subject's too, under the same look. `cut` is the subject whose hole
/// this is (its kind, mask and pixels).
#[allow(clippy::too_many_arguments)]
pub(in crate::generate_cmds) fn fills_own_work(
    term: &str,
    cov: &[f32],
    hole: &[f32],
    grown: &[f32],
    cut: (&str, &[f32], u64),
    behind: &[(String, Vec<f32>, u64)],
    looks: &dyn Fn(&[f32]) -> f32,
    w: usize,
    fog: &dyn Fn(&[f32]) -> bool,
) -> Option<String> {
    let (region, share) = if creature(term) { (grown, LEFT_IN_HOLE) } else { (hole, THING_IN_HOLE) };
    if !mostly_in_hole(cov, region, share) {
        return None;
    }
    let (cut_term, cut_cov, cut_pixels) = cut;
    // What a completion of someone behind that still looks like what was cut is called.
    let painted_over = |kept: f32| {
        if term == cut_term {
            format!("the {term} painted back over the one behind ({:.0} % of it as it was)", kept * 100.0)
        } else {
            format!("the {cut_term} still there under the {term} behind ({:.0} % of it as it was)", kept * 100.0)
        }
    };
    let continued = behind.iter().filter(|(t, bc, bn)| t == term && continues(cov, bc, *bn)).collect::<Vec<_>>();
    if continued.iter().any(|(_, bc, bn)| added_in(cov, region, bc) <= *bn * COMPLETION_MAX) {
        let kept = looks(cov);
        return (kept >= KEPT_LOOKS).then(|| painted_over(kept));
    }
    if term == cut_term && overlap(cov, cut_cov) * 2 >= cut_pixels {
        return Some(format!("the {term} painted back"));
    }
    if !continued.is_empty() {
        return Some(format!("a {term} over the one behind, more than a completion"));
    }
    if creature(term) {
        // A part of a listed person behind, seen on its own (the detector returned the woman's
        // upper body beside the hole, less than the 60 % that continues her), is that person.
        let on = cov.iter().filter(|c| **c >= 0.5).count() as u64;
        if behind.iter().any(|(t, bc, _)| t == term && overlap(cov, bc) * 2 >= on) {
            return None;
        }
        // So is their completed part seen on its own, joined to them ([`fragment_of`]: the
        // first strips behind the rightmost man were refused for the two people waiting their
        // turn, rounds 8 and 9), unless it still looks like what was cut.
        if fragment_of(term, cov, region, borrowed(behind), w) {
            let kept = looks(cov);
            return (kept >= KEPT_LOOKS).then(|| painted_over(kept));
        }
    }
    // A thing seen apart from the one behind it that it joins (the table's stretch across the
    // hole, found as a table of its own) is that thing's completion, not a new one: "a table in
    // the hole that was not there" refused five strip fills behind the people at the café table
    // (measured in the app 2026-11-04). A person is never excused this way: a second copy of
    // the woman painted beside her would pass as the man at the frame's edge.
    if !creature(term) && behind.iter().any(|(t, bc, _)| t == term && near(cov, bc, w, JOIN_REACH as usize)) {
        return None;
    }
    // A figure with no more detail than the fog around it is no figure: the detector called the
    // fog of the striped man's strips a person three times (measured in the app 2026-11-04).
    if creature(term) && fog(cov) {
        return None;
    }
    Some(format!("a {term} in the hole that was not there"))
}

/// Why a detection of `term` (a person or an animal) on a fill's result is someone the fill
/// painted: it stands mostly ([`NEW_INSIDE`]) where the fill was laid in (`touched`), and it is
/// neither the subject whose hole this is (painted back) nor someone behind seen again (a part of
/// them, their completion by at most [`COMPLETION_MAX`] times their visible pixels, or their
/// completed part on its own, joined to them: [`fragment_of`]). `None` when it was there before.
/// Small figures the frame-wide check never saw (ranked below the people still there, or under
/// its share of the hole): six invented diners and office people on the café backgrounds, and
/// one on the portrait's (measured in the app 2026-11-04). Whoever was in the picture before any
/// fill (`already`, the people apart from the subjects: a man in a framed photograph beside the
/// hole refused the waiter's removal four times, round 7) is nobody's work ([`was_there`]). A
/// rule that also excused what the fill left as it was by its pixels (half of the detection's
/// 8-px blocks where the fill was laid in keeping their colour) was withdrawn after round 11: the
/// cut subject painted back in part keeps its pixels there too, and a repainted person passed on
/// the portrait (measured in the app 2026-11-04); a white shard on the near table the detector
/// calls a person costs its fills instead (peel seed 2 of round 10).
#[allow(clippy::too_many_arguments)]
pub(in crate::generate_cmds) fn someone_new(
    term: &str,
    cov: &[f32],
    bounds: Rect,
    touched: &[f32],
    cut: (&str, &[f32], u64),
    behind: &[(String, Vec<f32>, u64)],
    already: &[Figure],
    w: usize,
) -> Option<String> {
    let on = cov.iter().filter(|c| **c >= 0.5).count() as u64;
    let inside = overlap(cov, touched);
    if on == 0 || (inside as f32) < NEW_INSIDE * on as f32 {
        return None;
    }
    let (cut_term, cut_cov, cut_pixels) = cut;
    if term == cut_term && overlap(cov, cut_cov) * 2 >= cut_pixels {
        return Some(format!("the {term} painted back"));
    }
    if was_there(cov, on, touched, cut_cov, already, w) {
        return None;
    }
    if behind
        .iter()
        .any(|(t, bc, bn)| t == term && (overlap(cov, bc) * 2 >= on || (continues(cov, bc, *bn) && added_in(cov, touched, bc) <= *bn * COMPLETION_MAX)))
    {
        return None;
    }
    if fragment_of(term, cov, touched, borrowed(behind), w) {
        return None;
    }
    Some(format!("a {term} painted into the hole ({}×{} px)", bounds.width(), bounds.height()))
}

/// Why a detection on a fill's result is the fill's own work: nothing when it was there before
/// any fill ([`was_there`]); else [`fills_own_work`] (the subject painted back, more than a
/// completion, a thing on its spot), then, for a person or an animal, [`someone_new`] wherever
/// the fill painted at all. With `patched` (a thing's patch fill, which
/// copies the picture around the hole and so shows the detector the thing again by construction)
/// a thing's detection does not count.
#[allow(clippy::too_many_arguments)]
pub(super) fn judge(
    term: &str,
    cov: &[f32],
    bounds: Rect,
    hole: &[f32],
    grown: &[f32],
    touched: &[f32],
    cut: (&str, &[f32], u64),
    behind: &[(String, Vec<f32>, u64)],
    already: &[Figure],
    looks: &dyn Fn(&[f32]) -> f32,
    fog: &dyn Fn(&[f32]) -> bool,
    w: usize,
    patched: bool,
) -> Option<String> {
    // Whoever was in the picture before any fill, apart from the subjects, is nobody's work (the
    // cut subject painted back over them is), wherever the detection lies.
    if creature(term) {
        let (cut_term, cut_cov, cut_pixels) = cut;
        let on = cov.iter().filter(|c| **c >= 0.5).count() as u64;
        if !(term == cut_term && overlap(cov, cut_cov) * 2 >= cut_pixels) && was_there(cov, on, touched, cut_cov, already, w) {
            return None;
        }
    }
    if (creature(term) || !patched)
        && let Some(why) = fills_own_work(term, cov, hole, grown, cut, behind, looks, w, fog)
    {
        return Some(why);
    }
    if creature(term) {
        // A figure with no more detail than the fog around it is no figure.
        let new = someone_new(term, cov, bounds, touched, cut, behind, already, w);
        return if new.is_some() && fog(cov) { None } else { new };
    }
    None
}

/// What the check asks the detector for after a fill: the subject's kind (painted back, and
/// one more for someone new of its kind), the kinds still behind it (their completed parts),
/// and people in any case (anyone painted in). A person's or animal's kind is asked for
/// [`NEW_AT_MOST`] at least, down to [`new_floor`].
pub(super) fn asks_for<'a>(cut_term: &str, behind: impl Iterator<Item = &'a Subject>) -> Vec<(String, usize)> {
    let mut asks: Vec<(String, usize)> = vec![(cut_term.to_string(), 2)];
    for b in behind {
        match asks.iter_mut().find(|(t, _)| t == b.check_term()) {
            Some((_, n)) => *n += 1,
            None => asks.push((b.check_term().to_string(), 2)),
        }
    }
    if !asks.iter().any(|(t, _)| t == PERSON) {
        asks.push((PERSON.to_string(), 1));
    }
    asks
}

/// The error when no fill cleared any subject, with every try's verdict: "could not clear the
/// person out of the background on 6 tries (removal kept what stood there (78 % of it as it
/// was); …), so nothing was changed".
pub(super) fn could_not_clear(what: &str, tries: &[Value]) -> EngineError {
    let why: Vec<String> = tries.iter().map(|t| format!("{} {}", t["fill"].as_str().unwrap_or("fill"), t["verdict"].as_str().unwrap_or("refused"))).collect();
    EngineError::Other(format!(
        "Split into Layers could not clear the {what} out of the background on {} tries ({}), so nothing was changed",
        tries.len(),
        why.join("; ")
    ))
}

/// The smallest detection (pixels) that counts as someone painted in, on a `w`×`h` picture.
pub(super) fn new_floor(w: usize, h: usize) -> u64 {
    (((w * h) as f64 * NEW_MIN_SHARE).ceil() as u64).max(NEW_MIN_PIXELS)
}

/// The instances of `term` the check looks at on a fill's result: every one down to the floor
/// for a person or an animal, the largest `at_most` for a thing.
pub(super) fn seen_of(finder: &SubjectFinder, filled: &[u8], area: Rect, term: &str, at_most: usize, ctx: &JobCtx) -> Result<Vec<Instance>> {
    let (wu, hu) = (area.width() as usize, area.height() as usize);
    if creature(term) {
        finder.find_all_in(filled, area, term, at_most.max(NEW_AT_MOST), new_floor(wu, hu), ctx)
    } else {
        finder.find_in(filled, area, term, at_most, ctx)
    }
}

/// Whether any pixel of `a` lies within `reach` pixels of one of `b` (both over rows of `w`).
pub(super) fn near(a: &[f32], b: &[f32], w: usize, reach: usize) -> bool {
    if w == 0 {
        return false;
    }
    let h = a.len() / w;
    let r = reach as i64;
    a.iter().enumerate().filter(|(_, v)| **v >= 0.5).any(|(k, _)| {
        let (x, y) = ((k % w) as i64, (k / w) as i64);
        (-r..=r).any(|dy| {
            (-r..=r).any(|dx| {
                let (nx, ny) = (x + dx, y + dy);
                nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h && b[ny as usize * w + nx as usize] >= 0.5
            })
        })
    })
}

/// People the detector still finds on the finished background, apart from the subjects left in
/// it (`left`, their holes): their bounds, `leftBehind` in the result, so a sheet says at once
/// whether anyone was invented or missed (the app's own detector found one on every round-6
/// café background the per-fill checks had passed; measured 2026-11-04).
/// Looked for in the whole frame and, unless the backdrop is plain (`by_tiles`), tile by tile
/// ([`found_by_tiles`]); whoever was in the picture before any fill (`already`, whole or
/// completed by the fills) is not listed, nor a figure with no more detail than fog (the smears
/// at the left edge listed in three round-8 backgrounds; measured 2026-11-04).
#[allow(clippy::too_many_arguments)]
pub(super) fn left_behind(
    finder: &SubjectFinder,
    background: &[u8],
    area: Rect,
    left: &[Vec<f32>],
    already: &[Figure],
    picture_tex: f32,
    by_tiles: bool,
    ctx: &JobCtx,
) -> Result<Vec<Value>> {
    let (wu, hu) = (area.width() as usize, area.height() as usize);
    let mut found = finder.find_all_in(background, area, PERSON, NEW_AT_MOST, new_floor(wu, hu), ctx)?;
    if by_tiles {
        found.extend(found_by_tiles(finder, background, area, area, PERSON, ctx)?);
    }
    let mut out = Vec::new();
    let mut listed: Vec<Vec<f32>> = Vec::new();
    // The pictures on the background, looked for once someone is about to be listed.
    let mut pictures = Pictures::new(finder, background, area, ctx);
    for (cov, b, n) in found {
        // Someone there before, whole or completed (the man in the framed photograph, whole once
        // the waiter was gone, was listed in three round-9 backgrounds), or listed already.
        if left.iter().chain(already.iter().map(|(a, _)| a)).chain(&listed).any(|other| overlap(&cov, other) * 2 >= n)
            || already.iter().any(|(a, an)| continues(&cov, a, *an))
        {
            continue;
        }
        if fog_counted(finder, background, &cov, wu, hu, picture_tex) {
            continue;
        }
        // Whoever is in a framed picture is a picture of a person (the man in the framed
        // photograph was listed once more in round 10's sweep at 16 layers).
        if pictures.holds(&cov, b)? {
            continue;
        }
        out.push(json!([b.x0, b.y0, b.width(), b.height()]));
        listed.push(cov);
    }
    Ok(out)
}

/// A figure in the picture: its mask over the area and its pixels.
pub(super) type Figure = (Vec<f32>, u64);

/// Whether a detection has no more detail than fog, counted on the finder's record: its texture
/// under [`FIGURE_TEXTURE`] of the picture's (`picture_tex`, [`picture_texture`]); never on a
/// plain picture, whose texture says nothing.
pub(super) fn fog_counted(finder: &SubjectFinder, img: &[u8], cov: &[f32], w: usize, h: usize, picture_tex: f32) -> bool {
    let foggy = no_more_than_fog(img, cov, w, h, picture_tex);
    if foggy {
        finder.note_fog();
    }
    foggy
}

/// Whether a detection on `img` has no more detail than fog: its texture under
/// [`FIGURE_TEXTURE`] of the picture's (`picture_tex`). The detector called the fog of the
/// striped man's strips a person three times (81×318, 47×112 and 37×86 px) and he stayed in
/// the background; a painted figure has the detail of the picture around it (measured in the
/// app 2026-11-04). Never on a plain picture (texture under [`PLAIN_TEXTURE`]).
pub(in crate::generate_cmds) fn no_more_than_fog(img: &[u8], cov: &[f32], w: usize, h: usize, picture_tex: f32) -> bool {
    picture_tex >= PLAIN_TEXTURE && texture_and_luma(img, cov, w, h, texture_step(w, h)).is_some_and(|(tex, _)| tex < FIGURE_TEXTURE * picture_tex)
}

/// Whether the picture outside the subjects is a plain backdrop (texture under
/// [`PLAIN_TEXTURE`]). Then the detector is not shown it tile by tile: a figure painted on it is
/// seen in the whole frame, and the tiles cost images.jpg two seconds of six (rounds 8 and 9;
/// measured in the app 2026-11-04).
pub(super) fn plain_backdrop(picture: &[u8], subjects: &[Subject], w: usize, h: usize) -> bool {
    let mut off = vec![1.0f32; w * h];
    for s in subjects {
        for (d, c) in off.iter_mut().zip(&s.cov) {
            if *c >= 0.5 {
                *d = 0.0;
            }
        }
    }
    texture_and_luma(picture, &off, w, h, texture_step(w, h)).is_none_or(|(t, _)| t < PLAIN_TEXTURE)
}

/// The figures' bounds as `[x, y, width, height]` boxes (`already` in the result).
pub(super) fn boxes_of(figures: &[Figure], area: Rect) -> Vec<Value> {
    figures
        .iter()
        .map(|(cov, _)| {
            let (b, _) = crate::select_ml_cmds::coverage_bounds(cov, area);
            json!([b.x0, b.y0, b.width(), b.height()])
        })
        .collect()
}

/// The texture of the whole picture (at [`texture_step`]).
pub(in crate::generate_cmds) fn picture_texture(picture: &[u8], w: usize, h: usize) -> f32 {
    texture_and_luma(picture, &vec![1.0f32; w * h], w, h, texture_step(w, h)).map_or(0.0, |(tex, _)| tex)
}

/// The tiles over `bounds` within `area`, each a part of the picture the detector is shown on
/// its own: the bounds ([`ZOOM_MARGIN`] around them for context) in pieces no longer than
/// [`zoom_tile`] a side, each piece widened to that side, centred and slid to stay within the
/// picture; none for empty bounds, and none that is the whole picture (the detector saw that
/// already, and a picture no larger than a tile has nothing to zoom into).
pub(super) fn tiles_over(bounds: Rect, area: Rect) -> Vec<Rect> {
    let bounds = bounds.inflate(ZOOM_MARGIN as i32).intersect(&area);
    if bounds.is_empty() {
        return Vec::new();
    }
    let side = zoom_tile(area) as u32;
    let (bw, bh) = (bounds.width(), bounds.height());
    let (nx, ny) = (bw.div_ceil(side).max(1), bh.div_ceil(side).max(1));
    let (pw, ph) = (bw.div_ceil(nx) as i32, bh.div_ceil(ny) as i32);
    let side = side as i32;
    let span = |p0: i32, p1: i32, a0: i32, a1: i32| -> (i32, i32) {
        let len = (p1 - p0).max(side).min(a1 - a0);
        let s0 = ((p0 + p1 - len) / 2).clamp(a0, a1 - len);
        (s0, s0 + len)
    };
    let mut tiles: Vec<Rect> = Vec::with_capacity((nx * ny) as usize);
    for j in 0..ny as i32 {
        for i in 0..nx as i32 {
            let (px0, py0) = (bounds.x0 + i * pw, bounds.y0 + j * ph);
            let (x0, x1) = span(px0, (px0 + pw).min(bounds.x1), area.x0, area.x1);
            let (y0, y1) = span(py0, (py0 + ph).min(bounds.y1), area.y0, area.y1);
            let tile = Rect::new(x0, y0, x1, y1);
            if !tile.is_empty() && tile != area && !tiles.contains(&tile) {
                tiles.push(tile);
            }
        }
    }
    tiles
}

/// How long a tile's side is at most: a third of the picture's longer side, so the detector sees
/// a tile three times the size it sees the frame at, and [`ZOOM_TILE`] at least.
pub(super) fn zoom_tile(area: Rect) -> usize {
    (area.width().max(area.height()) as usize / ZOOM_ACROSS).max(ZOOM_TILE)
}

/// `tile` of `picture` (RGBA over `area`) as a picture of its own.
pub(super) fn crop_rgba(picture: &[u8], area: Rect, tile: Rect) -> Vec<u8> {
    let aw = area.width() as usize;
    let (tw, th) = (tile.width() as usize, tile.height() as usize);
    let mut out = Vec::with_capacity(tw * th * 4);
    for y in 0..th {
        let row = (tile.y0 - area.y0) as usize + y;
        let start = (row * aw + (tile.x0 - area.x0) as usize) * 4;
        out.extend_from_slice(&picture[start..start + tw * 4]);
    }
    out
}

/// A mask over `tile` laid into one over `area` (off elsewhere).
pub(super) fn paste_mask(cov: &[f32], tile: Rect, area: Rect) -> Vec<f32> {
    let aw = area.width() as usize;
    let (tw, th) = (tile.width() as usize, tile.height() as usize);
    let mut out = vec![0.0f32; aw * area.height() as usize];
    for y in 0..th {
        let row = (tile.y0 - area.y0) as usize + y;
        let start = row * aw + (tile.x0 - area.x0) as usize;
        out[start..start + tw].copy_from_slice(&cov[y * tw..(y + 1) * tw]);
    }
    out
}

/// The instances of `term` the detector finds in `picture` shown tile by tile over `bounds`
/// ([`tiles_over`]), down to [`ZOOM_MIN_PIXELS`] and [`ZOOM_MIN_SIDE`], as masks over `area`.
/// A figure 20 px tall is a third of that in the whole frame sent at the detector's size, and
/// was missed or fell under the floor (round 7's invented diners, 15–40 px tall, in five of
/// eight café backgrounds; measured in the app 2026-11-04).
/// Only figures small for the tile count (under half its side either way): the frame-wide look
/// has judged anything larger, and a mask over the whole tile is no figure (the portrait's
/// tiles came back as 1344-px "people", a third of its side, and every try was refused;
/// measured in the app 2026-11-04).
pub(super) fn found_by_tiles(finder: &SubjectFinder, picture: &[u8], area: Rect, bounds: Rect, term: &str, ctx: &JobCtx) -> Result<Vec<Instance>> {
    let mut out = Vec::new();
    let tiles = tiles_over(bounds, area);
    finder.note_tiles(tiles.len() as u32);
    for tile in tiles {
        ctx.check()?;
        let crop = crop_rgba(picture, area, tile);
        for (cov, b, n) in finder.find_all_in(&crop, tile, term, NEW_AT_MOST, ZOOM_MIN_PIXELS, ctx)? {
            let small = b.width() * 2 <= tile.width() && b.height() * 2 <= tile.height();
            if small && b.width() >= ZOOM_MIN_SIDE && b.height() >= ZOOM_MIN_SIDE {
                out.push((paste_mask(&cov, tile, area), b, n));
            }
        }
    }
    Ok(out)
}

/// Who was in the picture before any fill, apart from the subjects: every instance of each of
/// the creature `terms`, in the whole frame and, unless the backdrop is plain (`by_tiles`), by
/// tiles ([`found_by_tiles`]), that is not mostly on a subject (those the fills take out). A fill is not blamed for them (the man in a framed
/// photograph beside the waiter's hole refused four removals, round 7; measured in the app
/// 2026-11-04), nor are they listed in `leftBehind`.
pub(super) fn already_there(
    finder: &SubjectFinder,
    picture: &[u8],
    area: Rect,
    terms: &[String],
    subjects: &[Subject],
    by_tiles: bool,
    ctx: &JobCtx,
) -> Result<Vec<Figure>> {
    let (wu, hu) = (area.width() as usize, area.height() as usize);
    let mut on_subjects = vec![0.0f32; wu * hu];
    for s in subjects {
        for (d, c) in on_subjects.iter_mut().zip(&s.cov) {
            *d = d.max(*c);
        }
    }
    let mut out: Vec<Figure> = Vec::new();
    for term in terms {
        let mut found = finder.find_all_in(picture, area, term, NEW_AT_MOST, new_floor(wu, hu), ctx)?;
        if by_tiles {
            found.extend(found_by_tiles(finder, picture, area, area, term, ctx)?);
        }
        for (cov, _, n) in found {
            if overlap(&cov, &on_subjects) * 2 < n && !out.iter().any(|(o, _)| overlap(&cov, o) * 2 >= n) {
                out.push((cov, n));
            }
        }
    }
    Ok(out)
}

/// The creature terms among the subjects' (people, animals), and "person" always: whom a fill
/// is checked for.
pub(super) fn creature_terms(subjects: &[Subject]) -> Vec<String> {
    let mut terms: Vec<String> = subjects.iter().map(|s| s.check_term().to_string()).filter(|t| creature(t)).collect();
    terms.push(PERSON.to_string());
    terms.sort();
    terms.dedup();
    terms
}

/// Anyone small a fill painted: each creature term of `asks`, looked for tile by tile over the
/// pixels the fill touched ([`found_by_tiles`]), every find judged as [`someone_new`] (fog and
/// whoever is in a picture, [`Pictures`], excused); the verdict with the find's bounds.
#[allow(clippy::too_many_arguments)]
pub(super) fn anyone_small(
    finder: &SubjectFinder,
    filled: &[u8],
    area: Rect,
    touched: &[f32],
    asks: &[(String, usize)],
    cut: (&str, &[f32], u64),
    behind: &[(String, Vec<f32>, u64)],
    already: &[Figure],
    fog: &dyn Fn(&[f32]) -> bool,
    pictures: &mut Pictures<'_>,
    ctx: &JobCtx,
) -> Result<Option<(String, Rect)>> {
    let (bounds, _) = crate::select_ml_cmds::coverage_bounds(touched, area);
    for (term, _) in asks.iter().filter(|(t, _)| creature(t)) {
        for (cov, b, _) in found_by_tiles(finder, filled, area, bounds, term, ctx)? {
            if let Some(why) = someone_new(term, &cov, b, touched, cut, behind, already, area.width() as usize)
                && !fog(&cov)
                && !(says_someone_new(&why) && pictures.holds(&cov, b)?)
            {
                return Ok(Some((why, b)));
            }
        }
    }
    Ok(None)
}

/// Whether a refusal is the one a hole may fall back on when every try is refused: a thing
/// painted where none was ("a table in the hole that was not there"). Never a person or an
/// animal, a blank or a shape over the hole.
pub(super) fn tolerable(why: &str) -> bool {
    why.strip_prefix("a ").and_then(|r| r.strip_suffix(" in the hole that was not there")).is_some_and(|term| !creature(term))
}

/// Whether a verdict says the fill painted a person or an animal where none was (the wordings
/// of [`fills_own_work`] and [`someone_new`]): the refusal a picture of them lifts.
pub(in crate::generate_cmds) fn says_someone_new(why: &str) -> bool {
    why.strip_prefix("a ")
        .and_then(|r| r.split_once(" painted into the hole (").or_else(|| r.split_once(" in the hole that was not there")).map(|(t, _)| t))
        .is_some_and(creature)
}

/// The framed pictures in one result, looked for once, when a person or an animal is about to be
/// refused as someone the fill painted: whoever is in one is a picture of a person, nobody's
/// work. The man in a framed photograph behind the waiter, completed by the waiter's removal and
/// so never found before any fill, refused that removal in five runs of round 10, and the framed
/// photograph beside it refused a fill behind the striped man (measured in the app 2026-11-04).
/// A framed picture the fill painted is décor, not someone in the room.
pub(super) struct Pictures<'a> {
    pub(super) finder: &'a SubjectFinder,
    pub(super) img: &'a [u8],
    pub(super) area: Rect,
    pub(super) ctx: &'a JobCtx,
    pub(super) found: Option<Vec<Rect>>,
}

impl<'a> Pictures<'a> {
    pub(super) fn new(finder: &'a SubjectFinder, img: &'a [u8], area: Rect, ctx: &'a JobCtx) -> Self {
        Self { finder, img, area, ctx, found: None }
    }

    /// Whether the detection (`cov` over the area, `bounds`) is in a picture ([`in_a_picture`]);
    /// counted on the finder's record.
    pub(super) fn holds(&mut self, cov: &[f32], bounds: Rect) -> Result<bool> {
        if self.found.is_none() {
            self.found = Some(pictures_in(self.finder, self.img, self.area, self.ctx)?);
        }
        let in_one = in_a_picture(cov, bounds, self.found.as_deref().unwrap_or(&[]), self.area);
        if in_one {
            self.finder.note_picture();
        }
        Ok(in_one)
    }
}

/// The framed pictures the detector finds in `img` ([`PICTURE_TERMS`]): each one's bounds, at
/// most [`PICTURE_MOST`] of the frame (a doorway or a window with someone in it is bigger).
pub(super) fn pictures_in(finder: &SubjectFinder, img: &[u8], area: Rect, ctx: &JobCtx) -> Result<Vec<Rect>> {
    let most = PICTURE_MOST * area.width() as f32 * area.height() as f32;
    let mut out: Vec<Rect> = Vec::new();
    for term in PICTURE_TERMS {
        ctx.check()?;
        for (_, b, _) in finder.find_in(img, area, term, PICTURES_AT_MOST, ctx)? {
            if b.width() as f32 * b.height() as f32 <= most && !out.contains(&b) {
                out.push(b);
            }
        }
    }
    Ok(out)
}

/// Whether a detection (`cov` over `area`, `bounds`) is in one of the `pictures`: [`IN_PICTURE`]
/// of its pixels inside the picture's bounds, which hold [`PICTURE_OVER`] times its own at least
/// (the figure handed back as the picture is no picture).
pub(in crate::generate_cmds) fn in_a_picture(cov: &[f32], bounds: Rect, pictures: &[Rect], area: Rect) -> bool {
    let w = area.width() as usize;
    let on = cov.iter().filter(|c| **c >= 0.5).count();
    if on == 0 || w == 0 {
        return false;
    }
    let own = bounds.width() as f32 * bounds.height() as f32;
    pictures.iter().any(|p| {
        if (p.width() as f32 * p.height() as f32) < PICTURE_OVER * own {
            return false;
        }
        let inside = cov
            .iter()
            .enumerate()
            .filter(|(k, c)| {
                let (x, y) = (area.x0 + (k % w) as i32, area.y0 + (k / w) as i32);
                **c >= 0.5 && x >= p.x0 && x < p.x1 && y >= p.y0 && y < p.y1
            })
            .count();
        inside as f32 >= IN_PICTURE * on as f32
    })
}

/// Whether a sweep's later find is a part of a subject of its kind apart from their mask:
/// smaller than them, its top below their middle and at most [`PART_GAP`] of their height under
/// their bottom, within their width widened by [`PART_SIDE`] of it each side. The woman in blue's
/// lower blouse, below the table edge that cut her mask off it, came back as a person of its own
/// once the man before her was gone, and was cut as a layer of its own at 16 layers, its fills
/// refused five times (measured in the app 2026-11-04).
pub(in crate::generate_cmds) fn part_below(find: &Subject, of: &Subject) -> bool {
    let (f, o) = (find.bounds, of.bounds);
    let (ow, oh) = (o.width() as f32, o.height() as f32);
    find.pixels < of.pixels
        && f.y0 as f32 >= o.y0 as f32 + oh / 2.0
        && (f.y0 - o.y1) as f32 <= PART_GAP * oh
        && f.x0 as f32 >= o.x0 as f32 - PART_SIDE * ow
        && f.x1 as f32 <= o.x1 as f32 + PART_SIDE * ow
}

/// Whether a term names a person or an animal (judged over the widened hole and by their
/// pixels) rather than a thing (judged on its own spot).
pub(super) fn creature(term: &str) -> bool {
    matches!(
        bare(term).to_ascii_lowercase().as_str(),
        PERSON | ANIMAL | "people" | "man" | "woman" | "child" | "boy" | "girl" | "baby" | "face" | "dog" | "cat" | "horse" | "bird" | "cow" | "sheep"
    )
}

/// Where both masks are on (at 0.5 and over).
pub(super) fn both(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| if *x >= 0.5 && *y >= 0.5 { 1.0 } else { 0.0 }).collect()
}

/// How much of what stood in `region` a fill left as it was: the share of the region's blocks
/// ([`KEPT_BLOCK`] px square, at least half of each in the region) whose mean colour `after`
/// keeps within [`KEPT_TOLERANCE`] of `before`. Blocks, not pixels, so a fill that redraws the
/// subject a little moved or softened still counts as keeping it.
pub(in crate::generate_cmds) fn kept_share(before: &[u8], after: &[u8], region: &[f32], w: usize, h: usize) -> f32 {
    let (mut blocks, mut kept) = (0u32, 0u32);
    for by in (0..h).step_by(KEPT_BLOCK) {
        for bx in (0..w).step_by(KEPT_BLOCK) {
            let (mut n, mut sum_before, mut sum_after) = (0u32, [0u64; 3], [0u64; 3]);
            for y in by..(by + KEPT_BLOCK).min(h) {
                for x in bx..(bx + KEPT_BLOCK).min(w) {
                    let k = y * w + x;
                    if region.get(k).is_none_or(|r| *r < 0.5) {
                        continue;
                    }
                    n += 1;
                    for (c, (sb, sa)) in sum_before.iter_mut().zip(sum_after.iter_mut()).enumerate() {
                        *sb += u64::from(before[k * 4 + c]);
                        *sa += u64::from(after[k * 4 + c]);
                    }
                }
            }
            if (n as usize) * 2 < KEPT_BLOCK * KEPT_BLOCK {
                continue;
            }
            blocks += 1;
            let moved = sum_before.iter().zip(&sum_after).map(|(b, a)| (*b as f32 - *a as f32).abs() / n as f32).sum::<f32>() / 3.0;
            if moved <= KEPT_TOLERANCE {
                kept += 1;
            }
        }
    }
    if blocks == 0 { 0.0 } else { kept as f32 / blocks as f32 }
}

/// The pixels of `detection` inside `region` that `subject` did not hold: what a fill added.
pub(super) fn added_in(detection: &[f32], region: &[f32], subject: &[f32]) -> u64 {
    detection.iter().zip(region).zip(subject).filter(|((d, r), s)| **d >= 0.5 && **r >= 0.5 && **s < 0.5).count() as u64
}

/// The pixels of `wanted` joined (4-neighbours) to a pixel within [`JOIN_REACH`] of `seed` (at
/// 0.5 and over): the part of a detection that continues the subject, not a patch of its own.
pub(in crate::generate_cmds) fn joined_to(wanted: &[bool], seed: &[f32], w: usize, h: usize) -> Vec<bool> {
    let mut out = vec![false; wanted.len()];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (k, want) in wanted.iter().enumerate() {
        if !want {
            continue;
        }
        let (x, y) = ((k % w) as i64, (k / w) as i64);
        let near = (-JOIN_REACH..=JOIN_REACH).any(|dy| {
            (-JOIN_REACH..=JOIN_REACH).any(|dx| {
                let (nx, ny) = (x + dx, y + dy);
                nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h && seed[ny as usize * w + nx as usize] >= 0.5
            })
        });
        if near {
            out[k] = true;
            queue.push_back(k);
        }
    }
    while let Some(k) = queue.pop_front() {
        let (x, y) = (k % w, k / w);
        let mut step = |n: usize| {
            if wanted[n] && !out[n] {
                out[n] = true;
                queue.push_back(n);
            }
        };
        if x > 0 {
            step(k - 1);
        }
        if x + 1 < w {
            step(k + 1);
        }
        if y > 0 {
            step(k - w);
        }
        if y + 1 < h {
            step(k + w);
        }
    }
    out
}

/// Why a fill is no fill, judged against the picture in the `ring` around the hole: it left the
/// hole blank where the picture has texture (the grey box handed back nearly as it was, which a
/// detector and the pixel check both pass; 6 of 8 café backgrounds, measured in the app
/// 2026-11-04), or painted a dark or light shape over it (a silhouette-shaped smear where
/// someone stood). The hole as a whole first, then by cells of [`BLANK_CELL`] px, each against
/// the ring around it: a pale box over a third of a hole with fog over the rest, and a flat
/// white quarter, passed the whole hole's averages (measured in the app 2026-11-04). `None` on a
/// plain backdrop, whose fill is rightly plain (though not another brightness), and when the fill
/// has texture and brightness like its surroundings.
pub(in crate::generate_cmds) fn blank_or_ghost(before: &[u8], after: &[u8], hole: &[f32], ring: &[f32], w: usize, h: usize) -> Option<String> {
    let step = texture_step(w, h);
    let (ring_tex, ring_luma) = texture_and_luma(before, ring, w, h, step)?;
    let (hole_tex, hole_luma) = texture_and_luma(after, hole, w, h, step)?;
    let d = hole_luma - ring_luma;
    if ring_tex < PLAIN_TEXTURE {
        return (d.abs() >= GHOST_LUMA).then(|| ghost_words(d, 1.0));
    }
    if hole_tex < BLANK_SHARE * ring_tex {
        return Some(format!("left the hole blank (texture {hole_tex:.3} against {ring_tex:.3} around it)"));
    }
    if d.abs() >= GHOST_LUMA {
        return Some(ghost_words(d, 1.0));
    }
    // By cells over the hole's bounds (the ring's width around them): a cell of the hole against
    // the ring in the cells about it, or the whole ring where none is near.
    let area = Rect::new(0, 0, w as i32, h as i32);
    let (bounds, holed) = crate::select_ml_cmds::coverage_bounds(hole, area);
    let grid = bounds.inflate(RING_WIDTH as i32).intersect(&area);
    let (gw, gh) = ((grid.width() as usize).div_ceil(BLANK_CELL).max(1), (grid.height() as usize).div_ceil(BLANK_CELL).max(1));
    let hole_cells = cell_stats(after, hole, w, h, grid, (gw, gh), step);
    let ring_cells = cell_stats(before, ring, w, h, grid, (gw, gh), step);
    let (mut blank, mut ghost, mut shade) = (0u64, 0u64, 0.0f32);
    // Each ghost cell's brightness difference and pixels, for the clusters below.
    let mut ghost_cells: Vec<Option<(f32, u32)>> = vec![None; gw * gh];
    for cy in 0..gh {
        for cx in 0..gw {
            let cell = &hole_cells[cy * gw + cx];
            let Some((tex, luma)) = cell.values() else { continue };
            let mut around = CellStats::default();
            for y in cy.saturating_sub(1)..(cy + 2).min(gh) {
                for x in cx.saturating_sub(1)..(cx + 2).min(gw) {
                    around.add(&ring_cells[y * gw + x]);
                }
            }
            let (r_tex, r_luma) = around.values().unwrap_or((ring_tex, ring_luma));
            if r_tex >= PLAIN_TEXTURE && tex < BLANK_SHARE * r_tex {
                blank += u64::from(cell.ln);
            } else if (luma - r_luma).abs() >= GHOST_LUMA {
                ghost += u64::from(cell.ln);
                shade += (luma - r_luma) * cell.ln as f32;
                ghost_cells[cy * gw + cx] = Some((luma - r_luma, cell.ln));
            }
        }
    }
    if blank as f32 >= BLANK_CELLS * holed as f32 {
        return Some(format!("left {:.0} % of the hole blank (texture {hole_tex:.3} against {ring_tex:.3} around it)", 100.0 * blank as f32 / holed as f32));
    }
    if ghost > 0 && ghost as f32 >= BLANK_CELLS * holed as f32 {
        return Some(ghost_words(shade / ghost as f32, ghost as f32 / holed as f32));
    }
    // A shape over part of the hole: ghost cells joined, [`GHOST_CLUSTER_CELLS`] or more of them
    // and [`GHOST_CLUSTER_SHARE`] of the hole (a soft dark blob where the woman's head had been
    // passed the share above in four café backgrounds; measured in the app 2026-11-04).
    if let Some((cells, pixels, d, (cw, ch))) = ghost_cluster(&ghost_cells, gw, gh)
        && cells >= GHOST_CLUSTER_CELLS
        && pixels as f32 >= GHOST_CLUSTER_SHARE * holed as f32
    {
        let shape = if d < 0.0 { "dark" } else { "light" };
        return Some(format!(
            "painted a {shape} shape over part of the hole (about {}×{} px, {:+.0} % in brightness)",
            cw * BLANK_CELL,
            ch * BLANK_CELL,
            d * 100.0
        ));
    }
    None
}

/// The largest cluster of ghost cells (joined side to side), by its pixels: its cells, its
/// pixels, its mean brightness difference and its bounds in cells.
pub(super) fn ghost_cluster(cells: &[Option<(f32, u32)>], gw: usize, gh: usize) -> Option<(usize, u64, f32, (usize, usize))> {
    let mut seen = vec![false; cells.len()];
    let mut best: Option<(usize, u64, f32, (usize, usize))> = None;
    for start in 0..cells.len() {
        if seen[start] || cells[start].is_none() {
            continue;
        }
        let (mut count, mut pixels, mut shade) = (0usize, 0u64, 0.0f32);
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(k) = stack.pop() {
            let Some((d, n)) = cells[k] else { continue };
            count += 1;
            pixels += u64::from(n);
            shade += d * n as f32;
            let (x, y) = (k % gw, k / gw);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
            let mut visit = |j: usize| {
                if !seen[j] && cells[j].is_some() {
                    seen[j] = true;
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(k - 1);
            }
            if x + 1 < gw {
                visit(k + 1);
            }
            if y > 0 {
                visit(k - gw);
            }
            if y + 1 < gh {
                visit(k + gw);
            }
        }
        if best.as_ref().is_none_or(|b| pixels > b.1) {
            best = Some((count, pixels, shade / pixels.max(1) as f32, (x1 - x0, y1 - y0)));
        }
    }
    best
}

/// "painted a dark shape over the hole (-40 % in brightness)", or over `share` of it.
pub(super) fn ghost_words(d: f32, share: f32) -> String {
    let shape = if d < 0.0 { "dark" } else { "light" };
    if share >= 0.999 {
        format!("painted a {shape} shape over the hole ({:+.0} % in brightness)", d * 100.0)
    } else {
        format!("painted a {shape} shape over {:.0} % of the hole ({:+.0} % in brightness)", share * 100.0, d * 100.0)
    }
}

/// The texture and luminance sums of a region's pixels (in one cell, or in all).
#[derive(Clone, Copy, Default)]
pub(super) struct CellStats {
    pub(super) tex: f32,
    pub(super) tn: u32,
    pub(super) lum: f32,
    pub(super) ln: u32,
}

impl CellStats {
    pub(super) fn add(&mut self, o: &CellStats) {
        self.tex += o.tex;
        self.tn += o.tn;
        self.lum += o.lum;
        self.ln += o.ln;
    }

    /// The mean texture (luminance step between neighbours) and luminance, 0..1; `None` with too
    /// few pixels to judge.
    pub(super) fn values(&self) -> Option<(f32, f32)> {
        (self.ln >= JUDGE_MIN_PIXELS && self.tn >= JUDGE_MIN_PIXELS).then(|| (self.tex / self.tn as f32, self.lum / self.ln as f32))
    }
}

/// Per cell of `grid` ([`BLANK_CELL`] px square, `(gw, gh)` of them), the sums over `region`'s
/// pixels in `img`: luminance, and the luminance step to the neighbour `step` px right and down
/// where that is in the region too.
pub(super) fn cell_stats(img: &[u8], region: &[f32], w: usize, h: usize, grid: Rect, (gw, gh): (usize, usize), step: usize) -> Vec<CellStats> {
    let luma = |k: usize| (0.299 * f32::from(img[k * 4]) + 0.587 * f32::from(img[k * 4 + 1]) + 0.114 * f32::from(img[k * 4 + 2])) / 255.0;
    let mut cells = vec![CellStats::default(); gw * gh];
    let (gx0, gy0) = (grid.x0.max(0) as usize, grid.y0.max(0) as usize);
    for y in gy0..(grid.y1.max(0) as usize).min(h) {
        for x in gx0..(grid.x1.max(0) as usize).min(w) {
            let k = y * w + x;
            if region[k] < 0.5 {
                continue;
            }
            let (cx, cy) = (((x - gx0) / BLANK_CELL).min(gw - 1), ((y - gy0) / BLANK_CELL).min(gh - 1));
            let cell = &mut cells[cy * gw + cx];
            let l = luma(k);
            cell.lum += l;
            cell.ln += 1;
            if x + step < w && region[k + step] >= 0.5 {
                cell.tex += (l - luma(k + step)).abs();
                cell.tn += 1;
            }
            if y + step < h && region[k + step * w] >= 0.5 {
                cell.tex += (l - luma(k + step * w)).abs();
                cell.tn += 1;
            }
        }
    }
    cells
}

/// The mean luminance step between pixels `step` apart in `region` (its texture) and its mean
/// luminance, 0..1; `None` for a region too small to judge.
pub(super) fn texture_and_luma(img: &[u8], region: &[f32], w: usize, h: usize, step: usize) -> Option<(f32, f32)> {
    let area = Rect::new(0, 0, w as i32, h as i32);
    let (bounds, _) = crate::select_ml_cmds::coverage_bounds(region, area);
    if bounds.is_empty() {
        return None;
    }
    let (gw, gh) = ((bounds.width() as usize).div_ceil(BLANK_CELL).max(1), (bounds.height() as usize).div_ceil(BLANK_CELL).max(1));
    let mut all = CellStats::default();
    for c in cell_stats(img, region, w, h, bounds, (gw, gh), step) {
        all.add(&c);
    }
    all.values()
}

/// How far apart the pixels compared for texture are: 1 up to [`TEXTURE_PIXELS`] a frame, more
/// on a larger picture, so a 12-megapixel ring's grain is not held against a fill generated at
/// a megapixel and scaled up.
pub(super) fn texture_step(w: usize, h: usize) -> usize {
    (((w * h) as f32 / TEXTURE_PIXELS as f32).sqrt().ceil() as usize).max(1)
}

/// What the detector is asked for on the filled background: the prompt that found the
/// subjects when it names a kind of thing ("person", the caller's prompt), else "person". The
/// generic prompts ("the main subject") find something on any picture: on a clean backdrop
/// they rejected all three fills (measured in the app 2026-11-03), and the failure this check
/// is for is a person painted back.
pub(super) fn check_prompt(found_by: &str) -> &str {
    match found_by {
        "the main subject" | "the foreground object" | "object" => PERSON,
        p => p,
    }
}

/// The detector's and the removal's word for a person.
pub(super) const PERSON: &str = "person";

#[cfg(test)]
mod check_tests {
    use super::check_prompt;

    #[test]
    fn the_fill_check_asks_for_a_kind_of_thing() {
        assert_eq!(check_prompt("the main subject"), "person");
        assert_eq!(check_prompt("the foreground object"), "person");
        assert_eq!(check_prompt("object"), "person");
        assert_eq!(check_prompt("person"), "person");
        assert_eq!(check_prompt("the lighthouse"), "the lighthouse");
    }
}
