//! Select Subject's pick: of everything the detector found for "the main subject" (and, when
//! that is weak, for the kinds of thing a picture is usually of), which is the subject. The
//! detector's own first answer is right most of the time, but on a picture with a small
//! figure in the distance and a large thing up front it can name the small one (5090 test,
//! 2026-11-23: one far man in a crowd, a toy figure behind a motorcycle). So each candidate is
//! weighed by what a person means by "the subject": the detector is sure of it, it takes a
//! fair share of the picture, it is near the middle, and it is not the backdrop. Candidates
//! of the same kind and standing close to the winner's size join it (a group on a sofa is one
//! subject). One rule, the same numbers for every picture.

use photocraft_geom::Rect;

/// A candidate's share of the picture counts up to this much: beyond it, bigger is not more
/// of a subject.
const SIZE_CAP: f32 = 0.10;
/// A candidate covering more than this share of the picture is taken for the backdrop.
const BACKDROP_SHARE: f32 = 0.80;
/// What a backdrop keeps of its weight: next to nothing, so it is chosen only when it is all
/// there is.
const BACKDROP_WEIGHT: f32 = 0.05;
/// A candidate reaching within this share of the picture's width or height of an edge
/// touches it.
const EDGE_REACH: f64 = 0.01;
/// A find of a catch-all prompt touching this many of the picture's edges is the ground, a
/// wall or a table top (5090 test, 2026-11-24: the tundra under a lone hiker, 57 % of the
/// picture, named "the foreground object"). A person's bust touches three edges too, so what
/// "the main subject", "person" or "animal" found is not judged this way.
const BACKDROP_EDGES: usize = 3;
/// How much the far corner costs against the middle of the picture (0: nothing, 1: all).
const CENTRE_PULL: f32 = 0.5;
/// The detector's score when the worker sent none (an older worker).
const UNKNOWN_SCORE: f32 = 0.75;
/// Two candidates overlapping this much (intersection over union) are one thing found twice.
const SAME_THING_IOU: f32 = 0.6;
/// A candidate of the winner's kind joins it when it has this share of the winner's pixels
/// and this share of its weight.
const GROUP_PIXELS: f32 = 0.30;
const GROUP_WEIGHT: f32 = 0.50;
/// Below this share of the picture, the best "main subject" answer is weak and the other
/// kinds are asked for too.
pub(crate) const WEAK_SHARE: f32 = 0.03;

/// One thing the detector found.
#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    /// Which prompt found it (an index into the prompts asked) and what that prompt weighs.
    pub prompt: usize,
    pub prompt_weight: f32,
    /// A catch-all prompt found it ("the foreground object", "object"): such prompts name
    /// surfaces as readily as things.
    pub catch_all: bool,
    /// The detector's score, 0..1, when the worker sent it.
    pub score: Option<f32>,
    /// Canvas coverage, its bounds and pixel count.
    pub cov: Vec<f32>,
    pub bounds: Rect,
    pub pixels: u64,
}

/// A candidate's share of a `w` × `h` picture.
pub(crate) fn share(c: &Candidate, w: usize, h: usize) -> f32 {
    c.pixels as f32 / (w.max(1) * h.max(1)) as f32
}

/// How much of a subject a candidate is, 0..1: see the module's rule.
pub(crate) fn weight(c: &Candidate, w: usize, h: usize) -> f32 {
    let share = share(c, w, h);
    let size = (share.min(SIZE_CAP) / SIZE_CAP).sqrt();
    let (rx, ry) = (w.max(1) as f64 * EDGE_REACH, h.max(1) as f64 * EDGE_REACH);
    let edges = [f64::from(c.bounds.x0) <= rx, f64::from(c.bounds.y0) <= ry, f64::from(c.bounds.x1) >= w as f64 - rx, f64::from(c.bounds.y1) >= h as f64 - ry];
    let touching = edges.iter().filter(|e| **e).count();
    let backdrop = if share > BACKDROP_SHARE || (c.catch_all && touching >= BACKDROP_EDGES) { BACKDROP_WEIGHT } else { 1.0 };
    let (cx, cy) = (f64::from(c.bounds.x0 + c.bounds.x1) * 0.5, f64::from(c.bounds.y0 + c.bounds.y1) * 0.5);
    let (hw, hh) = (w.max(1) as f64 * 0.5, h.max(1) as f64 * 0.5);
    let off = (((cx - hw) / hw).powi(2) + ((cy - hh) / hh).powi(2)).sqrt() / std::f64::consts::SQRT_2;
    let centre = 1.0 - CENTRE_PULL * off.clamp(0.0, 1.0) as f32;
    c.score.unwrap_or(UNKNOWN_SCORE).clamp(0.0, 1.0) * size * centre * backdrop * c.prompt_weight
}

fn iou(a: &Candidate, b: &Candidate) -> f32 {
    if a.bounds.intersect(&b.bounds).is_empty() {
        return 0.0;
    }
    let (mut inter, mut union) = (0u64, 0u64);
    for (x, y) in a.cov.iter().zip(&b.cov) {
        let (x, y) = (*x >= 0.5, *y >= 0.5);
        inter += u64::from(x && y);
        union += u64::from(x || y);
    }
    if union == 0 { 0.0 } else { inter as f32 / union as f32 }
}

/// The candidates that make up the subject, the winner first, and every candidate's weight
/// (0 for one that is another found twice). Empty in, empty out.
pub(crate) fn choose(cands: &[Candidate], w: usize, h: usize) -> (Vec<usize>, Vec<f32>) {
    let mut weights: Vec<f32> = cands.iter().map(|c| weight(c, w, h)).collect();
    // One thing found twice (by two prompts, or twice by one) counts once, at its best weight.
    for i in 0..cands.len() {
        for j in i + 1..cands.len() {
            if weights[i] > 0.0 && weights[j] > 0.0 && iou(&cands[i], &cands[j]) >= SAME_THING_IOU {
                if weights[j] > weights[i] {
                    weights[i] = 0.0;
                } else {
                    weights[j] = 0.0;
                }
            }
        }
    }
    let Some(best) = (0..cands.len()).filter(|i| weights[*i] > 0.0).max_by(|a, b| weights[*a].total_cmp(&weights[*b])) else {
        return (Vec::new(), weights);
    };
    let mut chosen = vec![best];
    for i in 0..cands.len() {
        let same_kind = cands[i].prompt == cands[best].prompt;
        let near_size = cands[i].pixels as f32 >= GROUP_PIXELS * cands[best].pixels as f32;
        if i != best && same_kind && near_size && weights[i] >= GROUP_WEIGHT * weights[best] {
            chosen.push(i);
        }
    }
    (chosen, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 100;
    const H: usize = 100;

    fn cand(prompt: usize, score: f32, r: (i32, i32, i32, i32)) -> Candidate {
        let bounds = Rect::new(r.0, r.1, r.2, r.3);
        let mut cov = vec![0.0f32; W * H];
        for y in r.1..r.3 {
            for x in r.0..r.2 {
                cov[y as usize * W + x as usize] = 1.0;
            }
        }
        let pixels = ((r.2 - r.0) * (r.3 - r.1)) as u64;
        Candidate { prompt, prompt_weight: 1.0, catch_all: false, score: Some(score), cov, bounds, pixels }
    }

    #[test]
    fn a_small_far_figure_loses_to_the_large_thing_in_front() {
        // The detector is surer of the small figure, as on the motorcycle picture.
        let figure = cand(0, 0.92, (70, 20, 78, 32));
        let bike = cand(0, 0.70, (15, 35, 70, 90));
        let (chosen, _) = choose(&[figure, bike], W, H);
        assert_eq!(chosen, vec![1]);
    }

    #[test]
    fn the_detectors_first_answer_stands_between_fair_sized_things() {
        let first = cand(0, 0.9, (30, 30, 70, 70));
        let second = cand(1, 0.6, (5, 5, 45, 45));
        let (chosen, _) = choose(&[first, second], W, H);
        assert_eq!(chosen, vec![0]);
    }

    #[test]
    fn a_group_of_the_same_kind_is_one_subject() {
        let a = cand(0, 0.9, (10, 30, 35, 90));
        let b = cand(0, 0.85, (38, 30, 62, 90));
        let c = cand(0, 0.8, (65, 30, 90, 90));
        let speck = cand(0, 0.8, (2, 2, 5, 5));
        let (mut chosen, _) = choose(&[a, b, c, speck], W, H);
        chosen.sort_unstable();
        assert_eq!(chosen, vec![0, 1, 2]);
    }

    #[test]
    fn another_kind_does_not_join_the_winner() {
        let person = cand(0, 0.9, (30, 20, 70, 95));
        let sofa = cand(1, 0.85, (5, 40, 95, 98));
        let (chosen, _) = choose(&[person, sofa], W, H);
        assert_eq!(chosen.len(), 1);
    }

    #[test]
    fn the_same_thing_found_twice_counts_once() {
        let once = cand(0, 0.9, (30, 30, 70, 70));
        let again = cand(1, 0.8, (31, 30, 70, 70));
        let (chosen, weights) = choose(&[once, again], W, H);
        assert_eq!(chosen, vec![0]);
        assert_eq!(weights[1], 0.0);
    }

    #[test]
    fn the_backdrop_is_not_the_subject() {
        let wall = cand(0, 0.95, (0, 0, 100, 95));
        let vase = cand(0, 0.7, (40, 40, 60, 80));
        let (chosen, _) = choose(&[wall, vase], W, H);
        assert_eq!(chosen, vec![1]);
    }

    #[test]
    fn the_ground_a_catch_all_prompt_names_does_not_beat_a_small_sure_subject() {
        // A lone hiker, a thousandth of the picture, and the tundra under them: three edges,
        // over half the picture, named by "the foreground object".
        let hiker = cand(0, 0.90, (48, 40, 51, 44));
        let mut tundra = cand(1, 0.79, (0, 43, 100, 100));
        tundra.prompt_weight = 0.9;
        tundra.catch_all = true;
        let (chosen, _) = choose(&[hiker.clone(), tundra.clone()], W, H);
        assert_eq!(chosen, vec![0]);
        // A bust touching three edges, found as the main subject, is the subject.
        let bust = cand(0, 0.9, (0, 20, 100, 100));
        let hat = cand(0, 0.8, (35, 20, 65, 35));
        assert_eq!(choose(&[bust, hat], W, H).0, vec![0]);
        // A large thing a catch-all prompt names that stands clear of the edges still wins.
        let mut car = cand(1, 0.79, (10, 45, 90, 95));
        car.prompt_weight = 0.9;
        car.catch_all = true;
        assert_eq!(choose(&[hiker, car], W, H).0, vec![1]);
    }

    #[test]
    fn a_lone_small_subject_is_still_chosen_and_an_older_worker_sends_no_score() {
        let mut bird = cand(0, 0.0, (48, 48, 52, 52));
        bird.score = None;
        assert!(share(&bird, W, H) < WEAK_SHARE);
        let (chosen, _) = choose(&[bird], W, H);
        assert_eq!(chosen, vec![0]);
        assert!(choose(&[], W, H).0.is_empty());
    }
}
