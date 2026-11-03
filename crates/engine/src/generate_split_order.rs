//! Split into Layers' depth order: who stands in front of whom, read from the subjects' masks
//! alone. Two masks that touch share a boundary; that boundary is the silhouette of the one in
//! front (an occluder's outline bows out into the thing it hides, which is left concave around
//! it). So at each point where the masks meet, the boundary's local bend says which side is in
//! front: the side the curve bends around. A straight contact (a table's edge across a torso)
//! says nothing, and the lower bottom edge is then taken as nearer (what stands on the ground
//! nearer the camera reaches lower in the frame), then the larger.

use photocraft_geom::Rect;

/// What one subject contributes: its coverage over the canvas, its bounds and pixel count.
pub(crate) struct Shape<'a> {
    pub cov: &'a [f32],
    pub bounds: Rect,
    pub pixels: u64,
}

/// A pairwise verdict: `Some(true)` the first is in front, `Some(false)` the second, with the
/// cue's strength (0..1), `None` when the masks do not meet or the contact says nothing.
pub(crate) fn contact_cue(a: &Shape, b: &Shape, w: usize, h: usize) -> Option<(bool, f32)> {
    // Only where both bounds (a little widened) overlap can they touch.
    let zone = a.bounds.inflate(CONTACT_REACH as i32 + 1).intersect(&b.bounds.inflate(CONTACT_REACH as i32 + 1));
    if zone.is_empty() || w == 0 || h == 0 {
        return None;
    }
    let on = |cov: &[f32], x: i32, y: i32| -> bool { x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && cov[y as usize * w + x as usize] >= 0.5 };
    // The contact pixels of each mask: its pixels with the other within reach.
    let reach = CONTACT_REACH as i32;
    let mut contacts: Vec<(i32, i32, bool)> = Vec::new();
    for y in zone.y0..zone.y1 {
        for x in zone.x0..zone.x1 {
            let (in_a, in_b) = (on(a.cov, x, y), on(b.cov, x, y));
            if !in_a && !in_b {
                continue;
            }
            // Which mask this pixel is a contact pixel of: its own, when the other is near. A
            // pixel in both (soft masks overlap a little) counts for neither.
            if in_a == in_b {
                continue;
            }
            let other = if in_a { b.cov } else { a.cov };
            let near = (-reach..=reach).any(|dy| (-reach..=reach).any(|dx| on(other, x + dx, y + dy)));
            if near {
                contacts.push((x, y, in_a));
            }
        }
    }
    if contacts.len() < MIN_CONTACT {
        return None;
    }
    // The bend's radius: a share of the smaller subject's size, at least a few pixels.
    let smaller = a.bounds.width().max(a.bounds.height()).min(b.bounds.width().max(b.bounds.height()));
    let r = ((smaller as f32) * BEND_SHARE).max(MIN_BEND).round() as i32;
    // A raster of the contact pixels, for the neighbourhood search.
    let (zw, zh) = (zone.width() as usize + 2 * r as usize, zone.height() as usize + 2 * r as usize);
    let mut raster = vec![false; zw * zh];
    for (x, y, _) in &contacts {
        let (zx, zy) = ((x - zone.x0 + r) as usize, (y - zone.y0 + r) as usize);
        raster[zy * zw + zx] = true;
    }
    // At each contact pixel, the centroid of the contact pixels around it bends toward the
    // inside of the curve; the mask found a little way along that bend is the one in front.
    let (mut sum, mut n) = (0.0f64, 0u32);
    for (x, y, _) in &contacts {
        let (cx, cy) = (x - zone.x0 + r, y - zone.y0 + r);
        let (mut sx, mut sy, mut m) = (0.0f32, 0.0f32, 0u32);
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy > r * r {
                    continue;
                }
                let (px, py) = (cx + dx, cy + dy);
                if px < 0 || py < 0 || px as usize >= zw || py as usize >= zh {
                    continue;
                }
                if raster[py as usize * zw + px as usize] {
                    sx += dx as f32;
                    sy += dy as f32;
                    m += 1;
                }
            }
        }
        if m < 3 {
            continue;
        }
        let (bx, by) = (sx / m as f32, sy / m as f32);
        let len = (bx * bx + by * by).sqrt();
        if len < MIN_BEND_PX {
            continue;
        }
        // Probe halfway along the radius in the bend's direction.
        let probe = r as f32 * 0.5;
        let (qx, qy) = (*x as f32 + bx / len * probe, *y as f32 + by / len * probe);
        let (qx, qy) = (qx.round() as i32, qy.round() as i32);
        let (qa, qb) = (on(a.cov, qx, qy), on(b.cov, qx, qy));
        match (qa, qb) {
            (true, false) => sum += 1.0,
            (false, true) => sum -= 1.0,
            _ => continue,
        }
        n += 1;
    }
    if n < MIN_CONTACT as u32 {
        return None;
    }
    let mean = (sum / f64::from(n)) as f32;
    (mean.abs() >= BEND_MARGIN).then_some((mean > 0.0, mean.abs()))
}

/// Whether two masks meet (within [`CONTACT_REACH`]).
pub(crate) fn touching(a: &Shape, b: &Shape, w: usize, h: usize) -> bool {
    let zone = a.bounds.inflate(CONTACT_REACH as i32).intersect(&b.bounds.inflate(CONTACT_REACH as i32));
    if zone.is_empty() {
        return false;
    }
    let reach = CONTACT_REACH as i32;
    let on = |cov: &[f32], x: i32, y: i32| -> bool { x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && cov[y as usize * w + x as usize] >= 0.5 };
    for y in zone.y0..zone.y1 {
        for x in zone.x0..zone.x1 {
            if on(a.cov, x, y) && (-reach..=reach).any(|dy| (-reach..=reach).any(|dx| on(b.cov, x + dx, y + dy))) {
                return true;
            }
        }
    }
    false
}

/// Whether the smaller of two boxes nests in the larger (a cup on a table, a hat on a head):
/// most of it inside, and it much smaller. `Some(true)` the first is the small one.
pub(crate) fn nested(a: &Shape, b: &Shape) -> Option<bool> {
    let area = |r: Rect| f64::from(r.width()) * f64::from(r.height());
    let (small, large, first_small) = if area(a.bounds) <= area(b.bounds) { (a, b, true) } else { (b, a, false) };
    let (sa, la) = (area(small.bounds), area(large.bounds));
    if sa <= 0.0 || la <= 0.0 || sa > NEST_MAX_SHARE * la {
        return None;
    }
    let inter = small.bounds.intersect(&large.bounds);
    let inside = if inter.is_empty() { 0.0 } else { area(inter) };
    (inside >= NEST_MIN_INSIDE * sa).then_some(first_small)
}

/// Which of a pair is nearer when the contact says nothing: the lower bottom edge, then the
/// larger. `true` when the first is.
pub(crate) fn nearer_by_edge(a: &Shape, b: &Shape) -> bool {
    match a.bounds.y1.cmp(&b.bounds.y1) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => a.pixels >= b.pixels,
    }
}

/// The subjects' indices front to back. Every pair that touches gets the contact cue, else
/// (touching, or nested boxes) the fallbacks; the verdicts form an "in front of" graph whose
/// weakest edges go until it is acyclic, and a topological order with the edge rule as the
/// tie-break comes out.
pub(crate) fn front_to_back(shapes: &[Shape], w: usize, h: usize) -> Vec<usize> {
    let n = shapes.len();
    // (front, back, weight)
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            let (a, b) = (&shapes[i], &shapes[j]);
            if touching(a, b, w, h) {
                match contact_cue(a, b, w, h) {
                    Some((true, s)) => edges.push((i, j, 1.0 + s)),
                    Some((false, s)) => edges.push((j, i, 1.0 + s)),
                    None => {
                        if nearer_by_edge(a, b) {
                            edges.push((i, j, EDGE_RULE_WEIGHT));
                        } else {
                            edges.push((j, i, EDGE_RULE_WEIGHT));
                        }
                    }
                }
            } else if let Some(first_small) = nested(a, b) {
                if first_small {
                    edges.push((i, j, NEST_WEIGHT));
                } else {
                    edges.push((j, i, NEST_WEIGHT));
                }
            }
        }
    }
    topological(n, edges, |i, j| nearer_by_edge(&shapes[i], &shapes[j]))
}

/// A topological order of `n` nodes over `edges` (front, back, weight): nodes with nothing in
/// front of them first; among several, `nearer(i, j)` decides; a cycle loses its weakest edge.
pub(crate) fn topological(n: usize, mut edges: Vec<(usize, usize, f32)>, nearer: impl Fn(usize, usize) -> bool) -> Vec<usize> {
    let mut out = Vec::with_capacity(n);
    let mut left: Vec<bool> = vec![true; n];
    while out.len() < n {
        // Nodes still left with no edge from another left node in front of them.
        let sources: Vec<usize> = (0..n).filter(|&i| left[i] && !edges.iter().any(|(f, b, _)| *b == i && left[*f])).collect();
        if sources.is_empty() {
            // A cycle: drop the weakest edge among the left nodes.
            let Some(k) = (0..edges.len()).filter(|&k| left[edges[k].0] && left[edges[k].1]).min_by(|&x, &y| edges[x].2.total_cmp(&edges[y].2)) else {
                break;
            };
            edges.swap_remove(k);
            continue;
        }
        let mut best = sources[0];
        for &s in &sources[1..] {
            if nearer(s, best) {
                best = s;
            }
        }
        left[best] = false;
        out.push(best);
    }
    // Anything not placed (should not happen): as found.
    for (i, l) in left.iter().enumerate() {
        if *l {
            out.push(i);
        }
    }
    out
}

/// How near (in pixels) two masks must come to count as touching.
const CONTACT_REACH: usize = 2;
/// The fewest contact pixels for a verdict.
const MIN_CONTACT: usize = 12;
/// The bend's radius as a share of the smaller subject's larger side (a gentle curve shows
/// its bend only over a long enough arc: a tenth of a head's circumference bends a pixel)...
const BEND_SHARE: f32 = 0.15;
/// ...and at least this many pixels.
const MIN_BEND: f32 = 6.0;
/// The least displacement of the contact centroid that counts as a bend (pixels).
const MIN_BEND_PX: f32 = 0.25;
/// The least mean verdict (−1..1) over the contact pixels that decides.
const BEND_MARGIN: f32 = 0.15;
/// A nested box: the small one at most this share of the large one's box...
const NEST_MAX_SHARE: f64 = 0.3;
/// ...and at least this share of it inside the large one.
const NEST_MIN_INSIDE: f64 = 0.8;
/// The weight of a fallback verdict (the edge rule, a nested box) against a contact cue's 1+.
const EDGE_RULE_WEIGHT: f32 = 0.2;
const NEST_WEIGHT: f32 = 0.3;

#[cfg(test)]
mod tests {
    use super::*;

    fn disc(w: usize, h: usize, cx: f32, cy: f32, r: f32) -> Vec<f32> {
        (0..w * h).map(|k| if ((k % w) as f32 - cx).powi(2) + ((k / w) as f32 - cy).powi(2) <= r * r { 1.0 } else { 0.0 }).collect()
    }

    fn rect(w: usize, h: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<f32> {
        (0..w * h).map(|k| if (x0..x1).contains(&(k % w)) && (y0..y1).contains(&(k / w)) { 1.0 } else { 0.0 }).collect()
    }

    fn minus(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| if *y >= 0.5 { 0.0 } else { *x }).collect()
    }

    fn shape(cov: &[f32], w: usize) -> Shape<'_> {
        let h = cov.len() / w;
        let (mut x0, mut y0, mut x1, mut y1, mut n) = (w as i32, h as i32, 0, 0, 0u64);
        for (k, c) in cov.iter().enumerate() {
            if *c >= 0.5 {
                let (x, y) = ((k % w) as i32, (k / w) as i32);
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
                n += 1;
            }
        }
        Shape { cov, bounds: Rect::new(x0, y0, x1, y1), pixels: n }
    }

    #[test]
    fn a_disc_biting_a_rectangle_is_in_front_of_it() {
        let (w, h) = (200usize, 200usize);
        let d = disc(w, h, 100.0, 100.0, 40.0);
        let r = minus(&rect(w, h, 20, 60, 180, 190), &d);
        let (ds, rs) = (shape(&d, w), shape(&r, w));
        let cue = contact_cue(&ds, &rs, w, h).expect("they touch along an arc");
        assert!(cue.0, "the disc is in front: {cue:?}");
        let cue = contact_cue(&rs, &ds, w, h).expect("symmetric");
        assert!(!cue.0, "the rectangle is behind: {cue:?}");
        // The order puts the disc first even though the rectangle reaches lower and is larger.
        assert_eq!(front_to_back(&[rs, ds], w, h), vec![1, 0]);
    }

    #[test]
    fn a_straight_contact_says_nothing_and_the_lower_one_is_nearer() {
        let (w, h) = (120usize, 160usize);
        let top = rect(w, h, 20, 10, 100, 80);
        let bottom = rect(w, h, 10, 80, 110, 150);
        let (ts, bs) = (shape(&top, w), shape(&bottom, w));
        assert!(contact_cue(&ts, &bs, w, h).is_none(), "a straight edge has no bend");
        assert!(touching(&ts, &bs, w, h));
        assert_eq!(front_to_back(&[ts, bs], w, h), vec![1, 0], "the lower one first");
    }

    #[test]
    fn separate_shapes_order_by_bottom_edge_and_nested_boxes_put_the_small_one_in_front() {
        let (w, h) = (200usize, 200usize);
        let far = disc(w, h, 40.0, 40.0, 20.0);
        let near = disc(w, h, 150.0, 150.0, 20.0);
        let (fs, ns) = (shape(&far, w), shape(&near, w));
        assert!(!touching(&fs, &ns, w, h));
        assert_eq!(front_to_back(&[fs, ns], w, h), vec![1, 0]);
        // A small box inside a large one that does not touch it: the small one is in front.
        let table = minus(&rect(w, h, 10, 100, 190, 200), &rect(w, h, 80, 110, 120, 150));
        let cup = rect(w, h, 85, 115, 115, 145);
        let (ts, cs) = (shape(&table, w), shape(&cup, w));
        assert!(!touching(&ts, &cs, w, h));
        assert_eq!(nested(&ts, &cs), Some(false));
        assert_eq!(front_to_back(&[ts, cs], w, h), vec![1, 0], "the cup before the table");
    }

    #[test]
    fn a_cycle_loses_its_weakest_edge() {
        let order = topological(3, vec![(0, 1, 1.5), (1, 2, 1.2), (2, 0, 0.2)], |i, j| i < j);
        assert_eq!(order, vec![0, 1, 2]);
        let order = topological(3, Vec::new(), |i, j| i > j);
        assert_eq!(order, vec![2, 1, 0], "no edges: the tie-break alone");
    }
}
