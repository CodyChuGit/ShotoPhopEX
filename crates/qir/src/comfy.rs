//! ComfyUI's sampling conventions, so a `--comfy` run reproduces a ComfyUI picture seed for seed
//! (plan Phase 2 parity). Three things differ between the diffusers pipeline this crate follows
//! and ComfyUI's `KSampler` over `ModelSamplingDiscreteFlow`:
//!
//! 1. **Noise.** ComfyUI draws it on the CPU with `torch.manual_seed(seed)` then
//!    `torch.randn([1, 64, H/16, W/16], float32)`: PyTorch's MT19937 and its 16-wide Box–Muller
//!    fill ([`torch_randn`]), in channel-major order ([`noise_packed`] reorders it to this
//!    crate's `(H·W, 64)` token layout).
//! 2. **Schedule.** The `simple` scheduler over the model's flow sigma table ([`FlowSchedule`]):
//!    Qwen-Image-2.1 is a `FLUX`-type model in ComfyUI, so its table is `ModelSamplingFlux`'s,
//!    10 000 entries of the exponential time shift `σ(t) = e^s / (e^s + 1/t − 1)` with `shift
//!    0.69` ([`Shift::Flux`]); other flow models use `ModelSamplingDiscreteFlow`'s 1000-entry
//!    SNR shift `σ(t) = s·t / (1 + (s − 1)·t)` ([`Shift::Snr`]). A trailing 0, and the Euler
//!    step `x += (σ_next − σ)·v`.
//! 3. **Timestep rounding.** The DiT gets `t = bf16(bf16(σ·1000) / 1000)` ([`comfy_timestep`]).
//!
//! The latent space is the same on both sides: ComfyUI's `QwenImage21` latent format normalises
//! with the VAE's per-channel statistics (`process_in`/`process_out`), as the diffusers pipeline
//! and this crate's VAE do.
//!
//! The prompt template and the system-turn drop already match (`text_encoder::prompt`, and
//! ComfyUI's `qwen_image21.py` drops everything before the second `<|im_start|>`).
//!
//! Every constant here was checked against ComfyUI's own python (torch 2.14, 2026-10-19); the
//! tests carry the numbers.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UMASK: u32 = 0x8000_0000;
const LMASK: u32 = 0x7fff_ffff;

/// PyTorch's CPU generator engine (`at::mt19937`), seeded the way `torch.manual_seed` does.
pub struct Mt19937 {
    state: [u32; N],
    left: usize,
    next: usize,
}

impl Mt19937 {
    pub fn new(seed: u64) -> Self {
        let mut state = [0u32; N];
        state[0] = (seed & 0xffff_ffff) as u32;
        for j in 1..N {
            let prev = state[j - 1];
            state[j] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(j as u32);
        }
        Self {
            state,
            left: 1,
            next: 0,
        }
    }

    fn next_state(&mut self) {
        self.left = N;
        self.next = 0;
        for j in 0..N {
            let y = (self.state[j] & UMASK) | (self.state[(j + 1) % N] & LMASK);
            let mag = if y & 1 == 1 { MATRIX_A } else { 0 };
            self.state[j] = self.state[(j + M) % N] ^ (y >> 1) ^ mag;
        }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.left -= 1;
        if self.left == 0 {
            self.next_state();
        }
        let mut y = self.state[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `at::uniform_real_distribution<float>(0, 1)`: the low 24 bits over 2^24.
    pub fn uniform_f32(&mut self) -> f32 {
        (self.next_u32() & 0x00ff_ffff) as f32 / 16_777_216.0
    }
}

/// `torch.randn(n, dtype=float32, device="cpu")` after `torch.manual_seed(seed)`, for `n ≥ 16`
/// (the vectorised `normal_fill`: `n` uniforms, Box–Muller over 16-wide chunks, the last 16
/// redrawn when `n % 16 != 0`). Smaller tensors take torch's scalar path, not reproduced here.
pub fn torch_randn(seed: u64, n: usize) -> Result<Vec<f32>, String> {
    if n < 16 {
        return Err(format!("torch_randn reproduces tensors of 16+ elements, {n} asked"));
    }
    let mut g = Mt19937::new(seed);
    let mut data: Vec<f32> = (0..n).map(|_| g.uniform_f32()).collect();
    let mut i = 0;
    while i + 16 <= n {
        normal_fill_16(&mut data[i..i + 16]);
        i += 16;
    }
    if !n.is_multiple_of(16) {
        let tail = &mut data[n - 16..];
        for v in tail.iter_mut() {
            *v = g.uniform_f32();
        }
        normal_fill_16(tail);
    }
    Ok(data)
}

/// torch's `normal_fill_16` (mean 0, std 1): `u1 = 1 − d[j]`, `u2 = d[j+8]`,
/// `r = sqrt(−2·log(u1))` in f32, `θ = 2π·u2` and the cos/sin in f64.
fn normal_fill_16(d: &mut [f32]) {
    for j in 0..8 {
        let u1 = 1.0f32 - d[j];
        let u2 = d[j + 8];
        let radius = (-2.0f32 * u1.ln()).sqrt();
        let theta = 2.0f64 * std::f64::consts::PI * u2 as f64;
        d[j] = (radius as f64 * theta.cos()) as f32;
        d[j + 8] = (radius as f64 * theta.sin()) as f32;
    }
}

/// ComfyUI's noise for a `[1, 64, h, w]` latent, reordered to this crate's packed `(h·w, 64)`
/// token layout (token `y·w + x`, channel `c` ← torch's `c·h·w + y·w + x`).
pub fn noise_packed(seed: u64, h: usize, w: usize) -> Result<Vec<f32>, String> {
    const C: usize = 64;
    let hw = h * w;
    let chw = torch_randn(seed, C * hw)?;
    let mut packed = vec![0f32; C * hw];
    for c in 0..C {
        for p in 0..hw {
            packed[p * C + c] = chw[c * hw + p];
        }
    }
    Ok(packed)
}

/// `time_snr_shift(alpha, t)` in f32, as `ModelSamplingDiscreteFlow` builds its table.
fn time_snr_shift(alpha: f32, t: f32) -> f32 {
    if alpha == 1.0 {
        return t;
    }
    alpha * t / (1.0 + (alpha as f64 - 1.0) as f32 * t)
}

/// `flux_time_shift(mu, 1.0, t)` in f32, as `ModelSamplingFlux` builds its table:
/// `e^mu / (e^mu + (1/t − 1))`.
fn flux_time_shift(mu: f32, t: f32) -> f32 {
    let e = (mu as f64).exp() as f32;
    e / (e + (1.0 / t - 1.0))
}

/// How a ComfyUI model type builds its sigma table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shift {
    /// `ModelSamplingFlux` (model type `FLUX`, what Qwen-Image-2.1 is): the exponential shift
    /// over `timesteps` entries (10 000).
    Flux { shift: f32, timesteps: usize },
    /// `ModelSamplingDiscreteFlow` (model type `FLOW`): the SNR shift with a `multiplier` over
    /// `timesteps` entries (1000).
    Snr { shift: f32, multiplier: f32, timesteps: usize },
}

/// The `simple` scheduler over a model's sigma table.
#[derive(Clone, Debug)]
pub struct FlowSchedule {
    /// `steps + 1` sigmas, decreasing, trailing 0.
    pub sigmas: Vec<f32>,
}

impl FlowSchedule {
    /// Qwen-Image-2.1's model sampling in ComfyUI: `ModelSamplingFlux`, `shift 0.69`.
    pub const QWEN_IMAGE_21: Shift = Shift::Flux { shift: 0.69, timesteps: 10_000 };
    /// Krea 2's: a `FLUX`-type model too, `sampling_settings.shift = 1.15` (ComfyUI's
    /// `supported_models.Krea2`; read off the running model's `model_sampling` 2026-10-20).
    pub const KREA2: Shift = Shift::Flux { shift: 1.15, timesteps: 10_000 };

    /// The sigma at every training timestep `1..=timesteps`, computed in f32 like the torch
    /// buffer.
    pub fn model_sigmas(shift: Shift) -> Vec<f32> {
        match shift {
            Shift::Flux { shift, timesteps } => (1..=timesteps).map(|i| flux_time_shift(shift, i as f32 / timesteps as f32)).collect(),
            Shift::Snr { shift, multiplier, timesteps } => (1..=timesteps)
                .map(|i| {
                    let t = (i as f32 / timesteps as f32) * multiplier;
                    time_snr_shift(shift, t / multiplier)
                })
                .collect(),
        }
    }

    /// `comfy.samplers.simple_scheduler(model_sampling, steps)`.
    pub fn simple(steps: usize, shift: Shift) -> Result<Self, String> {
        if steps == 0 {
            return Err("a schedule needs at least one step".into());
        }
        let table = Self::model_sigmas(shift);
        let ss = table.len() as f64 / steps as f64;
        let mut sigmas: Vec<f32> = (0..steps)
            .map(|x| {
                let back = 1 + (x as f64 * ss) as usize;
                table[table.len() - back]
            })
            .collect();
        sigmas.push(0.0);
        Ok(Self { sigmas })
    }

    pub fn steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    /// The sigma the step `i` denoises at (what `KSampler` hands the model as its timestep,
    /// `multiplier 1.0`).
    pub fn sigma(&self, i: usize) -> f32 {
        self.sigmas[i]
    }

    /// Euler step delta `σ_{i+1} − σ_i`.
    pub fn dt(&self, i: usize) -> f32 {
        self.sigmas[i + 1] - self.sigmas[i]
    }
}

/// Round to the nearest bf16 (ties to even), returned as f32.
pub fn bf16_round(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    let r = b.wrapping_add(0x7fff + lsb);
    f32::from_bits(r & 0xffff_0000)
}

/// The timestep ComfyUI's Qwen-Image-2.1 DiT embeds for a sigma (`multiplier 1.0`, compute
/// dtype bf16): `((t·1000).to(bf16) / 1000).to(bf16)`. The DiT multiplies by 1000 again in
/// f32 for the sinusoidal embedding, as this crate's `TimestepEmbed` does.
pub fn comfy_timestep(sigma: f32) -> f32 {
    bf16_round(bf16_round(sigma * 1000.0) / 1000.0)
}

#[cfg(test)]
#[allow(clippy::excessive_precision)] // the constants are python's f64 prints of torch's f32 values
mod tests {
    use super::*;

    // From ComfyUI's python (torch 2.14.0+cu130): torch.manual_seed(s); torch.randn(shape).
    const S7_HEAD: [f32; 8] = [
        -0.8201345205307007,
        0.39563116431236267,
        0.8989084959030151,
        -1.3884038925170898,
        -0.16699601709842682,
        0.2851499617099762,
        -0.641091525554657,
        -0.8936554193496704,
    ];
    const S7_MID: [f32; 8] = [
        1.3169000148773193,
        -0.7144812941551208,
        -0.5096297264099121,
        -0.8810493350028992,
        0.10512683540582657,
        1.3846077919006348,
        0.00198268867097795,
        -0.07893617451190948,
    ];
    const S7_TAIL: [f32; 8] = [
        0.41666659712791443,
        -1.7636547088623047,
        0.34453269839286804,
        0.4359288513660431,
        0.15902310609817505,
        1.0355253219604492,
        1.3987053632736206,
        -0.15069834887981415,
    ];
    const S3_HEAD: [f32; 4] = [
        -0.07664429396390915,
        0.35988152027130127,
        -0.7820168137550354,
        0.07152773439884186,
    ];
    const S3_TAIL: [f32; 4] = [
        0.9464710354804993,
        2.6077301502227783,
        0.036344464868307114,
        -0.7857550978660583,
    ];

    fn close(a: &[f32], b: &[f32]) {
        let max = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(max <= 2e-6, "max abs diff {max}: {a:?} vs {b:?}");
    }

    #[test]
    fn randn_matches_torch_seed_7() {
        let n = 64 * 64 * 64;
        let v = torch_randn(7, n).unwrap();
        close(&v[..8], &S7_HEAD);
        close(&v[100_000..100_008], &S7_MID);
        close(&v[n - 8..], &S7_TAIL);
        let sum: f64 = v.iter().map(|x| *x as f64).sum();
        let sumsq: f64 = v.iter().map(|x| (*x as f64).powi(2)).sum();
        assert!((sum - 551.6456314527445).abs() < 1e-2, "sum {sum}");
        assert!((sumsq - 261_997.774_446_639_64).abs() < 1.0, "sumsq {sumsq}");
    }

    #[test]
    fn randn_matches_torch_seed_3_non_square() {
        // [1, 64, 48, 84]: 258048 elements, a multiple of 16; the tail is torch's last chunk.
        let n = 64 * 48 * 84;
        let v = torch_randn(3, n).unwrap();
        close(&v[..4], &S3_HEAD);
        close(&v[n - 4..], &S3_TAIL);
        let sum: f64 = v.iter().map(|x| *x as f64).sum();
        assert!((sum - 706.0164727073129).abs() < 1e-2, "sum {sum}");
    }

    #[test]
    fn randn_redraws_the_last_chunk_when_not_a_multiple_of_16() {
        // Same generator: element 0 is the only one outside the redrawn last 16 of 17.
        let a = torch_randn(1, 32).unwrap();
        let b = torch_randn(1, 17).unwrap();
        assert_eq!(a[0], b[0]);
        assert_ne!(a[1], b[1]);
        assert_ne!(a[16], b[16]);
        assert!(torch_randn(1, 15).is_err());
    }

    #[test]
    fn noise_packed_transposes_channels() {
        let (h, w) = (3, 5);
        let chw = torch_randn(9, 64 * h * w).unwrap();
        let p = noise_packed(9, h, w).unwrap();
        for c in 0..64 {
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(p[(y * w + x) * 64 + c], chw[c * h * w + y * w + x]);
                }
            }
        }
    }

    #[test]
    fn model_sigmas_match_comfy_tables() {
        // ModelSamplingDiscreteFlow(shift 0.69, multiplier 1.0): 1000 entries.
        let t = FlowSchedule::model_sigmas(Shift::Snr { shift: 0.69, multiplier: 1.0, timesteps: 1000 });
        close(&t[..4], &[0.000690213986672461, 0.0013808561488986015, 0.002071926835924387, 0.00276342686265707]);
        close(&t[996..], &[0.9956579804420471, 0.9971041679382324, 0.9985513687133789, 1.0]);
        // ModelSamplingFlux(shift 0.69): 10 000 entries, the table the Qwen-Image-2.1 model object holds.
        let t = FlowSchedule::model_sigmas(FlowSchedule::QWEN_IMAGE_21);
        assert_eq!(t.len(), 10_000);
        close(&t[..3], &[0.00019935173622798175, 0.00039866389124654233, 0.0005979364505037665]);
        close(&t[9997..], &[0.999899685382843, 0.9999498128890991, 1.0]);
    }

    #[test]
    fn simple_schedule_matches_comfy() {
        // comfy.samplers.simple_scheduler over the model's ModelSamplingFlux, as the KSampler runs it.
        let q = FlowSchedule::QWEN_IMAGE_21;
        let s = FlowSchedule::simple(25, q).unwrap();
        assert_eq!(s.steps(), 25);
        close(&s.sigmas[..4], &[1.0, 0.9795287847518921, 0.9582074880599976, 0.935981810092926]);
        close(&s.sigmas[22..], &[0.21375630795955658, 0.14775140583515167, 0.07669990509748459, 0.0]);
        let s = FlowSchedule::simple(8, q).unwrap();
        close(&s.sigmas, &[1.0, 0.9331371784210205, 0.8567571043968201, 0.7686715722084045, 0.6659669280052185, 0.5446740984916687, 0.39924493432044983, 0.2216787487268448, 0.0]);
        // 30 steps: ss = 333.33.., int() truncation picks the rows.
        let s = FlowSchedule::simple(30, q).unwrap();
        close(&s.sigmas[..3], &[1.0, 0.9830156564712524, 0.9654479622840881]);
        close(&s.sigmas[28..], &[0.12471463531255722, 0.06445096433162689, 0.0]);
        let s = FlowSchedule::simple(1, q).unwrap();
        close(&s.sigmas, &[1.0, 0.0]);
        // Krea 2 Turbo's 8 steps, as `comfy.samplers.calculate_sigmas(model_sampling, "simple", 8)`
        // printed them off the loaded model (shift 1.15).
        let s = FlowSchedule::simple(8, FlowSchedule::KREA2).unwrap();
        close(&s.sigmas, &[1.0, 0.956724, 0.904531, 0.840349, 0.759511, 0.654567, 0.512844, 0.310901, 0.0]);
        // The SNR table, for the models that use it.
        let s = FlowSchedule::simple(4, Shift::Snr { shift: 0.69, multiplier: 1.0, timesteps: 1000 }).unwrap();
        close(&s.sigmas, &[1.0, 0.6742671132087708, 0.4082840085029602, 0.1869918704032898, 0.0]);
        assert!(s.dt(0) < 0.0);
        assert!(FlowSchedule::simple(0, q).is_err());
    }

    #[test]
    fn bf16_rounding() {
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(0.9430525), 0.94140625);
        assert_eq!(bf16_round(943.05), 944.0);
        // ComfyUI: bf16(bf16(943.05) / 1000) = bf16(0.944) = 0.9453125
        assert_eq!(comfy_timestep(0.9430525), 0.9453125);
        assert_eq!(comfy_timestep(1.0), 1.0);
        assert_eq!(comfy_timestep(0.0), 0.0);
    }

    /// The prompt tokens ComfyUI's `TextEncodeQwenImage21` feeds the encoder, from the same
    /// `tokenizer.json`: needs the official snapshot, so it runs only where it is.
    #[test]
    fn prompt_tokens_match_comfy() {
        let model = std::path::Path::new(r"C:\Users\5090\Documents\_eval\weights\qwen-image-2.1");
        let path = model.join("processor/tokenizer.json");
        if !path.is_file() {
            eprintln!("skipped: no snapshot at {}", model.display());
            return;
        }
        let tok = tokenizers::Tokenizer::from_file(&path).unwrap();
        let ids = |p: &str| -> Vec<u32> {
            tok.encode(crate::model::text_encoder::prompt::t2i_template(p), false)
                .unwrap()
                .get_ids()
                .to_vec()
        };
        assert_eq!(
            ids("a cat"),
            [151644, 8948, 198, 1092, 30782, 408, 323, 23643, 279, 3897, 9934, 13, 151645, 198, 151644, 872, 198, 64, 8251, 151645, 198, 151644, 77091, 198]
        );
        assert_eq!(
            ids(""),
            [151644, 8948, 198, 1092, 30782, 408, 323, 23643, 279, 3897, 9934, 13, 151645, 198, 151644, 872, 198, 220, 151645, 198, 151644, 77091, 198]
        );
        assert_eq!(
            ids("A (red) fox, 4k\nsecond line"),
            [151644, 8948, 198, 1092, 30782, 408, 323, 23643, 279, 3897, 9934, 13, 151645, 198, 151644, 872, 198, 32, 320, 1151, 8, 38835, 11, 220, 19, 74, 198, 5569, 1555, 151645, 198, 151644, 77091, 198]
        );
        // The system turn ends at the second <|im_start|>: 14 tokens, what `generate` drops.
        let sys = tok
            .encode(format!("<|im_start|>system\n{}<|im_end|>\n", crate::model::text_encoder::prompt::SYS_PROMPT), false)
            .unwrap()
            .get_ids()
            .len();
        assert_eq!(sys, 14);
    }
}
