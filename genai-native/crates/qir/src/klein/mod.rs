//! FLUX 3 Image Klein (Black Forest Labs, FLUX Kommunity License; bring your own files) on CUDA:
//! text to image, editing with up to 10 reference pictures, and a fill that paints only inside
//! a mask (the editor's Expand with AusBoss's Outpaint LoRA, as on the Mac). From ComfyUI's
//! files as BFL publishes them (`flux3-image-klein-int8-convrot`, ConvRot INT8 like
//! Comfy-Org's MOSS, or the bf16 file), Qwen3-VL-4B as its text encoder (the file Krea 2 uses
//! serves) and the FLUX.2 VAE MOSS uses. See [`pipeline`].

pub mod dit;
pub mod pipeline;
