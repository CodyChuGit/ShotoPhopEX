"""Export ComfyUI's SAM 3.1 (the exact modules PhotoshopEX's reference runs) to ONNX.

Run with ComfyUI's own Python:

    python -s tools/export_sam31.py --comfy <ComfyUI dir> --out <dir> [--parts vision,text,geometry,decoder,heads]
                                    [--verify] [--image <png>]

Five graphs come out, one model directory (`<out>/`):

    vision.onnx     images [1,3,1008,1008] (RGB 0..1, as ComfyUI feeds the trunk: no normalisation)
                    → fpn_feat_0 [1,256,288,288], fpn_feat_1 [1,256,144,144], fpn_feat_2 [1,256,72,72],
                      fpn_pos_2 [1,256,72,72], geo_cls [1,1,256],
                      trk_feat_0 [1,256,288,288], trk_feat_1 [1,256,144,144], trk_feat_2 [1,256,72,72]
                    (trk_feat_2 already carries the tracker's no-memory embedding)
    text.onnx       input_ids [1,32] i64, attention_mask [1,32] i64 → text_features [1,32,256], text_mask [1,32] bool
    geometry.onnx   boxes [1,M,4] (cx,cy,w,h in 0..1), fpn_feat_2 → geo_features [1,M,256]
    decoder.onnx    fpn_feat_0..2, fpn_pos_2, prompt_features [1,P,256], prompt_mask [1,P] bool
                    → pred_boxes [1,200,4] (x0,y0,x1,y1 in 0..1), pred_logits [1,200], pred_masks [1,200,288,288], presence [1]
    heads.onnx      trk_feat_0..2, point_coords [1,N,2] (pixels of the 1008 window), point_labels [1,N] i32
                    (-1 pad, 0 negative, 1 positive, 2 box top-left, 3 box bottom-right),
                    mask_prompt [1,1,1008,1008] (logits), has_mask [1] (0 or 1)
                    → masks [1,4,288,288] (token 0 = single-mask head, 1..3 = the multimask heads),
                      iou [1,4], object_score [1,1]

Everything is traced in fp32 from the fp16 checkpoint, batch 1, fixed 1008 window; M, P and N are dynamic.
The caller (gn-perception) reproduces ComfyUI's SAM3_Detect node on top: prompt assembly, scoring,
mask selection, the NO_OBJ rule, interpolation chains and the crop-refine passes.
"""
import argparse
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--ckpt", default="sam3.1_multiplex_fp16.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="vision,text,geometry,decoder,heads")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--verify", action="store_true", help="compare each graph with torch through onnxruntime (CPU)")
    ap.add_argument("--image", default=None, help="a picture for the verification inputs (random otherwise)")
    ap.add_argument("--fp16", action="store_true", help="trace the vision, geometry, decoder and heads graphs in fp16 (inputs and outputs stay fp32; the text graph stays fp32)")
    ap.add_argument("--mha", action="store_true", help="translate unmasked scaled_dot_product_attention to ONNX Runtime's fused com.microsoft.MultiHeadAttention (the vision graph; the exporter's own MatMul-Softmax-MatMul chain otherwise)")
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    os.makedirs(args.out, exist_ok=True)

    sys.argv = [sys.argv[0]]
    sys.path.insert(0, args.comfy)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import comfy.sd
    import comfy.model_management
    import folder_paths
    from comfy.ldm.sam3.detector import box_cxcywh_to_xyxy
    from comfy.ops import cast_to_input

    # ComfyUI routes RoPE through its `comfy_kitchen` custom op outside training, which has no
    # ONNX translation; the training flag selects the plain-torch implementation (same math).
    comfy.model_management.in_training = True

    ckpt = folder_paths.get_full_path_or_raise("checkpoints", args.ckpt)
    t0 = time.perf_counter()
    mdtype = torch.float16 if args.fp16 else torch.float32
    model, clip, _vae, _cv = comfy.sd.load_checkpoint_guess_config(
        ckpt, output_vae=False, output_clip=True, embedding_directory=None,
        model_options={"dtype": mdtype}, te_model_options={"dtype": torch.float32})
    comfy.model_management.load_model_gpu(model)
    clip.load_model()
    device = comfy.model_management.get_torch_device()
    sam3 = model.model.diffusion_model.eval()
    det = sam3.detector
    trk = sam3.tracker
    te = clip.cond_stage_model.sam3_clip
    print(f"loaded in {time.perf_counter() - t0:.1f} s: dtype {model.model.get_dtype()}, device {device}, "
          f"{type(trk).__name__}, ops {type(det.transformer.encoder.layers[0].linear1).__module__}")

    # ---------------------------------------------------------------- wrappers
    class Vision(nn.Module):
        def forward(self, images):
            images = images.to(mdtype)
            bb = det.backbone["vision_backbone"]
            features, positions, tracker_features, _ = bb(images, tracker_mode="interactive")
            enc_feat, enc_pos = features[-1], positions[-1]
            img_flat = enc_feat.flatten(2).permute(0, 2, 1)
            pos_flat = enc_pos.flatten(2).permute(0, 2, 1)
            geo = det.geometry_encoder
            geo_cls = geo.norm(geo.final_proj(cast_to_input(geo.cls_embed.weight, img_flat).view(1, 1, -1)))
            for layer in geo.encode:
                geo_cls = det._run_geo_layer(layer, geo_cls, img_flat, pos_flat)
            geo_cls = geo.encode_norm(geo_cls)
            t2 = tracker_features[-1]
            B, C, H, W = t2.shape
            no_mem = getattr(trk, "interactivity_no_mem_embed", None)
            if no_mem is None:
                no_mem = getattr(trk, "no_mem_embed")
            flat = t2.flatten(2).permute(0, 2, 1) + cast_to_input(no_mem, t2)
            t2 = flat.view(B, H, W, C).permute(0, 3, 1, 2)
            outs = (features[0], features[1], features[2], positions[2], geo_cls, tracker_features[0], tracker_features[1], t2)
            return tuple(o.float() for o in outs)

    class Text(nn.Module):
        def forward(self, input_ids, attention_mask):
            outputs = te.transformer(input_ids, attention_mask, intermediate_output=te.layer_idx,
                                     final_layer_norm_intermediate=te.layer_norm_hidden_state, dtype=torch.float32)
            z = outputs[0].float()
            feats = det.backbone["language_backbone"]["resizer"](z.to(mdtype))
            return feats.float(), attention_mask.bool()

    class Geometry(nn.Module):
        def forward(self, boxes, fpn_feat_2):
            img_flat = fpn_feat_2.to(mdtype).flatten(2).permute(0, 2, 1)
            return det.geometry_encoder(points=None, boxes=boxes.to(mdtype), image_features=img_flat).float()

    class Decoder(nn.Module):
        def forward(self, f0, f1, f2, p2, prompt, prompt_mask):
            f0, f1, f2, p2, prompt = (t.to(mdtype) for t in (f0, f1, f2, p2, prompt))
            _, _, H, W = f2.shape
            img_flat = f2.flatten(2).permute(0, 2, 1)
            pos_flat = p2.flatten(2).permute(0, 2, 1)
            memory = det.transformer.encoder(img_flat, pos_flat, prompt, prompt_mask)
            dec = det.transformer.decoder(memory, pos_flat, prompt, prompt_mask, H, W)
            query_out, pred_boxes = dec["decoder_output"], dec["pred_boxes"]
            scores = det.dot_prod_scoring(query_out, prompt, prompt_mask)
            masks = det.segmentation_head(query_out, [f0, f1, f2], encoder_hidden_states=memory, prompt=prompt, prompt_mask=prompt_mask)
            return box_cxcywh_to_xyxy(pred_boxes).float(), scores.float(), masks.float(), dec["presence"].float()

    class Heads(nn.Module):
        def forward(self, t0, t1, t2, point_coords, point_labels, mask_prompt, has_mask):
            t0, t1, t2, point_coords, mask_prompt, has_mask = (t.to(mdtype) for t in (t0, t1, t2, point_coords, mask_prompt, has_mask))
            pe = getattr(trk, "interactive_sam_prompt_encoder", None) or trk.sam_prompt_encoder
            md = getattr(trk, "interactive_sam_mask_decoder", None) or trk.sam_mask_decoder
            # Sparse prompts: every label through the same table, -1 = "not a point".
            emb = pe.pe_layer.forward_with_coords(point_coords + 0.5, pe.input_image_size)
            table = torch.cat([cast_to_input(pe.point_embeddings[i].weight, emb) for i in range(4)], dim=0)  # [4, C]
            valid = (point_labels >= 0)
            idx = point_labels.clamp(min=0).long()
            emb = torch.where(valid.unsqueeze(-1), emb + table[idx], cast_to_input(pe.not_a_point_embed.weight, emb).expand_as(emb))
            sparse = emb
            # Dense prompt: the mask (antialiased to 288, as forward_sam_heads does) or the no-mask embedding.
            m = F.interpolate(mask_prompt, size=(pe.image_embedding_size[0] * 4, pe.image_embedding_size[1] * 4), mode="bilinear", align_corners=False, antialias=True)
            dense_mask = pe.mask_downscaling(m)
            dense_none = cast_to_input(pe.no_mask_embed.weight, emb).reshape(1, -1, 1, 1).expand(1, -1, pe.image_embedding_size[0], pe.image_embedding_size[1])
            hm = has_mask.reshape(1, 1, 1, 1)
            dense = hm * dense_mask + (1.0 - hm) * dense_none
            image_pe = cast_to_input(pe.get_dense_pe(), t2)
            # SAMMaskDecoder.forward up to the raw heads (all four mask tokens).
            B = 1
            tokens = torch.cat([cast_to_input(md.obj_score_token.weight, sparse), cast_to_input(md.iou_token.weight, sparse), cast_to_input(md.mask_tokens.weight, sparse)], dim=0)
            tokens = torch.cat([tokens.unsqueeze(0).expand(B, -1, -1), sparse], dim=1)
            src = t2 + dense
            pos_src = image_pe.expand(B, -1, -1, -1)
            b, c, h, w = src.shape
            hs, src_out = md.transformer(src.flatten(2).permute(0, 2, 1), pos_src.flatten(2).permute(0, 2, 1), tokens)
            obj_score_token_out = hs[:, 0, :]
            iou_token_out = hs[:, 1, :]
            mask_tokens_out = hs[:, 2:2 + md.num_mask_tokens, :]
            src_out = src_out.permute(0, 2, 1).view(b, c, h, w)
            dc1, ln1, act1, dc2, act2 = md.output_upscaling
            upscaled = act1(ln1(dc1(src_out) + md.conv_s1(t1)))
            upscaled = act2(dc2(upscaled) + md.conv_s0(t0))
            hyper_in = torch.stack([mlp(mask_tokens_out[:, i, :]) for i, mlp in enumerate(md.output_hypernetworks_mlps)], dim=1)
            masks = (hyper_in @ upscaled.flatten(2)).view(B, md.num_mask_tokens, upscaled.shape[2], upscaled.shape[3])
            iou_pred = md.iou_prediction_head(iou_token_out)
            object_score_logits = md.pred_obj_score_head(obj_score_token_out)
            return masks.float(), iou_pred.float(), object_score_logits.float()

    # ---------------------------------------------------------------- inputs
    from torch.export import Dim
    if args.image:
        from PIL import Image
        import numpy as np
        im = Image.open(args.image).convert("RGB").resize((1008, 1008), Image.BILINEAR)
        images = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    else:
        images = torch.rand(1, 3, 1008, 1008, device=device)

    with torch.no_grad():
        vis = Vision().eval()
        f0, f1, f2, p2, geo_cls, t0_, t1_, t2_ = vis(images)
        print("vision outputs:", [tuple(t.shape) for t in (f0, f1, f2, p2, geo_cls, t0_, t1_, t2_)])
        tokens = clip.tokenize("fox")
        ids = torch.tensor([[p[0] for p in tokens["sam3_clip"][0]]], dtype=torch.long, device=device)
        amask = (ids != 0).long()
        text_feats, text_mask = Text().eval()(ids, amask)
        print("text outputs:", tuple(text_feats.shape), tuple(text_mask.shape), "ids", ids[0, :6].tolist())
        boxes = torch.tensor([[[0.4, 0.5, 0.5, 0.6], [0.2, 0.2, 0.1, 0.1]]], device=device)
        geo = Geometry().eval()(boxes, f2)
        print("geometry output:", tuple(geo.shape))
        prompt = torch.cat([text_feats, geo, geo_cls], dim=1)
        pmask = torch.cat([text_mask, torch.ones(1, geo.shape[1] + 1, dtype=torch.bool, device=device)], dim=1)
        dboxes, dscores, dmasks, dpres = Decoder().eval()(f0, f1, f2, p2, prompt, pmask)
        print("decoder outputs:", tuple(dboxes.shape), tuple(dscores.shape), tuple(dmasks.shape), tuple(dpres.shape))
        pts = torch.tensor([[[480.0, 520.0], [0.0, 0.0]]], device=device)
        lbl = torch.tensor([[1, -1]], dtype=torch.int32, device=device)
        mprompt = torch.zeros(1, 1, 1008, 1008, device=device)
        hmask = torch.zeros(1, device=device)
        hmasks, hiou, hobj = Heads().eval()(t0_, t1_, t2_, pts, lbl, mprompt, hmask)
        print("heads outputs:", tuple(hmasks.shape), tuple(hiou.shape), tuple(hobj.shape))

    # ---------------------------------------------------------------- export
    # ORT's fused attention for the trunk: aten::scaled_dot_product_attention (q, k, v as
    # [B, H, S, D], no mask) → MultiHeadAttention over [B, S, H·D] and back. ORT's CUDA provider
    # runs that op with its flash / memory-efficient kernels in fp16; the exporter's own
    # decomposition is an unfused MatMul-Softmax-MatMul chain.
    def sdpa_as_mha(query, key, value, attn_mask=None, dropout_p: float = 0.0, is_causal: bool = False, scale=None, enable_gqa: bool = False):
        import onnxscript
        from onnxscript import opset18 as op
        msft = onnxscript.values.Opset("com.microsoft", 1)
        heads = int(query.shape[1])
        perm = [0, 2, 1, 3]
        to3 = op.Constant(value_ints=[0, 0, -1])
        q = op.Reshape(op.Transpose(query, perm=perm), to3)
        k = op.Reshape(op.Transpose(key, perm=perm), to3)
        v = op.Reshape(op.Transpose(value, perm=perm), to3)
        out = msft.MultiHeadAttention(q, k, v, num_heads=heads)
        shape4 = op.Shape(op.Transpose(query, perm=perm))
        return op.Transpose(op.Reshape(out, shape4), perm=perm)

    if args.mha:
        # onnxscript resolves `msft.MultiHeadAttention` through the ONNX schema registry, which
        # knows nothing of ORT's contrib domain: register the three-input form we emit.
        import onnx
        from onnx.defs import OpSchema
        try:
            onnx.defs.get_schema("MultiHeadAttention", 1, "com.microsoft")
        except Exception:
            onnx.defs.register_schema(OpSchema(
                "MultiHeadAttention", "com.microsoft", 1, doc="ORT fused attention (query, key, value as [B, S, H*D])",
                inputs=[OpSchema.FormalParameter("query", "T"), OpSchema.FormalParameter("key", "T"), OpSchema.FormalParameter("value", "T")],
                outputs=[OpSchema.FormalParameter("output", "T")],
                type_constraints=[("T", ["tensor(float16)", "tensor(float)"], "")],
                attributes=[OpSchema.Attribute("num_heads", OpSchema.AttrType.INT, "")]))
    translation = {torch.ops.aten.scaled_dot_product_attention.default: sdpa_as_mha} if args.mha else None

    def export(name, module, inputs, input_names, output_names, dynamic_shapes, fused=False):
        path = os.path.join(args.out, f"{name}.onnx")
        t = time.perf_counter()
        prog = torch.onnx.export(module, inputs, dynamo=True, opset_version=args.opset, input_names=input_names,
                                 output_names=output_names, dynamic_shapes=dynamic_shapes, external_data=True,
                                 optimize=True, verify=False, custom_translation_table=translation if fused else None)
        prog.save(path, external_data=True)
        size = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out) if f.startswith(f"{name}.onnx"))
        print(f"exported {name}: {size / 1e6:.0f} MB in {time.perf_counter() - t:.0f} s")
        return path

    exported = {}
    with torch.no_grad():
        if "vision" in parts:
            exported["vision"] = export("vision", vis, (images,), ["images"],
                                        ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2", "geo_cls", "trk_feat_0", "trk_feat_1", "trk_feat_2"], None, fused=True)
        if "text" in parts:
            exported["text"] = export("text", Text().eval(), (ids, amask), ["input_ids", "attention_mask"], ["text_features", "text_mask"], None)
        if "geometry" in parts:
            m = Dim("m", min=1, max=256)
            exported["geometry"] = export("geometry", Geometry().eval(), (boxes, f2), ["boxes", "fpn_feat_2"], ["geo_features"],
                                          ({1: m}, None))
        if "decoder" in parts:
            p = Dim("p", min=2, max=1024)
            exported["decoder"] = export("decoder", Decoder().eval(), (f0, f1, f2, p2, prompt, pmask),
                                         ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2", "prompt_features", "prompt_mask"],
                                         ["pred_boxes", "pred_logits", "pred_masks", "presence"],
                                         (None, None, None, None, {1: p}, {1: p}))
        if "heads" in parts:
            n = Dim("n", min=1, max=256)
            exported["heads"] = export("heads", Heads().eval(), (t0_, t1_, t2_, pts, lbl, mprompt, hmask),
                                       ["trk_feat_0", "trk_feat_1", "trk_feat_2", "point_coords", "point_labels", "mask_prompt", "has_mask"],
                                       ["masks", "iou", "object_score"],
                                       (None, None, None, {1: n}, {1: n}, None, None))

    # ---------------------------------------------------------------- verify
    if args.verify and args.fp16:
        print("verify: fp16 graphs are checked through the parity harness (gn-bench), not on the CPU; skipped")
        return
    if args.verify:
        try:
            import onnxruntime as ort
        except ImportError:
            print("verify: onnxruntime is not installed in this Python; skipped")
            return
        import numpy as np

        def check(name, feeds, refs):
            if name not in exported:
                return
            sess = ort.InferenceSession(exported[name], providers=["CPUExecutionProvider"])
            t = time.perf_counter()
            outs = sess.run(None, {k: v.detach().cpu().numpy() for k, v in feeds.items()})
            dt = time.perf_counter() - t
            worst = 0.0
            for o, r in zip(outs, refs):
                r = r.detach().cpu().numpy()
                if o.dtype == np.bool_:
                    diff = float((o != r).mean())
                else:
                    diff = float(np.abs(o.astype(np.float32) - r.astype(np.float32)).max())
                worst = max(worst, diff)
            print(f"verify {name}: max abs diff {worst:.3e} ({dt:.1f} s on CPU)")

        with torch.no_grad():
            check("text", {"input_ids": ids, "attention_mask": amask}, [text_feats, text_mask])
            check("geometry", {"boxes": boxes, "fpn_feat_2": f2}, [geo])
            check("decoder", {"fpn_feat_0": f0, "fpn_feat_1": f1, "fpn_feat_2": f2, "fpn_pos_2": p2, "prompt_features": prompt, "prompt_mask": pmask}, [dboxes, dscores, dmasks, dpres])
            check("heads", {"trk_feat_0": t0_, "trk_feat_1": t1_, "trk_feat_2": t2_, "point_coords": pts, "point_labels": lbl, "mask_prompt": mprompt, "has_mask": hmask}, [hmasks, hiou, hobj])
            if "vision" in exported and os.environ.get("GN_VERIFY_VISION"):
                check("vision", {"images": images}, [f0, f1, f2, p2, geo_cls, t0_, t1_, t2_])


if __name__ == "__main__":
    main()
