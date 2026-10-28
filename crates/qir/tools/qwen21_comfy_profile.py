"""Profile the installed PhotoshopEX graphs through ComfyUI's node events."""
import asyncio
import io
import json
from pathlib import Path
import time
import sys
import uuid

import aiohttp
from PIL import Image

ROOT = Path("C:/Users/5090")
WORKFLOWS = ROOT / "Documents/PhotoshopEX/crates/genai/workflows"
FIXTURES = ROOT / "ComfyUI/photocraft-tests/bench"
URL = "http://127.0.0.1:8188"


async def upload(session, image, name):
    data = io.BytesIO()
    image.save(data, format="PNG")
    form = aiohttp.FormData()
    form.add_field("image", data.getvalue(), filename=name, content_type="image/png")
    async with session.post(URL + "/upload/image", data=form) as response:
        response.raise_for_status()
        result = await response.json()
        return "/".join(filter(None, [result.get("subfolder"), result["name"]]))


async def main():
    async with aiohttp.ClientSession() as session:
        portrait = Image.open(FIXTURES / "speed/portrait-864x1152.png").convert("RGBA")
        ref = Image.open(FIXTURES / "stock/st-hat.jpg").convert("RGBA")
        scale = min(1.0, (1048576 / (ref.width * ref.height)) ** 0.5)
        ref = ref.resize((round(ref.width * scale), round(ref.height * scale)), Image.Resampling.LANCZOS)
        crop = portrait.crop((48, 384, 816, 1056))
        mask = Image.new("RGBA", crop.size, (0, 0, 0, 255))
        mask.paste((255, 255, 255, 255), (132, 136, 632, 556))
        images = {}
        for name, image in [("portrait", portrait), ("reference", ref), ("crop", crop), ("mask", mask)]:
            images[name] = await upload(session, image, "qir-profile-" + name + ".png")
        prompts = {
            "image-ref": "a portrait of a woman wearing this hat",
            "fill": "Edit image 1. Image 2 is a mask of image 1: change only the area that is white in image 2 so that it shows a denim jacket. Keep everything outside that area exactly as it is in image 1, with the same lighting, perspective and style.",
            "edit": "Make it winter: snow falling and a wool scarf around his neck.",
            "matte": "Remove the background, keeping only the man, and output a PNG image",
        }
        if "--tiles-only" in sys.argv:
            prompts = {"tiles": prompts["edit"]}
        tile_pixels = bytearray(portrait.tobytes())
        client = str(uuid.uuid4())
        async with session.ws_connect(URL + "/ws?clientId=" + client) as ws:
            for route, prompt in prompts.items():
                template = json.loads((WORKFLOWS / ("qwen-2.1-" + ("edit" if route == "tiles" else route) + ".json")).read_text(encoding="utf-8"))
                graph = template["graph"]
                values = {"prompt": prompt, "negative": "", "steps": 25, "cfg": 1.0, "seed": 42, "width": 864, "height": 1152, "prefix": "qir-profile"}
                values.update({m["placeholder"]: m["default"] for m in template["meta"]["models"]})
                values.update(image=images["crop" if route == "fill" else "portrait"], image1=images["reference"], mask=images["mask"])
                unused = {key for key, node in graph.items() if node["class_type"] == "LoadImage" and node["inputs"]["image"][2:-2] not in values}
                graph = {key: node for key, node in graph.items() if key not in unused}
                for node in graph.values():
                    node["inputs"] = {key: values.get(value[2:-2], value) if isinstance(value, str) and value.startswith("{{") else value for key, value in node["inputs"].items() if not (isinstance(value, list) and value[0] in unused)}
                    # Decode is the terminal output: measuring does not need a saved server-side PNG.
                    if node["class_type"] == "SaveImage":
                        node["class_type"] = "PreviewImage"
                        node["inputs"].pop("filename_prefix", None)
                for repeat in range(9 if route == "tiles" else 2):
                    if route == "tiles":
                        tile_pixels[repeat] ^= 1
                        name = await upload(session, Image.frombytes("RGBA", portrait.size, bytes(tile_pixels)), "qir-tile-" + str(repeat) + ".png")
                        graph["4"]["inputs"]["image"] = name
                    for node in graph.values():
                        if node["class_type"] == "KSampler":
                            node["inputs"]["seed"] = 42 + repeat
                    start = time.perf_counter()
                    async with session.post(URL + "/prompt", json={"prompt": graph, "client_id": client}) as response:
                        response.raise_for_status()
                        job = await response.json()
                    print(json.dumps({"route": route, "run": repeat, "queued": job}), flush=True)
                    while True:
                        packet = await ws.receive(timeout=600)
                        if packet.type != aiohttp.WSMsgType.TEXT:
                            continue
                        message = json.loads(packet.data)
                        kind, data = message.get("type"), message.get("data", {})
                        if data.get("prompt_id") != job["prompt_id"]:
                            continue
                        if kind in ("executing", "execution_cached", "execution_error", "execution_success"):
                            print(json.dumps({"route": route, "run": repeat, "seconds": time.perf_counter() - start, "event": kind, "data": data}), flush=True)
                        if kind == "execution_error":
                            raise RuntimeError(data)
                        if kind == "executing" and data.get("node") is None:
                            break


if __name__ == "__main__":
    asyncio.run(main())
