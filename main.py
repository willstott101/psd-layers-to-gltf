"""
psd2gltf — Convert PSD/XCF layers into transparent planes in a GLB file.

Controls (minimal, format-prefixed):
  • --layer-spacing       (default 0.5)
  • --px-per-unit         (default 1000)
  • --texture-format      png | webp | bmp
  • PNG : --png-compression
  • WebP: --webp-quality   (0–100 OR 'lossless')
  • BMP : (no options)

N.B. BMP is the fastest image format, WebP is the smallest but has resolution limitations.

Examples:
  # Default PNG textures, spacing 0.5, 1000 px/unit
  psd2gltf artwork.psd scene.glb

  # GLB with WebP, lossy @ quality 80
  psd2gltf artwork.xcf scene.glb --texture-format webp --webp-quality 80

  # GLB with WebP, lossless
  psd2gltf artwork.psd scene.glb --texture-format webp --webp-quality lossless

  # PNG with higher compression and denser spacing
  psd2gltf in.psd out.glb --texture-format png --png-compression 9 \
      --layer-spacing 0.25 --px-per-unit 2000
"""

import io
import os
import sys
import argparse
from array import array
from typing import Union, Sequence

import pygltflib
from psd_tools import PSDImage
from gimpformats.gimpXcfDocument import GimpDocument, GimpGroup
from gimpformats.GimpLayer import GimpLayer

DEFAULT_PNG_COMPRESSION = 6
DEFAULT_WEBP_QUALITY = 82


def shared_plane_geometry(gltf, buf):
    """
    Generates UVs and Indices suitable for use as part of a 2-tri Plane
    and adds them to the passed gltf.

    Returns (index_accessor_index, uvs_accessor_index)
    """
    indices = array("H", [2, 1, 0, 3, 2, 0])
    index_bytes = indices.tobytes()
    buf.write(index_bytes)

    uvs = array(
        "f",
        [
            0.0,
            0.0,  # Bottom left
            1.0,
            0.0,  # Bottom right
            1.0,
            1.0,  # Top right
            0.0,
            1.0,  # Top left
        ],
    )
    uvs_bytes = uvs.tobytes()
    buf.write(uvs_bytes)

    buffer_view_index = len(gltf.bufferViews)
    gltf.bufferViews.extend(
        [
            pygltflib.BufferView(
                buffer=0,
                byteOffset=0,
                byteLength=len(index_bytes),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(index_bytes),
                byteLength=len(uvs_bytes),
                target=pygltflib.ARRAY_BUFFER,
            ),
        ]
    )
    accessor_index = len(gltf.accessors)
    gltf.accessors.extend(
        [
            pygltflib.Accessor(
                bufferView=buffer_view_index,
                componentType=pygltflib.UNSIGNED_SHORT,
                count=6,
                type="SCALAR",
                max=[3],
                min=[0],
            ),
            pygltflib.Accessor(
                bufferView=buffer_view_index + 1,
                componentType=pygltflib.FLOAT,
                count=4,
                type="VEC2",
                max=[1.0, 1.0],
                min=[0.0, 0.0],
            ),
        ]
    )
    return accessor_index, accessor_index + 1


def plane_with_offset_and_size(
    name,
    material_index,
    pixel_offset: tuple[float, float],
    pixel_size: tuple[float, float],
    pixel_center: tuple[float, float],
    pixels_per_unit: int,
    layer_spacing: float,
    gltf,
    buf,
    index_accessor_index,
    uvs_accessor_index,
):
    r = float(pixel_size[0]) / (pixels_per_unit * 2)
    l = -r
    b = float(pixel_size[1]) / (pixels_per_unit * 2)
    t = -b

    x = (float(pixel_offset[0] - pixel_center[0]) / pixels_per_unit) + r
    y = (float(pixel_offset[1] - pixel_center[1]) / pixels_per_unit) + b

    # Define vertex positions (x, y, z) for a rectangular plane
    positions = array(
        "f",
        [
            l,
            b,
            0.0,  # Bottom left
            r,
            b,
            0.0,  # Bottom right
            r,
            t,
            0.0,  # Top right
            l,
            t,
            0.0,  # Top left
        ],
    )
    position_bytes = positions.tobytes()

    original_byte_len = buf.getbuffer().nbytes
    buf.write(position_bytes)
    buffer_view_index = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=original_byte_len,
            byteLength=len(position_bytes),
            target=pygltflib.ARRAY_BUFFER,
        )
    )
    accessor_index = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=buffer_view_index,
            componentType=pygltflib.FLOAT,
            count=4,
            type="VEC3",
            max=[r, t, 0],
            min=[l, b, 0],
        ),
    )
    mesh_index = len(gltf.meshes)
    gltf.meshes.append(
        pygltflib.Mesh(
            primitives=[
                pygltflib.Primitive(
                    attributes={
                        "POSITION": accessor_index,
                        "TEXCOORD_0": uvs_accessor_index,
                    },
                    indices=index_accessor_index,
                    material=material_index,
                )
            ]
        )
    )
    node_index = len(gltf.nodes)
    gltf.nodes.append(
        pygltflib.Node(
            name=name, mesh=mesh_index, translation=[x, -y, node_index * layer_spacing]
        )
    )
    return node_index


def exec_every_layer_psd(layer, group_fn, layer_fn, **kwargs):
    if layer.is_group():
        sub_layers_and_results = [
            (layer, exec_every_layer_psd(sub_layer, group_fn, layer_fn, **kwargs))
            for sub_layer in layer
            if sub_layer.is_visible()
        ]
        return group_fn(layer, sub_layers_and_results, **kwargs)
    else:
        return layer_fn(layer, **kwargs)


def exec_every_layer_xcf(layer, group_fn, layer_fn, **kwargs):
    if isinstance(layer, GimpGroup):
        sub_layers_and_results = [
            (layer, exec_every_layer_xcf(sub_layer, group_fn, layer_fn, **kwargs))
            for sub_layer in layer.children
            if sub_layer.visible
        ]
        return group_fn(layer, sub_layers_and_results, **kwargs)
    else:
        return layer_fn(layer, **kwargs)


def group_fn(layer, sub_layers_and_results, gltf, **_kws):
    node_index = len(gltf.nodes)
    children_node_indices = [
        child_node_index for sub_layer, child_node_index in sub_layers_and_results
    ]
    gltf.nodes.append(pygltflib.Node(name=layer.name, children=children_node_indices))
    return node_index


def layer_fn(
    layer,
    pixel_center,
    args,
    gltf,
    buf,
    index_accessor_index,
    uvs_accessor_index,
):
    if isinstance(layer, GimpLayer):
        layer_offset = layer.xOffset, layer.yOffset
        layer_size = layer.width, layer.height
        pil_image = layer.image
    else:
        layer_offset = layer.offset
        layer_size = layer.size
        pil_image = layer.composite()
    # We don't write directly to the main buffer, in-case Pillow decides to seek or truncate or summin.
    image_bytes_io = io.BytesIO()
    pil_image.save(image_bytes_io, **args.pillow_args)
    image_bytes = image_bytes_io.getvalue()
    pil_image = None

    original_byte_len = buf.getbuffer().nbytes
    buf.write(image_bytes)

    buffer_view_index = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=original_byte_len,
            byteLength=len(image_bytes),
        )
    )

    image_index = len(gltf.images)
    gltf.images.append(
        pygltflib.Image(
            bufferView=buffer_view_index,
            mimeType=args.gltf_image_mimetype,
            name=layer.name,
        )
    )

    texture_index = len(gltf.textures)
    gltf.textures.append(
        pygltflib.Texture(
            source=image_index,
        )
    )

    material_index = len(gltf.materials)
    gltf.materials.append(
        pygltflib.Material(
            alphaMode=pygltflib.BLEND,
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorTexture=pygltflib.TextureInfo(index=texture_index)
            ),
            extensions={"KHR_materials_unlit": {}},
        )
    )

    return plane_with_offset_and_size(
        layer.name,
        material_index,
        layer_offset,
        layer_size,
        pixel_center=pixel_center,
        pixels_per_unit=args.px_per_unit,
        layer_spacing=args.layer_spacing,
        gltf=gltf,
        buf=buf,
        index_accessor_index=index_accessor_index,
        uvs_accessor_index=uvs_accessor_index,
    )


def convert_layered_doc_to_gltf(args):
    if args.input.endswith(".xcf"):
        doc = GimpDocument(args.input)
        pixel_center = (doc.width / 2, doc.height / 2)
        root_layer = doc.walkTree()
        exec_every_layer = exec_every_layer_xcf
    else:
        root_layer = PSDImage.open(args.input)
        pixel_center = (root_layer.size[0] / 2, root_layer.size[1] / 2)
        exec_every_layer = exec_every_layer_psd

    gltf = pygltflib.GLTF2()
    buf = io.BytesIO()

    index_accessor_index, uvs_accessor_index = shared_plane_geometry(gltf=gltf, buf=buf)

    # Run through all layers and add them to the GLTF object graph
    # These objects respect layer folders from the PSD and end up with a single "Root" object
    node_index = exec_every_layer(
        root_layer,
        group_fn,
        layer_fn,
        buf=buf,
        gltf=gltf,
        pixel_center=pixel_center,
        args=args,
        index_accessor_index=index_accessor_index,
        uvs_accessor_index=uvs_accessor_index,
    )

    scene = pygltflib.Scene(nodes=[node_index])
    gltf.scenes.append(scene)

    # Embed the buffer in such a way it will be included in a single GLB file
    # TODO: Have an option write to an external .bin (and be way more memory efficient)
    final_buffer = buf.getvalue()
    gltf.set_binary_blob(final_buffer)
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(final_buffer)))

    return gltf


def parse_png_compression(val: str) -> Union[int, str]:
    """
    Accepts either an integer 0..9 or the literal 'optimize' (case-insensitive).
    Returns int for compression, or the string 'optimize' for optimize mode.
    """
    v = val.strip().lower()
    if v == "optimize":
        return "optimize"
    try:
        q = int(v)
    except ValueError as e:
        raise argparse.ArgumentTypeError("must be 0-9 or 'optimize'") from e
    if not 0 <= q <= 9:
        raise argparse.ArgumentTypeError("quality must be between 0 and 9")
    return q


def parse_webp_quality(val: str) -> Union[int, str]:
    """
    Accepts either an integer 0..100 or the literal 'lossless' (case-insensitive).
    Returns int for lossy, or the string 'lossless' for lossless mode.
    """
    v = val.strip().lower()
    if v == "lossless":
        return "lossless"
    try:
        q = int(v)
    except ValueError as e:
        raise argparse.ArgumentTypeError("must be 0-100 or 'lossless'") from e
    if not 0 <= q <= 100:
        raise argparse.ArgumentTypeError("quality must be between 0 and 100")
    return q


def build_parser() -> argparse.ArgumentParser:
    epilog = """\
Notes:
  • layer-spacing is the distance (in world units on Z) between adjacent planes.
  • px-per-unit sets how many pixels wide a layer must be to span 1.0 world unit.
  • When texture-format is PNG, WebP-specific flags are ignored; BMP has none.
"""
    p = argparse.ArgumentParser(
        prog="psd2gltf",
        description="Convert PSD/XCF layers into transparent GLTF/GLB planes.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Positional I/O ---
    p.add_argument("input", help="Input PSD/XCF file")
    p.add_argument(
        "output",
        help="Output file (must be .glb for now).",
    )

    # --- Scene layout ---
    p.add_argument(
        "--layer-spacing",
        type=float,
        default=0.5,
        help="Distance between adjacent layers in 3D units (default: 0.5)",
    )
    p.add_argument(
        "--px-per-unit",
        type=float,
        default=1000.0,
        help="Pixels per 1.0 world unit along X (default: 1000)",
    )

    # --- Texture format selection ---
    p.add_argument(
        "--texture-format",
        choices=["png", "webp", "bmp"],
        default="png",
        help="Image format for exported textures (default: png)",
    )

    # --- PNG compression ---
    png = p.add_argument_group("PNG options (apply when --texture-format png)")
    png.add_argument(
        "--png-compression",
        type=parse_png_compression,
        metavar="(0-9|'optimize')",
        default=argparse.SUPPRESS,
        help=f"zlib compression level (0=fast/large .. 9=slow/small; optimize=slowest/smallest; default: {DEFAULT_PNG_COMPRESSION})",
    )

    # --- WebP compression: single flag, dual meaning ---
    webp = p.add_argument_group("WebP options (apply when --texture-format webp)")
    webp.add_argument(
        "--webp-quality",
        type=parse_webp_quality,
        default=argparse.SUPPRESS,
        help=f"Lossy quality 0-100 (default: {DEFAULT_WEBP_QUALITY}) OR the literal 'lossless' to enable lossless mode",
        metavar="(0-100|'lossless')",
    )

    return p


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if os.path.splitext(args.output)[1].lower() != ".glb":
        raise SystemExit("Output file must be .glb")

    # Sanity checks
    if args.layer_spacing <= 0:
        raise SystemExit("--layer-spacing must be > 0")
    if args.px_per_unit <= 0:
        raise SystemExit("--px-per-unit must be > 0")

    png_set = getattr(args, "png_compression", None)
    webp_set = getattr(args, "webp_quality", None)

    irrelevant = []
    if args.texture_format != "png" and png_set is not None:
        irrelevant.append("--png-compression")
    if args.texture_format != "webp" and webp_set is not None:
        irrelevant.append("--webp-quality")
    if irrelevant:
        print(
            f"Note: ignored for {args.texture_format.upper()} textures: "
            + ", ".join(irrelevant),
            file=sys.stderr,
        )

    # Normalize image settings for downstream use
    if args.texture_format == "webp":
        args.gltf_image_mimetype = "image/png"
        if webp_set == "lossless":
            args.pillow_args = {"lossless": True, "format": "WEBP"}
        else:
            args.pillow_args = {
                "quality": DEFAULT_WEBP_QUALITY if webp_set is None else webp_set,
                "format": "WEBP",
            }
    elif args.texture_format == "png":
        if png_set == "optimize":
            args.pillow_args = {"optimize": True, "format": "PNG"}
        else:
            args.pillow_args = {
                "compress_level": DEFAULT_PNG_COMPRESSION
                if png_set is None
                else png_set,
                "format": "PNG",
            }
        args.gltf_image_mimetype = "image/png"
    elif args.texture_format == "bmp":
        args.pillow_args = {"format": "BMP"}
        args.gltf_image_mimetype = "image/bmp"

    return args


if __name__ == "__main__":
    args = parse_args()
    gltf = convert_layered_doc_to_gltf(args)
    gltf.save(args.output)
