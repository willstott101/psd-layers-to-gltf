import io
import sys
import struct
from array import array

import pygltflib
from psd_tools import PSDImage


LAYER_SPACING = 0.5  # How far apart the layers are in 3D space
PIXELS_PER_3D_UNIT = 1000  # How many pixels wide a layer needs to be to reach 1.0 units wide in 3D
# Using BMP results in a large file but is much faster to write than using a compressed format
IMAGE_FORMAT = "BMP"  # Must be valid for Image.save(format=IMAGE_FORMAT)
IMAGE_MIMETYPE = "image/bmp"  # For GLTF (must match IMAGE_FORMAT)


def remap(value, source_min, source_max, out_min, out_max):
    normalized = value / (source_max - source_min)
    return normalized * (out_max - out_min)


def plane_with_offset_and_size(
    name,
    material_index,
    pixel_offset: tuple[float, float],
    pixel_size: tuple[float, float],
    pixel_center: tuple[float, float],
    pixels_per_unit: int,
    gltf,
    buf,
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
                    attributes={"POSITION": accessor_index, "TEXCOORD_0": 1},
                    indices=0,
                    material=material_index,
                )
            ]
        )
    )
    node_index = len(gltf.nodes)
    gltf.nodes.append(
        pygltflib.Node(
            name=name, mesh=mesh_index, translation=[x, -y, node_index * LAYER_SPACING]
        )
    )
    return node_index


# Define triangle indices for two triangles (0-1-2, 0-2-3)
indices = [0, 1, 2, 0, 2, 3]

index_bytes = struct.pack(f"<{len(indices)}H", *indices)  # Use uint16 for indices


def exec_every_layer(layer, group_fn, layer_fn, **kwargs):
    if layer.is_group():
        sub_layers_and_results = [
            (layer, exec_every_layer(sub_layer, group_fn, layer_fn, **kwargs))
            for sub_layer in layer
            if sub_layer.is_visible()
        ]
        return group_fn(layer, sub_layers_and_results, **kwargs)
    else:
        return layer_fn(layer, **kwargs)


def main():
    psd = PSDImage.open(sys.argv[1])

    def group_fn(layer, sub_layers_and_results, gltf, **_kws):
        # TODO: Do I need to deal with an offset/transformation here at the group level?
        # print("GROUP", layer, layer.offset)
        node_index = len(gltf.nodes)
        children_node_indices = [
            child_node_index for sub_layer, child_node_index in sub_layers_and_results
        ]
        gltf.nodes.append(
            pygltflib.Node(name=layer.name, children=children_node_indices)
        )
        return node_index

    def layer_fn(layer, pixel_center, pixels_per_unit, gltf, buf):
        # print("LAYER", layer, layer.offset)
        pil_image = layer.composite()
        # We don't write directly to the main buffer, in-case Pillow decides to seek or truncate or summin.
        image_bytes_io = io.BytesIO()
        pil_image.save(image_bytes_io, format=IMAGE_FORMAT)
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
                bufferView=buffer_view_index, mimeType=IMAGE_MIMETYPE, name=layer.name
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
            layer.offset,
            layer.size,
            pixel_center=pixel_center,
            pixels_per_unit=pixels_per_unit,
            gltf=gltf,
            buf=buf,
        )

    gltf = pygltflib.GLTF2()
    buf = io.BytesIO()

    indices = array("H", [2, 1, 0, 3, 2, 0])  # uint16
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
    gltf.accessors.extend(
        [
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.UNSIGNED_SHORT,
                count=6,
                type="SCALAR",
                max=[3],
                min=[0],
            ),
            pygltflib.Accessor(
                bufferView=1,
                componentType=pygltflib.FLOAT,
                count=4,
                type="VEC2",
                max=[1.0, 1.0],
                min=[0.0, 0.0],
            ),
        ]
    )

    pixel_center = (psd.size[0] / 2, psd.size[1] / 2)
    node_index = exec_every_layer(
        psd,
        group_fn,
        layer_fn,
        buf=buf,
        gltf=gltf,
        pixel_center=pixel_center,
        pixels_per_unit=PIXELS_PER_3D_UNIT,
    )

    scene = pygltflib.Scene(nodes=[node_index])
    gltf.scenes.append(scene)  # scene available at gltf.scenes[0]

    final_buffer = buf.getvalue()
    gltf.set_binary_blob(final_buffer)
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(final_buffer)))

    # gltf.convert_buffers(pygltflib.BufferFormat.BINFILE)  # convert buffers to files
    # gltf.buffers[0].uri = "0.bin"  # TODO: Why is this not automatic?!
    # gltf.save("output.gltf")
    gltf.save("output.glb")


if __name__ == "__main__":
    main()
