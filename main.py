import struct
import tempfile
from array import array

import pygltflib
from psd_tools import PSDImage


def remap(value, source_min, source_max, out_min, out_max):
    normalized = value / (source_max - source_min)
    return normalized * (out_max - out_min)


def plane_with_offset_and_size(name, material_index, pixel_offset: tuple[float, float], pixel_size: tuple[float, float], pixel_center: tuple[float, float], pixels_per_unit: int, gltf):
    l = pixel_offset[0] - pixel_center[0]
    r = l + pixel_size[0]
    t = pixel_offset[1] - pixel_center[1]
    b = t + pixel_size[1]

    # Define vertex positions (x, y, z) for a rectangular plane
    positions = array('f', [
        l * pixels_per_unit, b * pixels_per_unit, 0.0,  # Bottom left
        r * pixels_per_unit, b * pixels_per_unit, 0.0,  # Bottom right
        r * pixels_per_unit, t * pixels_per_unit, 0.0,  # Top right
        l * pixels_per_unit, t * pixels_per_unit, 0.0   # Top left
    ])
    position_bytes = positions.tobytes()

    original_byte_len = gltf.buffers[0].byteLength
    combined_bytes = gltf.binary_blob() + position_bytes
    gltf.set_binary_blob(combined_bytes)
    gltf.buffers[0].byteLength = len(combined_bytes)
    buffer_view_index = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(buffer=0, byteOffset=original_byte_len, byteLength=len(position_bytes), target=pygltflib.ARRAY_BUFFER)
    )
    accessor_index = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(bufferView=buffer_view_index, componentType=pygltflib.FLOAT, count=4, type="VEC3", max=[r * pixels_per_unit, b * pixels_per_unit, 0], min=[l * pixels_per_unit, t * pixels_per_unit, 0]),
    )
    mesh_index = len(gltf.meshes)
    gltf.meshes.append(pygltflib.Mesh(primitives=[pygltflib.Primitive(attributes={"POSITION": accessor_index}, indices=0, material=material_index)]))
    node_index = len(gltf.nodes)
    gltf.nodes.append(pygltflib.Node(name=name, mesh=mesh_index))
    return node_index


# Define triangle indices for two triangles (0-1-2, 0-2-3)
indices = [0, 1, 2, 0, 2, 3]

index_bytes = struct.pack(f'<{len(indices)}H', *indices)  # Use uint16 for indices


def exec_every_layer(layer, group_fn, layer_fn, **kwargs):
    if layer.is_group():
        sub_layers_and_results = [
            (layer, exec_every_layer(sub_layer, group_fn, layer_fn, **kwargs))
            for sub_layer in layer if sub_layer.is_visible()
        ]
        return group_fn(layer, sub_layers_and_results, **kwargs)
    else:
        return layer_fn(layer, **kwargs)


def main():
    psd = PSDImage.open('/home/will/Dropbox/Projects/Bal Maiden Shared/Animation Working Files/act 1 folder/(Footage)/bal maiden moon layered.psd')

    def group_fn(layer, sub_layers_and_results, gltf, **_kws):
        # TODO: Do I need to deal with an offset/transformation here at the group level?
        print("GROUP", layer, layer.offset, sub_layers_and_results)
        node_index = len(gltf.nodes)
        children_node_indices = [child_node_index for sub_layer, child_node_index in sub_layers_and_results]
        gltf.nodes.append(pygltflib.Node(name=layer.name, children=children_node_indices))
        return node_index

    def layer_fn(layer, pixel_center, pixels_per_unit, gltf):
        print("LAYER", layer, layer.offset)

        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            layer_image = layer.composite()
            layer_image.save(f.name)

            image = pygltflib.Image()
            image.uri = f.name
            image_index = len(gltf.images)
            gltf.images.append(image)
            gltf.convert_images(pygltflib.ImageFormat.DATAURI)

        texture_index = len(gltf.textures)
        gltf.textures.append(pygltflib.Texture(
            source=image_index,
        ))


        material_index = len(gltf.materials)
        gltf.materials.append(pygltflib.Material(
            alphaMode=pygltflib.BLEND,
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(baseColorTexture=pygltflib.TextureInfo(index=texture_index)),
            extensions={
                "KHR_materials_unlit": {}
            },
        ))

        return plane_with_offset_and_size(layer.name, material_index, layer.offset, layer.size, pixel_center, pixels_per_unit, gltf)


    gltf = pygltflib.GLTF2()

    indices = array('H', [0, 1, 2, 0, 2, 3])  # uint16
    index_bytes = indices.tobytes()
    gltf.set_binary_blob(index_bytes)
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(index_bytes)))
    gltf.bufferViews.append(
        pygltflib.BufferView(buffer=0, byteOffset=0, byteLength=len(index_bytes), target=pygltflib.ELEMENT_ARRAY_BUFFER),
    )
    gltf.accessors.append(
        pygltflib.Accessor(bufferView=0, componentType=pygltflib.UNSIGNED_SHORT, count=6, type="SCALAR", max=[3], min=[0]),
    )

    pixels_per_unit = 1000
    pixel_center = (psd.size[0] / 2, psd.size[1] / 2)
    node_index = exec_every_layer(psd, group_fn, layer_fn, gltf=gltf, pixel_center=pixel_center, pixels_per_unit=pixels_per_unit)

    scene = pygltflib.Scene(nodes=[node_index])
    gltf.scenes.append(scene)  # scene available at gltf.scenes[0]

    # gltf.convert_images(pygltflib.ImageFormat.BUFFERVIEW)

    gltf.convert_buffers(pygltflib.BufferFormat.BINFILE)   # convert buffers to files
    gltf.buffers[0].uri = "0.bin" # TODO: Why is this not automatic?!
    gltf.save("output.gltf")



if __name__ == "__main__":
    main()
