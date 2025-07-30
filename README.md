This is an experimental program to convert a layered PSD file to a bunch of planes in a GLTF file.

Make sure you have uv, and python3.10. Run the program with:

    uv run main.py PATH/TO/MY.psd

This will generate an `output.glb` file with all neccessary data embedded.

This process can be quite slow and memory intensive for large photoshop documents.
