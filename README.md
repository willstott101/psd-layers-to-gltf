This is an experimental program to convert a layered PSD or XCF file to a bunch of planes in a GLTF file.

Make sure you have uv, and python3.10. Run the program with:

    uv run main.py PATH/TO/MY.psd PATH/TO/OUTPUT.glb

This will generate an `PATH/TO/OUTPUT.glb` file with all textures embedded.

This process can be quite slow and memory intensive for large photoshop documents.

Layer bounds will be respected in the generated planes (we don't generate new blank pixels).
