# DLO asset status

## Step 0 (initial): assets auth-gated
At Step 0 the DLO-Lab textured assets were unavailable: `genesis/assets/dlo-lab` was absent and the upstream source is auth-gated by UMass SharePoint (HTTP 401). Step-0's shipped render therefore used the asset-free DLO-Lab README quick-start scene (two ropes, fixed ends, default surfaces) rendered headlessly via NVIDIA EGL. All Step-0 smoke probes are unaffected (they do not require textures).

## Step 1 onward: assets provided by the owner and installed
The owner provided the textured asset payload locally at `~/Downloads/dlo-lab.zip` (143 MB). It was unpacked into the exact path the DLO-Lab code expects:

- Placement: `~/DLO-Lab/genesis/assets/dlo-lab/` (subdirs `textures/`, `exrs/`, `meshes/`, `ropes/`, `target_pos/`).
- Command: `unzip -o ~/Downloads/dlo-lab.zip -d ~/DLO-Lab/genesis/assets/` (the archive's top-level is `dlo-lab/`).
- Key files present: `textures/rope01.png`, `textures/rope02.png`, `textures/rope03.png`, `exrs/brown_photostudio_02_4k.exr`, `meshes/wooden_table.glb`.
- Clone cleanliness preserved: `~/DLO-Lab/.gitignore` ignores `genesis/assets/dlo-lab/`, so `git -C ~/DLO-Lab status --porcelain` remains empty (0 lines) — the assets are data, not a tracked source diff.

From Step 1 onward, rendered deliverables (task demo mp4s, rollout mp4s) use these textured assets.
