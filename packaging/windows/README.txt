NeuralFrame {version} for Windows
==================================

NeuralFrame makes videos and pictures sharper and larger (for example to 4K) using AI, and can make
videos smoother by adding frames (for example to 60 frames per second). Everything runs on your own
computer; nothing is uploaded anywhere.


WHAT YOU NEED
-------------
* Windows 10 (October 2018 update or newer) or Windows 11, 64-bit.
* Recommended: an NVIDIA graphics card (GeForce RTX 20 series or newer) with an up-to-date driver
  (version 580 or newer; get it from https://www.nvidia.com/drivers or the NVIDIA App).
  Without one, NeuralFrame still works on the processor, but neural upscaling is very slow.
* Free disk space: about 6 GB for the program, plus room for the models you download (most are 3 to
  250 MB) and for your output files (4K videos are large).
* Internet: only to download a model the first time you choose it (see "Models" below). After that
  everything works offline.


HOW TO USE IT
-------------
1. Start "NeuralFrame" from the Start menu (or double-click NeuralFrame.exe in this folder).
2. Drag a video or picture onto the window, or click "Open file...".
3. Choose what to do:
     Upscale            - make it bigger and sharper (4K by default; pick another size if you like)
     Frame generation   - make motion smoother (60 frames per second by default) and repair stutter
                          from dropped or frozen frames
     Both               - both of the above
   Pictures can only be upscaled.
4. Check "Save as" (the result goes next to the original by default) and press Start.
   If the model is not on your computer yet, it is downloaded first - the progress bar shows how much,
   how fast and how long it takes - then the work starts.
5. Watch the progress bar. When it says Done, use "Open result" or "Show in folder", and compare
   before/after with the slider at the bottom.

Models
* A "model" is the AI that does the upscaling. Each one is made for a kind of picture. The list shows
  what a model is for first and its name in brackets, grouped like this:
     Filmed video & photos
     Old, damaged or very low quality   (phone and web videos, VHS tapes, compressed TV recordings)
     Cartoons & anime
     AI images & digital art
     Games, textures & pixel art
     Text, documents & comics
     Plain resizing (no AI)
* "Automatic" is the default and usually the best choice. When you open a file, NeuralFrame looks at
  it - cartoon or filmed, grainy or clean, blocky from compression, how sharp it really is - and picks
  the model that suits it. The line under "Model" shows the choice and why.
* "Preset" decides how much time you are willing to spend:
     fast      - only the quick models (good for long videos or a first look)
     balanced  - quick to slow models (the default)
     quality   - also the very slow "transformer" models (HAT-L, DRCT-L, DAT2, ATD, SwinIR-L), which
                 create the most convincing detail; expect them to take many times longer
* You can also pick a model yourself. "fast", "medium", "slow" and "very slow" say how long it takes;
  hover over a model to read what it is good at. "[personal use only]" means its licence does not
  allow commercial use.
* Models are NOT included in the download of this program: each model is downloaded the first time
  you choose it. The line under "Model" says whether it is already on your computer ("Download now"
  gets it straight away).
* "Models..." (or the Models menu, Ctrl+M) opens the model manager: search for a model, see which ones
  you have, download them ahead of time (with a progress bar and a Cancel button), delete them to free
  space, check for updates, or use a model file you already have ("Use a file I have...").
* Upscaling cannot bring back detail that was never recorded: the models draw new, believable detail
  (skin, hair, fabric, edges). On very blurry or tiny videos the result looks cleaner and sharper, but
  not like a real 4K recording.
* For VERY bad videos and photos (tiny, blurry, heavily compressed), the group "Generative restoration"
  has research models (SeedVR2, FlashVSR, DOVE, OSEDiff, SUPIR...). They re-create the picture and invent
  realistic detail, so the result can differ from what was really filmed. They need an NVIDIA card, are
  very slow, and install a separate engine (several GB) the first time you use one - the program asks
  first and shows the progress. "seedvr2-3b-q4" is the one made for 8 GB graphics cards.
* "Restore faces" (under the model) restores small or blurry faces in the result.
* Models menu > "Add your own model file..." lets you use a new model you found yourself (for example
  on openmodeldb.info, .pth or .safetensors files). It is checked, copied and then appears in the list.

Tips
* The FIRST time you use a model on your graphics card, NeuralFrame optimises it for your card. This
  takes a few minutes once ("building TensorRT engine"); later runs start immediately.
* Long video? You can press Cancel at any time and then Resume while the program is still open:
  finished parts are kept. (If you close the program in the middle, the job starts over next time.)
* "Scan for dropped frames" shows whether a video has stutter that frame generation will repair.


IF SOMETHING GOES WRONG
-----------------------
* Windows says "Windows protected your PC": click "More info", then "Run anyway". The program is not
  signed with a paid certificate; this warning does not mean it is harmful.
* Very slow or the computer becomes sluggish: close other programs that use the graphics card (games,
  video editors) and try again. Updating the NVIDIA driver also helps.
* An error message appears: it usually says what to do. A detailed log is written to
  %LOCALAPPDATA%\neuralframe\logs\neuralframe-gui.log - include it if you report a problem.


FOR ADVANCED USERS
------------------
neuralframe-cli.exe in this folder is the command-line version ("NeuralFrame command line" in the Start
menu). Examples:
    neuralframe-cli.exe upscale "C:\Videos\clip.mp4" -o "C:\Videos\clip_4k.mkv"
    neuralframe-cli.exe upscale clip.mp4 -o clip_4k60.mkv --fps 60
    neuralframe-cli.exe interpolate clip.mp4 -o clip_60fps.mkv
    neuralframe-cli.exe recommend clip.mp4      (which model "automatic" would use, and why)
    neuralframe-cli.exe models list             (all models; add your own with "models add FILE")
    neuralframe-cli.exe serve          (web interface on http://127.0.0.1:8765)
    neuralframe-cli.exe doctor         (shows your graphics card and what NeuralFrame will use)
Run "neuralframe-cli.exe --help" for everything.

Data NeuralFrame keeps on your computer (safe to delete; it is rebuilt when needed):
    %LOCALAPPDATA%\neuralframe   (downloaded and optimised models, temporary files, logs; the Models
                                 window shows how much space the models use and can delete them)
    %APPDATA%\neuralframe        (models you added yourself: models.yaml)


UNINSTALLING
------------
Settings > Apps > Installed apps > NeuralFrame > Uninstall (or delete this folder if you use the
portable version). Then delete %LOCALAPPDATA%\neuralframe to remove cached data as well.


CREDITS AND LICENCES
--------------------
See the "licenses" folder. NeuralFrame includes FFmpeg (GPL; source: https://ffmpeg.org), PyTorch (BSD),
Qt for Python (LGPL), spandrel (MIT), NVIDIA CUDA/cuDNN/TensorRT runtime libraries (NVIDIA licence terms)
and AI models: Real-ESRGAN (BSD-3-Clause), BSRGAN, SwinIR, RealPLKSR, RealBasicVSR (Apache-2.0), HAT-L,
DRCT-L, DAT2, ATD trained by Philip Hofmann (CC BY 4.0) and RIFE (MIT); the full list with sources is in
licenses\THIRD-PARTY-NOTICES.txt.
