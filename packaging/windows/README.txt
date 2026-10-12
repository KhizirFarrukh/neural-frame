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
* Free disk space: about 6 GB for the program, plus room for your output files (4K videos are large).


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
5. Watch the progress bar. When it says Done, use "Open result" or "Show in folder", and compare
   before/after with the slider at the bottom.

Tips
* Model: "Automatic" is a good start. realesr-animevideov3 is fast and great for cartoons and anime;
  realesrgan-x4plus gives the most detail on real-world video and photos but is slower.
* The FIRST time you use a model on your graphics card, NeuralFrame optimises it for your card. This
  takes a few minutes once ("building TensorRT engine"); later runs start immediately.
* Long video? You can press Cancel at any time and later press Resume: finished parts are kept.
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
    neuralframe-cli.exe serve          (web interface on http://127.0.0.1:8765)
    neuralframe-cli.exe doctor         (shows your graphics card and what NeuralFrame will use)
Run "neuralframe-cli.exe --help" for everything.

Data NeuralFrame keeps on your computer (safe to delete; it is rebuilt when needed):
    %LOCALAPPDATA%\neuralframe   (optimised models, temporary files, logs)


UNINSTALLING
------------
Settings > Apps > Installed apps > NeuralFrame > Uninstall (or delete this folder if you use the
portable version). Then delete %LOCALAPPDATA%\neuralframe to remove cached data as well.


CREDITS AND LICENCES
--------------------
See the "licenses" folder. NeuralFrame includes FFmpeg (GPL; source: https://ffmpeg.org), PyTorch (BSD),
Qt for Python (LGPL), NVIDIA CUDA/cuDNN/TensorRT runtime libraries (NVIDIA licence terms) and AI models
Real-ESRGAN (BSD-3-Clause), RealBasicVSR (Apache-2.0) and RIFE (MIT).
