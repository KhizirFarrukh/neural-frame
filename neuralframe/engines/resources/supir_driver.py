"""NeuralFrame driver for SUPIR: points the project's configuration at the downloaded SDXL base and the
SUPIR checkpoint (and lets the CLIP encoders download themselves), then runs its test.py unchanged."""

import re
import runpy
import sys
from pathlib import Path


def main() -> None:
    argv = sys.argv[1:]

    def take(flag: str) -> str:
        i = argv.index(flag)
        value = argv[i + 1]
        del argv[i:i + 2]
        return value

    sdxl, supir = take("--sdxl"), take("--supir")
    sign = argv[argv.index("--SUPIR_sign") + 1] if "--SUPIR_sign" in argv else "Q"
    here = Path(__file__).resolve().parent
    ckpt = here / "CKPT_PTH.py"
    ckpt.write_text("LLAVA_CLIP_PATH = None\nLLAVA_MODEL_PATH = None\nSDXL_CLIP1_PATH = None\n"
                    "SDXL_CLIP2_CKPT_PTH = None\n", encoding="utf-8")
    options = here / "options" / "SUPIR_v0.yaml"
    text = options.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^SDXL_CKPT:.*$", f"SDXL_CKPT: {Path(sdxl).as_posix()}", text)
    text = re.sub(rf"(?m)^SUPIR_CKPT_{sign}:.*$", f"SUPIR_CKPT_{sign}: {Path(supir).as_posix()}", text)
    options.write_text(text, encoding="utf-8")
    sys.argv = [str(here / "test.py"), *argv]
    sys.path.insert(0, str(here))
    runpy.run_path(str(here / "test.py"), run_name="__main__")


if __name__ == "__main__":
    main()
