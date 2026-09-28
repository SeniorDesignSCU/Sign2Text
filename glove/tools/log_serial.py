#!/usr/bin/env python3
"""Record glove serial CSV to a labeled file.

    python glove/tools/log_serial.py --port /dev/ttyACM0 --label FIST --seconds 3 --signer karan

Writes data/glove/<LABEL>/<signer>_<timestamp>.csv. Press Ctrl-C to stop early.
Upload these CSVs to Edge Impulse (CSV wizard) or train locally.
"""
import argparse, sys, time
from pathlib import Path

import serial  # pyserial

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--label", required=True)
    ap.add_argument("--signer", required=True)
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    out_dir = ROOT / "data" / "glove" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.signer}_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    with serial.Serial(args.port, args.baud, timeout=1) as ser, open(out, "w") as f:
        ser.reset_input_buffer()
        header_written, t0, n = False, time.time(), 0
        print(f"recording {args.label} for {args.seconds}s -> {out.name}")
        try:
            while time.time() - t0 < args.seconds:
                line = ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("t_ms"):
                    if not header_written:
                        f.write(line + "\n"); header_written = True
                    continue
                if not header_written:
                    f.write("t_ms,f0,f1,f2,f3,f4,ax,ay,az,gx,gy,gz\n"); header_written = True
                f.write(line + "\n"); n += 1
        except KeyboardInterrupt:
            pass
    print(f"saved {n} samples ({n/max(args.seconds,1e-3):.0f} Hz)")


if __name__ == "__main__":
    sys.exit(main())
