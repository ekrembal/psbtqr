# PSBT QR - UR2 QR Code Tools

A standalone Python tool for UR2 PSBT encoding/decoding and QR code generation/scanning. This project provides a simple command-line interface for working with Bitcoin PSBTs and QR codes using the UR2 standard.

## What's Included

- **`ur2/`** - Complete UR2 implementation
- **`psbtqr.py`** - Main CLI tool with 4 commands
- **`requirements.txt`** - Python dependencies

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Main CLI Tool: `psbtqr.py`

The main CLI tool provides 4 commands in a single interface:

```bash
# Convert PSBT string to UR2 format
python psbtqr.py to-ur2 "cHNidP8BAF4CAAAAAXQPOXPXzLgu+VwaYo3UTf9MkcWJZnZI9m+SPguGiVuLAQAAAAD9////AQASegAAAAAAIlEgmQIRpk7kW78uK6z/DBX37H4Pi4OLy3Fau0pXquQSXLUAAAAAAAEBK4CWmAAAAAAAIlEgjRUdGcI2YdKgic2rJaTYmBnIgH+hmxALXCrb6ArEouxBFLSWv7rhSYeBfFPVkr4KpmxFx7lEQ8H3RVE3P5zjTSNGtKD9xF5dq3HfETGHkcGqcSFMXqZhcXPC4ZVFv41UuvlAhtNZ1bZZwt2RUEB+7w1VfkZNohECnP6iunK/nyTfj0nahT6UDAXmktRu2XWjUOhDreTsp+YR9QFtRcw614xFPSIVwVCSm3TBoElUt4tLYDXpel4HiloPKOyW1Ue/7prOgDrAaSC0lr+64UmHgXxT1ZK+CqZsRce5REPB90VRNz+c400jRqwgnAC4DXOZMziPE29FGf7SDLruQVOJmBBwPKIW0jIOIMS6IJlCg+TGSPvt7U7PV5SQYi3URpFS47S8gpBgftNl/Sm+ulKcwCEWUJKbdMGgSVS3i0tgNel6XgeKWg8o7JbVR7/ums6AOsAFAHxGHl0hFplCg+TGSPvt7U7PV5SQYi3URpFS47S8gpBgftNl/Sm+OQG0oP3EXl2rcd8RMYeRwapxIUxepmFxc8LhlUW/jVS6+XjzwwNWAACAAAAAgAAAAIAAAAAAAAAAACEWnAC4DXOZMziPE29FGf7SDLruQVOJmBBwPKIW0jIOIMQ5AbSg/cReXatx3xExh5HBqnEhTF6mYXFzwuGVRb+NVLr5YajXAFYAAIAAAACAAAAAgAAAAAAAAAAAIRa0lr+64UmHgXxT1ZK+CqZsRce5REPB90VRNz+c400jRjkBtKD9xF5dq3HfETGHkcGqcSFMXqZhcXPC4ZVFv41Uuvn/9jQjVgAAgAAAAIAAAACAAAAAAAAAAAABFyBQkpt0waBJVLeLS2A16XpeB4paDyjsltVHv+6azoA6wAEYILSg/cReXatx3xExh5HBqnEhTF6mYXFzwuGVRb+NVLr5AAA="

# Read UR2 QR codes from camera and decode PSBT
python psbtqr.py read-ur2

# Convert text to QR code
python psbtqr.py to-qr "Hello, World!"

# Read QR code from camera
python psbtqr.py read-qr
```

**Available Commands:**
- **`to-ur2 {PSBT_STRING}`** - Convert base64 PSBT string to UR2 format (shows all parts)
- **`read-ur2`** - Scan UR2 animated QR codes from camera and decode PSBT
- **`to-qr {SMALL_TEXT}`** - Convert text to QR code (displays ASCII and saves PNG)
- **`read-qr`** - Scan QR code from camera and extract content

## UR2 Implementation

The `ur2/` directory contains a complete UR2 implementation with:

- **UR Encoder/Decoder** - Core UR2 encoding and decoding
- **Fountain Encoder/Decoder** - Fountain code implementation for animated QR
- **Bytewords** - Bytewords encoding/decoding
- **CBOR Lite** - Lightweight CBOR implementation
- **Utilities** - Helper functions and constants

## Key Features

1. **No External Dependencies** - Complete UR2 implementation included
2. **Standalone Tool** - Can be used in any Python project
3. **Simplified Interface** - Focused on PSBT handling
4. **Easy Integration** - Can be dropped into any Python project
5. **OpenCV-based QR Scanning** - No zbar dependency issues
6. **Single CLI Tool** - `psbtqr.py` provides all functionality in one interface

## Example Integration

```python
from ur2.ur_encoder import UREncoder
from ur2.ur import UR
from embit.psbt import PSBT
from urtypes.crypto import PSBT as UR_PSBT

# Create UR2 encoder for PSBT
psbt = PSBT.parse(psbt_bytes)
qr_ur_bytes = UR("crypto-psbt", UR_PSBT(psbt.serialize()).to_cbor())
encoder = UREncoder(ur=qr_ur_bytes, max_fragment_len=30)

# Get QR code parts
part = encoder.next_part()
print(f"QR Part: {part}")
```

## Dependencies

### Required Dependencies
- **embit** - Bitcoin PSBT handling
- **urtypes** - UR type definitions
- **qrcode** - QR code generation
- **opencv-python** - Camera access and QR detection

### Optional Dependencies
- **Pillow** - Better QR code display

## Troubleshooting

### Camera Issues
- Make sure your camera is not in use by another application
- Try different camera indices (0, 1, 2) in the scanner code
- Check camera permissions on your system

### QR Detection Issues
- Ensure good lighting conditions
- Hold QR codes steady and at a reasonable distance
- Make sure QR codes are clearly visible and not damaged

## License

The UR2 implementation is licensed under the "BSD-2-Clause Plus Patent License" as per the original Foundation Devices implementation. 