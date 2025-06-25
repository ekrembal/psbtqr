#!/usr/bin/env python3
"""
PSBT QR CLI (psbtqr)

A command-line tool for UR2 PSBT encoding/decoding and QR code generation/scanning.
"""

import sys
import os
import argparse
import base64
import time
from pathlib import Path

# Add the ur2 directory to the path so we can import UR2 modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ur2'))

from ur2.ur_encoder import UREncoder
from ur2.ur_decoder import URDecoder
from ur2.ur import UR
from embit.psbt import PSBT
from urtypes.crypto import PSBT as UR_PSBT


class PSBTQR:
    """
    PSBT QR CLI tool.
    """
    
    def __init__(self):
        self.decoder = URDecoder()
    
    def to_ur2(self, psbt_string):
        """
        Convert PSBT string to UR2 format.
        
        Args:
            psbt_string (str): Base64 encoded PSBT string
        """
        try:
            # Parse the PSBT
            psbt_bytes = base64.b64decode(psbt_string)
            psbt = PSBT.parse(psbt_bytes)
            
            print(f"✅ Parsed PSBT: {len(psbt.inputs)} inputs, {len(psbt.outputs)} outputs")
            
            # Create UR2 encoder for PSBT
            qr_ur_bytes = UR("crypto-psbt", UR_PSBT(psbt.serialize()).to_cbor())
            encoder = UREncoder(ur=qr_ur_bytes, max_fragment_len=30)
            
            print(f"✅ Created UR2 encoder: {encoder.fountain_encoder.seq_len()} parts")
            
            # Display all parts
            print(f"\n📱 UR2 Parts:")
            for i in range(encoder.fountain_encoder.seq_len()):
                part = encoder.next_part()
                print(f"   Part {i+1}: {part}")
            
            # Ask user if they want to see QR codes
            print(f"\n🎬 Would you like to see QR codes? (y/n): ", end="")
            try:
                response = input().lower().strip()
                if response in ['y', 'yes']:
                    self._display_ur2_qr_codes(encoder)
                else:
                    print("QR display skipped.")
            except KeyboardInterrupt:
                print("\nQR display skipped.")
            
            return True
            
        except Exception as e:
            print(f"❌ Error converting to UR2: {e}")
            return False
    
    def _display_ur2_qr_codes(self, encoder, frame_delay=1.0):
        """
        Display UR2 QR codes in animated format.
        
        Args:
            encoder: UREncoder instance
            frame_delay: Delay between frames in seconds
        """
        try:
            import qrcode
            import os
            
            print(f"\n🎬 Starting QR Animation (Frame delay: {frame_delay}s)")
            print("Press Ctrl+C to stop\n")
            
            frame_count = 0
            
            try:
                while True:  # Loop indefinitely
                    # Clear screen (works on most terminals)
                    os.system('clear' if os.name == 'posix' else 'cls')
                    
                    # Get current part
                    part = encoder.next_part()
                    frame_count += 1
                    
                    # Display frame info
                    print(f"Frame {frame_count} | Part {frame_count % encoder.fountain_encoder.seq_len() or encoder.fountain_encoder.seq_len()} of {encoder.fountain_encoder.seq_len()}")
                    print(f"UR2 Part: {part[:30]}...")
                    print("=" * 50)
                    
                    # Display QR code
                    qr_ascii = self._qr_to_ascii(part, size=15)
                    print(qr_ascii)
                    
                    # Display instructions
                    print("=" * 50)
                    print("📱 Scan this QR code with a UR2-compatible wallet")
                    print("🔄 QR codes will cycle continuously")
                    print("⏹️  Press Ctrl+C to stop")
                    
                    time.sleep(frame_delay)
                    
            except KeyboardInterrupt:
                print(f"\n⏹️  Animation stopped after {frame_count} frames")
            except Exception as e:
                print(f"\n❌ Animation error: {e}")
                import traceback
                traceback.print_exc()
                
        except ImportError:
            print("❌ Error: qrcode not installed")
            print("   Install with: pip install qrcode[pil]")
    
    def _qr_to_ascii(self, qr_data, size=20):
        """
        Convert QR code data to ASCII art for terminal display.
        
        Args:
            qr_data (str): Data to encode as QR code
            size (int): Size of the QR code
            
        Returns:
            str: ASCII representation of the QR code
        """
        try:
            import qrcode
            
            # Create QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=1,
                border=1,
            )
            qr.add_data(qr_data)
            qr.make(fit=True)
            
            # Convert to ASCII
            matrix = qr.get_matrix()
            ascii_qr = ""
            
            for row in matrix:
                for cell in row:
                    ascii_qr += "██" if cell else "  "
                ascii_qr += "\n"
            
            return ascii_qr
            
        except ImportError:
            return "❌ qrcode library not available"
        except Exception as e:
            return f"❌ Error generating QR: {e}"
    
    def read_ur2(self):
        """
        Read UR2 parts from camera and decode PSBT.
        """
        try:
            import cv2
            
            print("🎥 Starting UR2 QR Code Scanner")
            print("📱 Point your camera at animated QR codes")
            print("🔄 The scanner will automatically decode UR2 PSBT data")
            print("⏹️  Press 'q' to quit\n")
            
            # Initialize camera
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("❌ Error: Could not open camera")
                return False
            
            # Set camera properties for better QR detection
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Initialize OpenCV QR code detector
            qr_detector = cv2.QRCodeDetector()
            
            scanned_parts = set()
            start_time = time.time()
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Error: Could not read frame from camera")
                        break
                    
                    # Convert frame to grayscale for better QR detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Detect QR codes using OpenCV
                    try:
                        data, bbox, _ = qr_detector.detectAndDecode(gray)
                        
                        if data and len(data) > 0 and data.startswith('UR:'):
                            if data not in scanned_parts:
                                scanned_parts.add(data)
                                self.decoder.receive_part(data)
                                progress = self.decoder.estimated_percent_complete()
                                print(f"🔄 UR2 Part received! Progress: {progress:.1%} ({len(scanned_parts)} parts)")
                                
                                if self.decoder.is_complete():
                                    self._handle_complete_decoding()
                                    break
                                    
                    except Exception:
                        pass
                    
                    # Display the frame with scanning info
                    elapsed_time = time.time() - start_time
                    cv2.putText(frame, f"Parts: {len(scanned_parts)}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if self.decoder.is_complete():
                        cv2.putText(frame, "DECODING COMPLETE!", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        progress = self.decoder.estimated_percent_complete()
                        cv2.putText(frame, f"Progress: {progress:.1%}", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow('UR2 QR Scanner', frame)
                    
                    # Check for quit command
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
            except KeyboardInterrupt:
                print("\n⏹️  Scanning stopped by user")
            finally:
                cap.release()
                cv2.destroyAllWindows()
                
                if not self.decoder.is_complete():
                    print(f"\n📊 Scanning Summary:")
                    print(f"   - Parts scanned: {len(scanned_parts)}")
                    print(f"   - Scanning time: {time.time() - start_time:.1f}s")
                    print(f"   - Decoding incomplete")
            
            return True
            
        except ImportError:
            print("❌ Error: opencv-python not installed")
            print("   Install with: pip install opencv-python")
            return False
        except Exception as e:
            print(f"❌ Error reading UR2: {e}")
            return False
    
    def to_qr(self, text):
        """
        Convert small text to QR code.
        
        Args:
            text (str): Text to encode as QR code
        """
        try:
            import qrcode
            
            print(f"📱 Generating QR code for: {text}")
            
            # Create QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=1,
                border=1,
            )
            qr.add_data(text)
            qr.make(fit=True)
            
            # Convert to ASCII
            matrix = qr.get_matrix()
            ascii_qr = ""
            
            for row in matrix:
                for cell in row:
                    ascii_qr += "██" if cell else "  "
                ascii_qr += "\n"
            
            print(f"\n📱 QR Code:")
            print(ascii_qr)
            
            # Save QR code to file
            filename = f"qr_code_{int(time.time())}.png"
            qr_image = qr.make_image(fill_color="black", back_color="white")
            qr_image.save(filename)
            print(f"💾 QR code saved to: {filename}")
            
            return True
            
        except ImportError:
            print("❌ Error: qrcode not installed")
            print("   Install with: pip install qrcode[pil]")
            return False
        except Exception as e:
            print(f"❌ Error generating QR code: {e}")
            return False
    
    def read_qr(self):
        """
        Read QR code from camera.
        """
        try:
            import cv2
            
            print("🎥 Starting QR Code Scanner")
            print("📱 Point your camera at a QR code")
            print("⏹️  Press 'q' to quit\n")
            
            # Initialize camera
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("❌ Error: Could not open camera")
                return False
            
            # Set camera properties for better QR detection
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Initialize OpenCV QR code detector
            qr_detector = cv2.QRCodeDetector()
            
            start_time = time.time()
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Error: Could not read frame from camera")
                        break
                    
                    # Convert frame to grayscale for better QR detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Detect QR codes using OpenCV
                    try:
                        data, bbox, _ = qr_detector.detectAndDecode(gray)
                        
                        if data and len(data) > 0:
                            print(f"\n📱 QR Code detected!")
                            print(f"📄 Content: {data}")
                            
                            # Save to file
                            filename = f"scanned_qr_{int(time.time())}.txt"
                            with open(filename, 'w') as f:
                                f.write(data)
                            print(f"💾 Content saved to: {filename}")
                            
                            break
                            
                    except Exception:
                        pass
                    
                    # Display the frame with scanning info
                    elapsed_time = time.time() - start_time
                    cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "Point camera at QR code", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow('QR Scanner', frame)
                    
                    # Check for quit command
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
            except KeyboardInterrupt:
                print("\n⏹️  Scanning stopped by user")
            finally:
                cap.release()
                cv2.destroyAllWindows()
            
            return True
            
        except ImportError:
            print("❌ Error: opencv-python not installed")
            print("   Install with: pip install opencv-python")
            return False
        except Exception as e:
            print(f"❌ Error reading QR code: {e}")
            return False
    
    def _handle_complete_decoding(self):
        """
        Handle successful UR2 decoding.
        """
        try:
            print("\n🎉 UR2 Decoding Complete!")
            print("=" * 50)
            
            # Get the decoded data
            decoded_data = self.decoder.result_message()
            
            if decoded_data.type == "crypto-psbt":
                # Handle PSBT data - extract the PSBT bytes from UR_PSBT wrapper
                ur_psbt = UR_PSBT.from_cbor(decoded_data.cbor)
                psbt_bytes = ur_psbt.data
                self._handle_psbt_data(psbt_bytes)
            else:
                print(f"📄 Decoded UR2 data type: {decoded_data.type}")
                print(f"📄 Data: {decoded_data.cbor[:100]}...")
                
        except Exception as e:
            print(f"❌ Error handling decoded data: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_psbt_data(self, psbt_bytes):
        """
        Handle decoded PSBT data.
        """
        try:
            # Parse the PSBT
            psbt = PSBT.parse(psbt_bytes)
            
            print("💰 PSBT Successfully Decoded!")
            print(f"   - Inputs: {len(psbt.inputs)}")
            print(f"   - Outputs: {len(psbt.outputs)}")
            print(f"   - Size: {len(psbt_bytes)} bytes")
            
            # Calculate total input and output amounts
            total_input = sum(input_.witness_utxo.value for input_ in psbt.inputs if input_.witness_utxo)
            total_output = sum(output.value for output in psbt.outputs)
            fee = total_input - total_output if total_input and total_output else 0
            
            print(f"   - Total Input: {total_input:,} sats")
            print(f"   - Total Output: {total_output:,} sats")
            print(f"   - Fee: {fee:,} sats")
            
            # Show output addresses
            print(f"\n📤 Outputs:")
            for i, output in enumerate(psbt.outputs):
                if hasattr(output, 'script_pubkey') and output.script_pubkey:
                    address = output.script_pubkey.address()
                    print(f"   {i+1}. {address}: {output.value:,} sats")
            
            # Print PSBT as base64 string
            psbt_base64 = base64.b64encode(psbt_bytes).decode('utf-8')
            print(f"\n📄 PSBT (Base64):")
            print(f"{psbt_base64}")
            
            # Save PSBT to file
            filename = f"decoded_psbt_{int(time.time())}.psbt"
            with open(filename, 'wb') as f:
                f.write(psbt_bytes)
            print(f"\n💾 PSBT saved to: {filename}")
            
            print(f"\n✅ Decoding complete!")
                
        except Exception as e:
            print(f"❌ Error parsing PSBT: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="PSBT QR CLI - UR2 PSBT encoding/decoding and QR code tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  psbtqr to-ur2 "cHNidP8BAF4CAAAAAXQPOXPXzLgu+VwaYo3UTf9MkcWJZnZI9m+SPguGiVuLAQAAAAD9////AQASegAAAAAAIlEgmQIRpk7kW78uK6z/DBX37H4Pi4OLy3Fau0pXquQSXLUAAAAAAAEBK4CWmAAAAAAAIlEgjRUdGcI2YdKgic2rJaTYmBnIgH+hmxALXCrb6ArEouxBFLSWv7rhSYeBfFPVkr4KpmxFx7lEQ8H3RVE3P5zjTSNGtKD9xF5dq3HfETGHkcGqcSFMXqZhcXPC4ZVFv41UuvlAhtNZ1bZZwt2RUEB+7w1VfkZNohECnP6iunK/nyTfj0nahT6UDAXmktRu2XWjUOhDreTsp+YR9QFtRcw614xFPSIVwVCSm3TBoElUt4tLYDXpel4HiloPKOyW1Ue/7prOgDrAaSC0lr+64UmHgXxT1ZK+CqZsRce5REPB90VRNz+c400jRqwgnAC4DXOZMziPE29FGf7SDLruQVOJmBBwPKIW0jIOIMS6IJlCg+TGSPvt7U7PV5SQYi3URpFS47S8gpBgftNl/Sm+ulKcwCEWUJKbdMGgSVS3i0tgNel6XgeKWg8o7JbVR7/ums6AOsAFAHxGHl0hFplCg+TGSPvt7U7PV5SQYi3URpFS47S8gpBgftNl/Sm+OQG0oP3EXl2rcd8RMYeRwapxIUxepmFxc8LhlUW/jVS6+XjzwwNWAACAAAAAgAAAAIAAAAAAAAAAACEWnAC4DXOZMziPE29FGf7SDLruQVOJmBBwPKIW0jIOIMQ5AbSg/cReXatx3xExh5HBqnEhTF6mYXFzwuGVRb+NVLr5YajXAFYAAIAAAACAAAAAgAAAAAAAAAAAIRa0lr+64UmHgXxT1ZK+CqZsRce5REPB90VRNz+c400jRjkBtKD9xF5dq3HfETGHkcGqcSFMXqZhcXPC4ZVFv41Uuvn/9jQjVgAAgAAAAIAAAACAAAAAAAAAAAABFyBQkpt0waBJVLeLS2A16XpeB4paDyjsltVHv+6azoA6wAEYILSg/cReXatx3xExh5HBqnEhTF6mYXFzwuGVRb+NVLr5AAA="
  psbtqr read-ur2
  psbtqr to-qr "Hello, World!"
  psbtqr read-qr
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # to-ur2 command
    to_ur2_parser = subparsers.add_parser('to-ur2', help='Convert PSBT string to UR2 format')
    to_ur2_parser.add_argument('psbt_string', help='Base64 encoded PSBT string')
    
    # read-ur2 command
    read_ur2_parser = subparsers.add_parser('read-ur2', help='Read UR2 QR codes from camera and decode PSBT')
    
    # to-qr command
    to_qr_parser = subparsers.add_parser('to-qr', help='Convert text to QR code')
    to_qr_parser.add_argument('text', help='Text to encode as QR code')
    
    # read-qr command
    read_qr_parser = subparsers.add_parser('read-qr', help='Read QR code from camera')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Create PSBTQR instance
    psbtqr = PSBTQR()
    
    # Execute command
    if args.command == 'to-ur2':
        success = psbtqr.to_ur2(args.psbt_string)
    elif args.command == 'read-ur2':
        success = psbtqr.read_ur2()
    elif args.command == 'to-qr':
        success = psbtqr.to_qr(args.text)
    elif args.command == 'read-qr':
        success = psbtqr.read_qr()
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 