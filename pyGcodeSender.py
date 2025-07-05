import argparse
import os
import sys
import time

import serial
from serial.tools import list_ports
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        prog="pyGcodeSender",
        description="A simple python script to send gcode file using serial to various machines."
    )

    parser.add_argument("filename", help="gcode to be sent")
    parser.add_argument("-p", "--port", help="serial port", metavar="port name")
    parser.add_argument("-b", "--baudrate", default=115200, type=int, help="baud rate, default 115200", metavar="baud rate")

    args = parser.parse_args()

    # Validate the filename.
    if os.path.exists(args.filename): # Whether exist.
        if os.path.isfile(args.filename): # Whether is a file.
            if args.filename.strip().lower().endswith(".gcode"): # Whether is a gcode file.
                pass
            else:
                print(f"{args.filename} is highly not a gcode file(end with .gcode). Please check it.")
                sys.exit()
        else:
            print(f"{args.filename} is not a file. Please check the filename.")
            sys.exit()
    else:
        print(f"{args.filename} does not exist. Please check the filename.")
        sys.exit()

    # Automatically detect available serial port.
    if len(list_ports.comports()) == 0: # No port detected.
        print("There are no ports available. Please check the connection.")
        sys.exit()
    elif len(list_ports.comports()) == 1: # Only one port detected.
        if args.port is None:
            port = list_ports.comports()[0].name
        else:
            ports = [list_ports.comports()[i].name.strip().lower() for i in range(len(list_ports.comports()))]
            if args.port.strip().lower() in ports:
                port = args.port
            else:
                print(f"Port {args.port} is not in available port list. Please check spelling.")
                print("Available port names:")
                for p in list_ports.comports():
                    print(p.name)
                sys.exit()
    else: # More than one ports are detected, need to specify the port name manually.
        if args.port is None:
            print("There are many ports available. Please specify the port name manually.")
            print("Available port names:")
            for p in list_ports.comports():
                print(p.name)
            sys.exit()
        else:
            ports = [list_ports.comports()[i].name.strip().lower() for i in range(len(list_ports.comports()))]
            if args.port.strip().lower() in ports:
                port = args.port
            else:
                print(f"Port {args.port} is not in available port list. Please check spelling.")
                print("Available port names:")
                for p in list_ports.comports():
                    print(p.name)
                sys.exit()

    # Start of the program.
    print('='*130)
    print("Welcome to use this simple python script to send gcode file using serial.")
    print()

    # Check settings before continue.
    print("Current setting:")
    print(f"\tFilename: {args.filename}")
    print(f"\tPort name: {port}")
    print(f"\tBaud rate: {args.baudrate}")
    print()
    while(1):
        command = input("Would you like to continue with these settings?[y/n]")
        if command.strip().lower() == 'y':
            break
        elif command.strip().lower() =='n':
            sys.exit()
        else:
            print("Oooops! Please input yes or no.")
            continue

    # Loading codes in file
    try:
        f = open(args.filename, 'r')
        codes = [code for code in f]
        f.close()
        print()
        print(f"Codes in file {args.filename} have been loaded successfully.")
    except:
        print(f"Cannot open {args.filename}.")
        sys.exit()

    # Connectting serial port.
    print()
    print(f"Trying to connect to port {port}.")
    try:
        s = serial.Serial('/dev/ttyUSB0', 115200)
        print(f"Port {port} has been successfully connected.")
    except:
        print(f"Cannot connect to port: {port}. Please check it.")
        print(f"The port is highly beening occupied by another progrom or the baud rate is wrong.")
        sys.exit()
    
    # Sending codes.
    print()
    print("Sending codes.")
    s.write(b"\r\n\r\n") # Wake up microcontroller
    time.sleep(1)
    s.reset_input_buffer()

    tqdm4codes = tqdm(codes, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', unit=" codes", ncols=130)
    for code in tqdm4codes:
        tqdm4codes.set_postfix(gcode=code) # Show gcode at postfix
        if code.strip().startswith(';') or code.isspace() or len(code) <=0:
            continue
        else:
            s.write((code+'\n').encode())
            while(1): # Wait untile the former gcode has been completed.
                if s.readline().startswith(b'ok'):
                    break
    s.close() 
    print()
    print("All codes have been sent successfully.")
    print("Congratulation!!!")

    # End of the program.
    print()
    print("Welcome to use this simple script again. Best wishes.")
    print('='*130)


if __name__ =="__main__":
    main()
    
