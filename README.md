# Portrait Drawing Robot

<img src="https://img.shields.io/badge/python-3.12-blue" alt="Python">  <img src="https://img.shields.io/badge/os-linux-green" alt="OS">


<p align="center">
  <img src="https://github.com/user-attachments/assets/bf7919b1-8231-47a2-beb9-9acc666d06c1" alt="IMG_7143" width="30%" />
  <img src="https://github.com/user-attachments/assets/d8299bb3-1972-426d-9bca-eac0ee84f003" alt="IMG_7134" width="30%" />
  <img src="https://github.com/user-attachments/assets/471f4930-0a1c-4066-8028-8b24ac9755e9" alt="IMG_7122" width="30%" />
</p>


This application takes a picture of a person, and converts the outlines of the picture into scaled Gcode for use on pen plotter or 3d printer with pen plotter attachment.

## Installation
Create and activate a virtual environment:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```
Clone repository
```bash
git clone https://github.com/horatioh13/Portrait-Draw.git
```
Install requirements
```bash
pip install -r requirements.txt
```

## Usage
Run default script by pulling image from thispersondoesnotexist.com 
```bash
python3 run_everything.py
```

To change the image source, edit the run_everything.py file. Possible image sources include: "laptopwebcam", "usbwebcam", "file", or "thispersondoesnotexist".

Images are 256x256 default size, and scaling and nudge can be specified in svgs_to_gcode.py, to position the portrait in the center of the bed.

In run_everything.py, the calibration settings for pendown and pen offset height are set.

## Hardware
I used [Pltr Toolhead V2](https://github.com/AndrewSink/pltr_toolhead) for the pen plotter, and [PyGcodeSender](https://github.com/ShyBoy233/PyGcodeSender) to send gcode to my printer over a serial connection. 


## References

This package uses the code from the following repositories:
* [Face Crop Plus](https://github.com/mantasu/face-crop-plus) - Image preprocessing package for automatic face alignment and cropping with additional features.





