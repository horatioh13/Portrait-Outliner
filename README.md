<h1 style="text-align: center;">
    <a href="https://github.com/horatioh13/Portrait-Draw/">Portrait Draw</a>
</h1>

<p align="center">
    <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python">
    <img src="https://img.shields.io/badge/os-linux-green" alt="OS">
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/bf7919b1-8231-47a2-beb9-9acc666d06c1" alt="IMG_7143" width="30%" />
  <img src="https://github.com/user-attachments/assets/d8299bb3-1972-426d-9bca-eac0ee84f003" alt="IMG_7134" width="30%" />
  <img src="https://github.com/user-attachments/assets/471f4930-0a1c-4066-8028-8b24ac9755e9" alt="IMG_7122" width="30%" />
</p>


This application takes a picture of a person, and converts the outlines of the picture into scaled Gcode for use on pen plotter or 3d printer with pen plotter attachment.

The image source is specified at the top of multiplemasks.py, you can pull images from your laptop webcam, a usb webcam, a file, or the website thispersondoesnotexist.com.

Images are 256x256 default size, and scaling and nudge can be specified in svgs_to_gcode.py, to position the portrait in the center of the bed.

In run_everything.py, the calibration settings for pendown and pen offset height are set.

I used https://github.com/AndrewSink/pltr_toolhead for the pen plotter, and https://github.com/ShyBoy233/PyGcodeSender to send gcode to my printer over a serial connection. 

credit to: https://github.com/mantasu/face-crop-plus, https://github.com/biubug6/Pytorch_Retinaface, https://github.com/cszn/BSRGAN, and https://github.com/zllrunning/face-parsing.PyTorch, for the AI facial segmentation algorithms.



