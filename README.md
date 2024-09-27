This application takes a picture of a person, and converts the outlines of the picture into scaled Gcode for use on pen plotter or 3d printer with pen plotter attachment.

The image source is specified at the top of multiplemasks.py, you can pull images from your laptop webcam, a usb webcam, a file, or the website thispersondoesnotexist.com.

Images are 256x256 default size, and scaling and nudge can be specified in svgs_to_gcode.py, to position the portrait in the center of the bed.

In run_everything.py, the calibration settings for pendown and pen offset height are set.

I used https://github.com/AndrewSink/pltr_toolhead for the pen plotter, and https://github.com/ShyBoy233/PyGcodeSender to send gcode to my printer over a serial connection. 

credit to: https://github.com/mantasu/face-crop-plus,https://github.com/biubug6/Pytorch_Retinaface,https://github.com/cszn/BSRGAN, and https://github.com/zllrunning/face-parsing.PyTorch, for the AI facial segmentation algorithms.



