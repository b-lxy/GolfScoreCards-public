# OCR on Golf Score Cards

Author: Xin Ying Leong (https://www.linkedin.com/in/blxy)

> This project is still in progress... working on cell recognition accuracy...

This project builds a specialized OCR pipeline to extract handwritten scores from a single snapshot for subsequent score digitalization.
Golf scores often include messy handwritting, superscripts and shapes, making it a challenge for the recognition of the main digit.
The pipeline includes the following stages:
- Image registration with ORB on a defined template with hand-picked defined cropping points, which are used to crop score cells. A double homography is used to match the left page and the right page separately, avoiding the crease of the card. Some parts of the registered image (right) are masked for privacy reasons.

![Image_matched](displays/image_match.png)
![Image_registered](displays/image_reg.png)

- Score cells are passed through a Convolutional Neural Network (CNN) for classification, classes include digits 1-9 and the blank cell (`-`). Confidence scores are returned with the prediction for threshold-based decision making.

![Image_recognition](displays/cell_rec.png)

Simple tools are also created to faciliate hand-picking points for a given template [manual_pointing.py](scripts/tools/manual_pointing.py), and to annotate cells prior to CNN training [cell_labeller.py](scripts/tools/cell_labeller.py)

<img src="displays/cell_labeller.png" width="50%">
<img src="displays/template.png" width="40%">