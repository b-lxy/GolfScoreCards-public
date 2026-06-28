# OCR on Golf Score Cards

Author: Xin Ying Leong (https://www.linkedin.com/in/blxy)

> This project is still in progress... working on cell recognition accuracy...

This project builds a specialized OCR pipeline to extract handwritten scores from a single snapshot for subsequent score digitalization.
Golf scores often include messy handwritting, superscripts and shapes, making it a challenge for the recognition of the main digit.
The pipeline includes the following stages:
- Image registration with ORB on a defined template with previously defined cropping points, which are used to crop score cells. A double homography is used to match the left page and the right page separately, avoiding the crease of the card. Some parts of the registered image (right) are masked for privacy reasons.
![Image1](displays/sample_registration.png)

- Score cells are passed through a Convolutional Neural Network (CNN) for classification, classes include digits 1-9 and the blank cell (`-`). Confidence scores are returned with the prediction for threshold-based decision making.

![Image2](displays/cell_rec.png)
