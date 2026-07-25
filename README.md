# OCR on Golf Score Cards

Author: Xin Ying Leong (https://www.linkedin.com/in/blxy)

This project builds a specialized OCR pipeline to extract handwritten scores from a single snapshot for subsequent score digitalization.
Golf scores often include messy handwritting, superscripts and shapes, making it a challenge for the recognition of the main digit.
The pipeline includes the following stages:

> Image Registration onto a template
- Image registration with SIFT + orientation invariance on a defined template with hand-picked defined cropping points, which are used to crop score cells. A double homography is used to match the left page and the right page separately, avoiding the crease of the card. Some parts of the registered image are masked for privacy reasons.

<img src="displays/image_match_new.png" width="60%">
<img src="displays/image_reg_new.png" width="38%">

- Older version: Image registration with ORB on a defined template with hand-picked defined cropping points, which are used to crop score cells. A double homography is used to match the left page and the right page separately, avoiding the crease of the card. Some parts of the registered image are masked for privacy reasons.

![Image_matched](displays/image_match_old.png)
![Image_registered](displays/image_reg_old.png)

> Single-cell Recognition

- Score cells are passed through a Fine-tuned Resnet-18 for classification, classes include digits 1-9 and the blank cell (`-`). Confidence scores are returned with the prediction for threshold-based decision making. (0 errors)

![Image_recognition](displays/cell_rec_resnet18.png)

- Older version: Score cells are passed through a Convolutional Neural Network (CNN) for classification, classes include digits 1-9 and the blank cell (`-`). Confidence scores are returned with the prediction for threshold-based decision making. (7 errors)

![Image_recognition](displays/cell_rec.png)

---

Simple tools are also created to faciliate hand-picking points for a given template [manual_pointing.py](scripts/tools/manual_pointing.py), and to annotate cells prior to CNN training [cell_labeller.py](scripts/tools/cell_labeller.py)

<img src="displays/cell_labeller.png" width="50%">
<img src="displays/template_data2.png" width="40%">

---

The most recent pipeline and API has been generalized, which also works on a different template type:
<img src="displays/template_data1.png" width="40%">

- Example 1
![Image_data1im4](displays/image_match_data1_im4.png)
![cell_rec_data1im4](displays/cell_rec_resnet18_data1_im4.png)
- Example 2
![Image_data1im15](displays/image_match_data1_im15.png)
![cell_rec_data1im15](displays/cell_rec_resnet18_data1_im15.png)