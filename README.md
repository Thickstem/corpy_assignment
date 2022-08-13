# Corpy & Co. Assignments for Engineer Application
applicant : Kanta Suga

## Usage
`pip install -r requirements.txt`

`cd src`

`python assignment.py --backbone [BACKBONE MODEL]`

- backbone is "resnet18" or "wide_resnet50_2" (default "wide_resnet50_2)

## Remarks
- PaDiM method was implemented with reference to the following repositories

https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master/tree/616004b9b7fa6a507d196dd3eeeba9e0a93f160f

PaDiM original paper : https://arxiv.org/pdf/2011.0878a5.pdf

- You can test the Deep Nearest Neighbor (DN2) method in `dn2.ipynb` notebook.

https://arxiv.org/pdf/2002.10445.pdf

