Required figures for the paper:

1. pipeline_overview.pdf  (Figure 1 - full page width)
   Draw 5 boxes connected by right arrows:
   [Selfie] -> [MediaPipe Align] -> [YOLO+Color] -> [DINOv2 x8 Regions]
                                                         -> [Severity MLP]
                                                         -> [Bayesian Engine]
                                                         -> [Deficiency Output]
   Recommended tool: draw.io (free, browser-based)
   Export as PDF for best quality in LaTeX.

2. confusion_matrix.pdf  (Figure 2 - optional but good)
   Use the confusion_matrix_normalized.png from your YOLO training results.
   Located in: runs/detect/runs/facefuel_v2/yolo_detector_r2/
   Convert to PDF or include as PNG with \includegraphics.

3. ablation_chart.pdf  (Figure 3 - optional)
   Simple bar chart of Table 3 values.
   Can be made in Excel, matplotlib, or even PowerPoint.
