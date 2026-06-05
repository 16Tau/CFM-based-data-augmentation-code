Hello,

The current folder contains the source code files corresponding to Figure 9 in the paper.

**Files in this folder:**
- `cfm_block_brighten.py` - CFM method with block-based brightening
- `gradcam_block_blur.py` - GradCAM-based block blur visualization
- `mifgsm_block_blur.py` - MI-FGSM with block blur attack
- `pgd_block_blur.py` - PGD with block blur attack

**Usage:**

Step 1: You need to modify the corresponding parameters in the respective attack files, including paths to image folders, model weights, and other configuration settings. For detailed setup instructions, refer to the comments in the source code.

Step 2: Run `cfm_block_brighten.py` to generate adversarial examples using the CFM method with block-based brightening.

Step 3: Run comparison methods (`mifgsm_block_blur.py`, `pgd_block_blur.py`) to compare attack performance.

Step 4: Run `gradcam_block_blur.py` to visualize the block-based attention regions.
