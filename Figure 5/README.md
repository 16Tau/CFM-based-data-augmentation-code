Hello,

The current folder contains the source code files corresponding to Figure 5 in the paper.

**Files in this folder:**
- `CFM_derivation_method.py` - Implementation of the CFM derivation method
- `CFM_Gaussian.py` - CFM method with Gaussian augmentation
- `attack.py` - Base attack class implementation

**Usage:**

Step 1: You need to modify the corresponding parameters in `CFM_derivation_method.py`, including paths to image folders, model weights, and other configuration settings. For detailed setup instructions, refer to the comments in the source code.

Step 2: Run `CFM_derivation_method.py` to generate adversarial examples for the experiments shown in Figure 5.

Step 3: Run `CFM_Gaussian.py` to generate adversarial examples with Gaussian augmentation for comparison.
