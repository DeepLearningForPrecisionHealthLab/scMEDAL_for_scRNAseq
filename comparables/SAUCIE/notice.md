## Third-party code notice

This directory /scMEDAL_for_scRNAseq/comparables/SAUCIE includes limited code adapted from the original SAUCIE repository:
https://github.com/KrishnaswamyLab/SAUCIE

Reference:
Amodio, M., van Dijk, D., Srinivasan, K., Chen, W. S., Mohsen, H., Moon, K. R., Campbell, A., Zhao, Y., Wang, X., Venkataswamy, M., Desai, A., Ravi, V., Kumar, P., Montgomery, R., Wolf, G., & Krishnaswamy, S. (2019). Exploring single-cell data with deep multitasking neural networks. Nature methods, 16(11), 1139–1145. https://doi.org/10.1038/s41592-019-0576-7

The adapted SAUCIE code remains subject to the original non-commercial SAUCIE license, provided in `SAUCIE_LICENSE.md`.

Modification made to the adapted SAUCIE code:
- `model.SAUCIE._build_layers` was modified to use a configurable n-dimensional latent space instead of the original 2-dimensional latent space.

The file `saucie.py` in this directory was developed for this project and is not copied from the original SAUCIE repository.

The original copyright and license notices for reused or adapted SAUCIE code are retained in `SAUCIE_LICENSE.md`.