# Deepspline: Data-driven reconstruction of parametric curves and surfaces

This is the official PyTorch implementation of Deepspline. For technical details, please refer to:  
----------------------- ------------------------------------
**Deepspline: Data-driven reconstruction of parametric curves and surfaces**  
[Jun Gao](http://www.cs.toronto.edu/~jungao/) <sup>1,2,3</sup>, [Chengcheng Tang](), [Vignesh Ganapathi-Subramanian](), [Jiahui Huang](), [Hao Su](), [Leonidas J. Guibas]()  

**[[Paper](https://arxiv.org/pdf/1901.03781.pdf)] **

* Reconstruction of geometry based on different input modes, such as images or point clouds, has been instrumental in the
development of computer aided design and computer graphics. Optimal implementations of these applications have traditionally
involved the use of spline-based representations at their core. Most such methods attempt to solve optimization problems that
minimize an output-target mismatch. However, these optimization techniques require an initialization that is close enough, as
they are local methods by nature. We propose a deep learning architecture that adapts to perform spline fitting tasks accordingly,
providing complementary results to the aforementioned traditional methods. We showcase the performance of our approach, by
reconstructing spline curves and surfaces based on input images or point clouds.
----------------------- ------------------------------------



If you use this code, please cite our paper:

    @article{gao2019deepspline,
    title={Deepspline: Data-driven reconstruction of parametric curves and surfaces},
    author={Gao, Jun and Tang, Chengcheng and Ganapathi-Subramanian, Vignesh and Huang, Jiahui and Su, Hao and Guibas, Leonidas J},
    journal={arXiv preprint arXiv:1901.03781},
    year={2019}
    }
    

# News
Due to many code requires of this paper, we release an initial version of the code. We are still working on cleaning the code base. 

# License

This work is licensed under a *GNU GENERAL PUBLIC LICENSE Version 3* License.
