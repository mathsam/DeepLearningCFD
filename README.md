# DeepLearningCFD
Applying deep learning in Computational Fluid Dynamics

## Scripts

run_pyqg.py: produce training/testing data from QG simulation

trainer.py: train Super Resolution Deep Learning model from produced data

## Experiments
### Initial Trial
```
Generator: 16 layers; small_kernel_size=5; large_kernel_size=15
Discriminator: 8 layers; kernel_size=5
learning rate = 1e-4
scaling factor = 4x
beta = 0.01 (generator_loss = content_loss + beta * adversarial_loss)
```

First, only train generator. At epoch=30, looking at potential voriticity field and its spectrum, super-resolution result is not that different from Bivariate Spline interpolation at this time.
![2d field](results/Sep8/super_res_field2d_compare.png)
![spectrum](results/Sep8/super_res_vorticity_spec_compare.png)

Continue training with GAN. At epoch 30 Small scale structure starts to show up. In this physical space, the small scale structure doesn't look realistic yet, though it fills up some gap in the spectral space.
![2d field](results/Sep8/epoch30_gan_super_res_field2d_compare.png)
![spectrum](results/Sep8/epoch30_gan_super_res_vorticity_spec_compare.png)

At epoch 100
![2d field](results/Sep8/epoch100_gan_super_res_field2d_compare.png)
![spectrum](results/Sep8/epoch100_gan_super_res_vorticity_spec_compare.png)

At epoch 200, now GAN super-resolution model output looks quite realstic. The fine structure is very close to the high resolution image. I was expecting the super-resolution model to generate some realistic fine structure, but not in the exact same location as the actual high resolution image as this information could be lost in the low resolution image. But at 4x scaling factor, it could be the case that this information is still preserved. GAN helps the model to focus on learning small scale structure. This can also been seen from 
![2d field](results/Sep8/epoch200_gan_super_res_field2d_compare.png)
![spectrum](results/Sep8/epoch200_gan_super_res_vorticity_spec_compare.png)


Learning curve. Content loss keeps decreasing, which means GAN helps generator to generate super-resolution image exactly the same as original high-resolution image. Adversarial/Disriminator loss fluctuates.

![learning curve](results/Sep8/learning_curve.png)