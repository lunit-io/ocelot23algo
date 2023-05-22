# Pytorch-based U-Net example
Here we offer yet another inference example based on PyTorch U-Net. You might find the model implemented in `unet.py`. Please, follow the instructions if you wish to reproduce the U-Net baseline results.

# Steps to use U-Net as a submission

1. Replace `ocelot23algo/Dockerfile` with `ocelot23algo/user/unet_example/Dockerfile`. This is necessary to install PyTorch and other dependencies.
2. Replace `ocelot23algo/requirements.txt` with `ocelot23alg/user/unet_example/requirements.txt`.
3. Reference the new inference code from the program entry-point i.e. `ocelot23algo/process.py`. Simply modify `L8` as follows:

```diff
- from user.inference import Model
+ from user.unet_example.unet import PytorchUnetCellModel as Model
``` 

4. Download the U-Net weights from [here](https://drive.google.com/file/d/1oQ8276P-CU0iPf_oatCYlvKYY7UoEaTm/view?usp=share_link), and put the file inside the directory `ocelot23algo/user/unet_example/checkpoints/` with name `ocelot_unet.pth`.

5. Verify the code works by running a simple test:

```sh
bash test.sh
```

6.  Export the code and submit to GC:

```sh
bash export.sh
```

# References

* U-Net code is borrowed from [here](https://github.com/milesial/Pytorch-UNet).
