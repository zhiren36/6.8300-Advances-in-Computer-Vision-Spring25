# Problem-3 Playing with a NeRF

Modified from 6.S980 Homework 2



## Getting Started

**Using Python 3.9 or newer,** create a virtual environment as follows:

```
python3 -m venv venv
source venv/bin/activate
```

Next, install PyTorch using the instructions [here](https://pytorch.org/get-started/locally/). Select pip as the installation method. **If you're on Linux and have a CUDA-capable GPU, select the latest CUDA version.** This will give you a command like this:

```
pip3 install torch torchvision  # cuda 12.4
```

Finally, install this homework's dependencies:

```
pip install -r requirements.txt
```



### Multilayer Perceptron (MLP) Neural Field



#### Positional Encodings

Please fill in several lines of code in  `src/components/positional_encoding.py` to implement a Positional Encoding described in Equation (4) of paper:  [Neural Radiance Fields, Mildenhall et al., 2020](https://arxiv.org/abs/2003.08934). . And attach the filled code in the submission report.

After your implementation, pleas run to test it.

`python -m tests.test_positional_encoding`



If you're interested in learning more about positional encodings, check out [Fourier Features Let Networks Learn
High Frequency Functions in Low Dimensional Domains](https://bmild.github.io/fourfeat/).



Neural fields can be categorized into two categories—explicit and implicit—as follows:

- **Implicit:** Coordinates are mapped to values via neural networks.
- **Explicit:** Coordinates are mapped to values via spatial data structures (grids, octrees, point clouds, etc.).

We will play an implicit one in this problem. It's already implemented in in `src/field/field_mlp.py`.  Read it. 



## Training Neural Radiance Fields (NeRFs)

### NeRF Rendering Procedure

In this part of the assignment, you'll training a neural radiance field (NeRF), mostly as described in [Neural Radiance Fields, Mildenhall et al., 2020](https://arxiv.org/abs/2003.08934). 



Since in problem-2, you have implemented a basic volumetric rendering, so you don't need to implement again. It's mostly done in `src/nerf.py`, just has one line needs to be filled in. Copy the line you added into the final report. 



Please also read the`src/nerf.py` code and answer the question:

The rendering model used in our code is slightly different than the one used in the original nerf paper. **Please indentify at least one difference**. (Not the size of the model, not the inititalization, etc. Hint: look at rendering, and ray sampling!)



### Downloading the Dataset

We will download a low-res `lego` scene of the NeRF-Synthetic dataset [here](https://drive.google.com/file/d/15brXJJZZg16NaeJ9KG3ibjHzD3qNRemY/view?usp=sharing). Place it at `data/nerf_synthetic`.  The folder structure would look like `data/nerf_synthetic/lego`

## Training

The training script is provided. 

Please run

`python -m scripts.train_nerf`

It will produce some visualization in the folder of `outputs/`. It will save the final rendering in `outputs/spin`



Please attach at least two final images inside the `outputs/spin` folder in the final report. If you have enough time and computes, you are encouraged play with different MLP size or training schedules, by modifying the `config/field/mlp.yaml`. It's fine that your final images got blured results.

For a reference of the amount of compute and results, I have been training it for 20 mins on a M1-pro Macbook Pro with four layers of MLP and hidden size of 256, and the results is blured in this case. But the 3D structure can be percieved well. 



### Note on GPUs

This assignment is designed to work on any relatively modern computer. However, NeRF training runs significantly faster on CUDA devices. For example, training the NeRF takes about 1 minute on a 3090 Ti (graphics card) and 10 minutes on an M2 Macbook Pro (CPU).

## 



## Collaboration Policy

You may work with other students and use AI tools (e.g. ChatGPT, GitHub Copilot), but must submit code that you understand fully and have written yourself.

## Submission Policy

For problem-3, you don't need to submit the codes. You just need to submit a report (pdf) that  contains code and answers for questions mentioned in 

1. positional encoding
2. nerf forward pass code. 
3. difference between our implemented nerf and original nerf paper. 
4. Attach at least two images that shows your final rendering. 



Otherwise, you will lose points. Submit your work using Gradescope.

## Bug Bounty

If you are the first to find and report a bug of any kind in this assignment or README, you will be given extra points. Bugs could include, but are not limited to:

- Code bugs
- Incorrect instructions
- Grammatical errors
- Broken links
- Outdated or broken dependencies

To report a bug, post to the bug bounty thread on Piazza. Make sure your name is visible to the instructors if you want to receive credit for your discovery. Rewards for individual bugs will be proportional to their severity, and total rewards per student can be up to 3% of this assignment's total point value
