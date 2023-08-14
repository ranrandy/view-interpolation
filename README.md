# View Synthesis using Stereo Vision
|    |   |
| ------------ | ------------ |
| <img src="outputs/Pipes.gif"/> | <img src="outputs/Classroom1.gif"/> |
| <img src="outputs/Motorcycle.gif"/> | <img src="outputs/Playroom.gif"/> |

## Usage
1. Download and prerprocess the dataset.
    ```
    cd dataset_middlebury
    python dataset_maker.py
    ```

2. Generate sample outputs. 
    ```
    python main.py
    ```
    Modify the value of `gif` in `main.py` to generate dynamic/static outputs.

## References
Dataset: [Middlebury 2014 Stereo](https://vision.middlebury.edu/stereo/data/scenes2014/)

Literature: [View Synthesis Using Stereo Vision](https://www.cs.middlebury.edu/~schar/papers/thesis-lncs.pdf)