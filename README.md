# SpAudSyn (Spatial Audio Synthesizer): a library for spatial sound scenes synthesis

# Introduction
SpAudSyn is a Python library for synthesizing multichannel audio mixtures from single-channel sources.

Originally developed for [*dcase2026_task4_baseline*](https://github.com/nttcslab/dcase2026_task4_baseline) of [DCASE 2026 Task 4](https://dcase.community/challenge2026/task-spatial-semantic-segmentation-of-sound-scenes), SpAudSyn is flexible and adaptable to tasks such as sound event detection and localization.
It supports configurable outputs (e.g., labels, dry sources, metadata, RIRs) and is optimized for on-the-fly mixture generation.
In addition, each mixture can be fully parameterized and saved as a JSON file, enabling exact reconstruction and reproducible experiments.

# Installation
Install the requirements
```
pip install -r requirements.txt
```
Place the library folder into your project and import `SpAudSyn` from `src/spatial_audio_synthesizer.py`.

# Data Organization
SpAudSyn requires input data organized into four directories:

- `foreground_dir`: Contains foreground single-channel source audio files (target signals).  
- `interference_dir`: *[optional]* Contains single-channel non-target or interfering sources.  
- `background_dir`: *[optional]* Contains multichannel background noise recordings.  
- `sofa_dir`: Contains room impulse responses (RIRs) stored as SOFA files.  

To synthesize a mixture, SpAudSyn loads single-channel sources from `foreground_dir` (and optionally from `interference_dir`), convolves with RIRs from `sofa_dir`, and adds multichannel background noise from `background_dir`.

The `foreground_dir` is organized as follows:

```
foreground_dir
|-- Class_label_1
|   |-- source_1.wav
|   |-- object
|   |   |-- object_source1.wav
|   |   |-- object_source2.wav
|   |   `-- ...
|   `-- ...
|-- Class_label_2
|   |-- source_1.wav
|   `-- source_2.wav
`-- ...
```
The `foreground_dir` contains multiple subdirectories, where each subdirectory corresponds to a `Class_label`.  
All `.wav` files belonging to a given class can be placed anywhere within that class’s directory, including nested subdirectories.  

The `interference_dir` has a structure similar to `foreground_dir`. Noise recordings and RIR SOFA files can be placed anywhere within `background_dir` and `sofa_dir`, respectively.

# Example
An example notebook demonstrating how to use SpAudSyn is available at: `example/example.ipynb`.

This example uses data from [*dcase2026_task4_baseline*](https://github.com/nttcslab/dcase2026_task4_baseline). To run the example, place the data folder into the SpAudSyn repository as follows
```
ln -s path/to/dcase2026_task4_baseline/data path/to/SpAudSyn
```

# License 
This project is licensed under the terms described in [LICENSE.pdf](LICENSE.pdf).

# Citation
If you use this library and/or the dataset, please cite the following paper:
```
@article{yasuda2026dcase,
  title={Description and Discussion on DCASE 2026 Challenge task 4: Spatial Semantic Segmentation of Sound Scenes},
  author={Yasuda, Masahiro and Nguyen, Binh Thien and Harada, Noboru and Serizel, Romain and Mishra, Mayank and Delcroix, Marc and Carlos, Hernandez-Olivan and Araki, Shoko and Takeuchi, Daiki and Nakatani, Tomohiro and Ono, Nobutaka},
  journal={arXiv preprint arXiv:2604.00776},
  year={2026},
  url={https://arxiv.org/pdf/2604.00776}
}
```
