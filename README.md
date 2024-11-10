# breath-remove

A command-line tool for detecting and removing breath sounds from audio files. This tool leverages the Respiro-en model, introduced in the paper "[Frame-Wise Breath Detection with Self-Training: An Exploration of Enhancing Breath Naturalness in Text-to-Speech](https://arxiv.org/abs/2402.00288)" by Dong Yang,  Yuxuan Wang,  Jiatong Shi,  Xin Yuan,  Lei Xie, and  Xunying Liu. The original Respiro-en model and code can be found [here](https://github.com/ydqmkkx/Breath-Detection).


## Installation

As this package is not yet available on PyPI, install directly from the GitHub repository using:

```bash
pip install git+https://github.com/lukaszliniewicz/breath-remove.git
```

## Usage

The `breath-remove` tool is invoked from the command line:

```bash
breath-removal -i <input_audio_file> -o <output_folder> [options]
```

### Arguments

| Argument        | Short Option | Description                                                                                                           | Default         |
|-----------------|--------------|-----------------------------------------------------------------------------------------------------------------------|-----------------|
| `--input`       | `-i`         | Path to the input audio file.                                                                                       | *required*      |
| `--output`      | `-o`         | Path to the output folder where the processed audio will be saved.                                                  | *required*      |
| `--model`       | `-m`         | Path to the Respiro-en model file. If not provided, the model will be downloaded automatically.                     | *optional*      |
| `--sr`          |              | Sample rate of the input audio.                                                                                      | 22050           |
| `--channels`    |              | Number of channels in the input audio (1 for mono, 2 for stereo).                                                   | 1               |
| `--silence`     |              | Level of silence applied to breath segments. Either 'full' for complete silence, or an integer percentage (1-100). | full           |
| `--plot`        |              | Generate a plot visualizing the waveform and detected breath segments.                                             | False           |
| `--max-minutes` |              | Maximum segment length in minutes for processing. Longer audio will be split into segments for improved performance. | 5.0             |



## How it Works

This tool detects breath sounds within an audio file using the Respiro-en model.  It then applies either full silence or volume reduction to the detected breath segments, effectively removing or minimizing their presence in the final output.  The `--silence` option controls the level of attenuation applied. The tool can also generate plots to visualize the analysis.  Longer audio files are automatically split into shorter segments for processing and then recombined, allowing for handling of files of arbitrary length.

## Acknowledgements

* **Original Model and Paper:**  Dong Yang, Yuxuan Wang, Jiatong Shi, Xin Yuan, Lei Xie, and Xunying Liu for their work on the Respiro-en model and the paper "[Frame-Wise Breath Detection with Self-Training: An Exploration of Enhancing Breath Naturalness in Text-to-Speech](https://arxiv.org/abs/2402.00288)".

