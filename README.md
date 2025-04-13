# BioDCASE-Tiny 2025

This repository contains the development framework for the BioDCASE-Tiny 2025 competition, focusing on TinyML implementation for bird species recognition on the ESP32-S3-Korvo development board.

## Background

BioDCASE-Tiny is a competition for developing efficient machine learning models for bird audio recognition that can run on resource-constrained embedded devices. The project uses the ESP32-S3-Korvo development board, which offers audio processing capabilities in a small form factor suitable for field deployment.

## Setup and Installation

### Prerequisites

1. Python 3.8+ with pip
2. [Docker](https://www.docker.com/get-started) for ESP-IDF environment
3. [FlatBuffers compiler](https://google.github.io/flatbuffers/flatbuffers_guide_building.html)
4. USB cable and ESP32-S3-Korvo development board

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/YourOrganization/BioDCASE-Tiny-2025.git
cd BioDCASE-Tiny-2025
```

2. Install Python dependencies:
```bash
pip install -e .
```

## Usage
- Modify model.py with your architecture (make sure to compile with optimizer and loss)
- Modify pipeline_config.yaml parameters of feature extraction
- run main.py

## Dataset

The BioDCASE-Tiny 2025 competition uses a specialized dataset of Yellowhammer bird vocalizations. Key features include:

- 2+ hours of audio recordings
- Songs from multiple individuals recorded at various distances (6.5m to 200m) 
- Recordings in different environments (forest and grassland)
- Includes negative samples (other bird species and background noise)
- Dataset is split into training, validation, and evaluation sets

The training set contains recordings from 8 individuals, while the validation set contains recordings from 2 individuals. An additional 2 individuals are reserved for the final evaluation.

### Dataset Structure

The dataset is organized as follows:

```
Development_Set/
├── Training_Set/
│   ├── Yellowhammer/
│   │   └── *.wav
│   └── Negatives/
│       └── *.wav
└── Validation_Set/
    ├── Yellowhammer/
    │   └── *.wav
    └── Negatives/
        ├── *.wav
        └── val_negative_species.csv
```

- **Yellowhammer/** - Contains target species vocalizations with filenames following format `YH_SongID_Location_Distance.wav`
  - SongID: 3-digit identifier for each song
  - Location: "forest", "grassland", or "speaker" for original recordings
  - Distance: A (original) through H (farthest at ~200m)

- **Negatives/** - Contains negative samples with filenames following format `Type_ID.wav`
  - Type: "Background" for noise or "bird" for non-target species vocalizations
  - ID: File identifier

### Download

Download the dataset from: [Dataset Link - TBD]()

## Development

```bash
# while in root folder
flatc --python -o biodcase_tiny/feature_extraction/ schemas/feature_config.fbs --gen-onefile
flatc --cpp -o biodcase_tiny/embedded/firmware/main/ schemas/feature_config.fbs
```

### Data Processing Pipeline

The data processing pipeline follows these steps:
1. Raw audio files are read and preprocessed
2. Features are extracted according to configuration in `pipeline_config.yaml`
3. The dataset is split into training/validation/testing sets
4. Features are used for model training

### Model Training

The model training process is managed in `model_training.py`. You can customize:
- Model architecture in `model.py`
- Training hyperparameters in `pipeline_config.yaml`
- Feature extraction parameters to optimize model input

### ESP32-S3 Deployment

To deploy your model to the ESP32-S3-Korvo-2 board, you'll use the built-in deployment tools that handle model conversion, code generation, and flashing. The deployment process:

1. Converts your trained Keras model to TensorFlow Lite format optimized for the ESP32-S3
2. Packages your feature extraction configuration for embedded use
3. Generates C++ code that integrates with the ESP-IDF framework
4. Compiles the firmware using Docker-based ESP-IDF toolchain
5. Flashes the compiled firmware to your connected ESP32-S3-Korvo-2 board

#### Step-by-Step Deployment Instructions

1. First, train your model by running:
   ```bash
   python main.py
   ```

2. Once training is complete, deploy to the ESP32-S3-Korvo-2 using:
   ```bash
   python deploy_to_esp.py --model model/your_model.h5 --config pipeline_config.yaml --port /dev/ttyACM0
   ```
   Replace `/dev/ttyACM0` with your board's serial port (on Windows, this would typically be `COM3` or similar).

3. To monitor the board's output after deployment:
   ```bash
   python monitor_esp.py --port /dev/ttyACM0
   ```

The following code snippet shows how the deployment scripts work internally:

```python
from biodcase_tiny.embedded.esp_target import ESPTarget
from biodcase_tiny.embedded.esp_toolchain import ESP_IDF_v5_2

# Setup the toolchain with your board's port
toolchain = ESP_IDF_v5_2("/dev/ttyACM0")  # Change to your board's port

# Create target with your model and feature configuration
target = ESPTarget(model, feature_config, reference_dataset)
target.process_target_templates(output_directory)

# Compile and flash
toolchain.compile(src_path=output_directory)
toolchain.flash(src_path=output_directory)

# Monitor output
toolchain.monitor(src_path=output_directory)
```

Once deployed, the model will run independently on the ESP32-S3, processing audio input from the onboard microphones in real-time and outputting bird species classification results via the serial monitor.

## ESP32-S3-Korvo-2 Development Board

The [ESP32-S3-Korvo-2](https://www.digikey.de/de/products/detail/espressif-systems/ESP32-S3-KORVO-2/15822448) board features:
- ESP32-S3 dual-core processor
- Built-in microphone array
- Audio codec for high-quality audio processing
- Wi-Fi and Bluetooth connectivity
- USB-C connection for programming and debugging

## Code Structure

### Key Entry Points

- `main.py` - Main execution pipeline
- `model.py` - Define your model architecture
- `feature_extraction.py` - Audio feature extraction implementations
- `embedded_code_generation.py` - ESP32 code generation utilities
- `biodcase_tiny/embedded/esp_target.py` - ESP target definition and configuration
- `biodcase_tiny/embedded/firmware/main` - Firmware source code

### Benchmarking

The codebase includes performance benchmarking tools that measure:
- Feature extraction time
- Model inference time
- Memory usage on the target device

## Development Tips

1. **Feature Extraction Parameters**: Carefully tune the feature extraction parameters in `pipeline_config.yaml` for your specific audio dataset.

2. **Model Size**: Keep your model compact. The ESP32-S3 has limited memory, so optimize your architecture accordingly.

3. **Profiling**: Use the profiling tools to identify bottlenecks in your implementation.

4. **Memory Management**: Be mindful of memory allocation on the ESP32. Monitor the allocations reported by the firmware.

5. **Docker Environment**: The toolchain uses Docker to provide a consistent ESP-IDF environment, making it easier to build on any host system.

## Evaluation Metrics

The BioDCASE-Tiny competition evaluates models based on multiple criteria:

### Classification Performance
- **Accuracy**: Overall percentage of correctly identified bird species
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of classification performance per class

### Resource Efficiency
- **Model Size**: Total parameter count and model file size (KB)
- **Inference Time**: Average time required for single audio classification (ms)
- **Peak Memory Usage**: Maximum RAM usage during inference (KB)
- **Energy Efficiency**: Power consumption during continuous operation

### Combined Score
The final ranking combines classification performance with resource efficiency metrics to reward both accurate and efficient implementations. Optimal solutions balance high classification accuracy with minimal resource usage suitable for long-term field deployment.

## License

This project is licensed under the Apache License 2.0 - see the license headers in individual files for details.

## Funding

This project is supported by Jake Holshuh (Cornell class of ´69) and The Arthur Vining Davis Foundations.
Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Education and Research through the project “BirdNET+” (FKZ 01|S22072).
The German Federal Ministry for the Environment, Nature Conservation and Nuclear Safety contributes through the “DeepBirdDetect” project (FKZ 67KI31040E).
In addition, the Deutsche Bundesstiftung Umwelt supports BirdNET through the project “RangerSound” (project 39263/01).

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Logos of all partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)