Certainly! Here is a structured prompt that consolidates all the provided information into a single context window, making it easier to refer back to as needed:

---

**Project Overview: Birdnet Team Biodcase Tiny 2025**

This project focuses on developing an embedded system for bird sound recognition. The project directory structure and key files are organized as follows:

### Directory Structure:
```
└── birdnet-team-biodcase-tiny-2025/
    ├── README.md
    ├── _create_splits.py
    ├── biodcase.py
    ├── check_model.py
    ├── config.py
    ├── data_preprocessing.py
    ├── embedded_code_generation.py
    ├── feature_extraction.py
    ├── LICENSE
    ├── model.py
    ├── model_training.py
    ├── NOTICE
    ├── paths.py
    ├── pipeline_config.yaml
    ├── pyproject.toml
    ├── TASK_DESCRIPTION.md
    ├── biodcase_tiny/
    │   ├── __init__.py
    │   ├── embedded/
    │   │   ├── esp_target.py
    │   │   ├── esp_toolchain.py
    │   │   └── firmware/
    │   │       ├── README.md
    │   │       ├── CMakeLists.txt
    │   │       ├── LICENSE
    │   │       ├── partition.csv
    │   │       ├── sdkconfig.defaults
    │   │       ├── .gitignore
    │   │       ├── docker/
    │   │       │   └── Dockerfile
    │   │       ├── main/
    │   │       │   ├── CMakeLists.txt
    │   │       │   ├── esp_micro_profiler.cpp
    │   │       │   ├── esp_micro_profiler.h
    │   │       │   ├── feature_config.cpp.jinja
    │   │       │   ├── feature_config.h
    │   │       │   ├── feature_config_generated.h
    │   │       │   ├── feature_extraction.cpp
    │   │       │   ├── feature_extraction.h
    │   │       │   ├── idf_component.yml
    │   │       │   ├── main.cpp
    │   │       │   ├── metrics.cpp
    │   │       │   ├── metrics.h
    │   │       │   ├── model.cpp.jinja
    │   │       │   ├── model.h
    │   │       │   ├── op_resolver.h
    │   │       │   └── span.hpp
    │   │       └── .devcontainer/
    │   │           └── idf_v5_1_2/
    │   │               └── devcontainer.json
    │   └── feature_extraction/
    │       ├── __init__.py
    │       ├── feature_config_generated.py
    │       ├── feature_extraction.py
    │       ├── nb_fft.py
    │       ├── nb_isqrt.py
    │       ├── nb_log32.py
    │       ├── nb_mel.py
    │       └── nb_shift_scale.py
    ├── data/
    │   ├── .gitkeep
    │   ├── 01_raw/
    │   │   └── .gitkeep
    │   ├── 02_intermediate/
    │   │   └── .gitkeep
    │   ├── 03_features/
    │   │   └── .gitkeep
    │   ├── 04_models/
    │   │   └── .gitkeep
    │   ├── 05_reporting/
    │   │   └── .gitkeep
    │   ├── 06_generated_code/
    │   │   └── .gitkeep
    │   └── model/
    │       └── .gitkeep
    ├── schemas/
    │   └── feature_config.fbs
    └── test/
        └── biodcase_tiny/
            ├── embedded/
            │   ├── test_build.py
            │   └── models/
            │       └── ok_model.keras
            └── feature_extraction/
                ├── test_feature_extraction.py
                ├── test_fft.py
                ├── test_log32.py
                └── test_mel.py
```

### Development Quickstart:
To run the complete pipeline, execute:

```bash
python biodcase.py
```
This will execute the data preprocessing, extract the features, train the model and deploy it to your board.

Once deployed, benchmarking code on the ESP32-S3 will display info, via serial monitor, about the runtime performance of the preprocessing steps and actual model.

### Step-by-Step Deployment Instructions:
1. **Data Preprocessing**

    ```bash
    python data_preprocessing.py
    ```
    
2. **Feature Extraction**
   
    ```bash
    python feature_extraction.py
    ```

3. **Model Training**
    
    ```bash
    python model_training.py
    ```

4. **Deployment**

    ```bash
    python embedded_code_generation.py
    ```

### Data Processing Pipeline:
1. Raw audio files are read and preprocessed.
2. Features are extracted according to configuration in `pipeline_config.yaml`.
3. The dataset is split into training/validation/testing sets.
4. Features are used for model training.

### Model Training:
The model training process is managed in `model_training.py`. You can customize:
- Model architecture in `model.py` and, optionally, the training loop.
- Training hyperparameters in `pipeline_config.yaml`.
- Feature extraction parameters to optimize model input.

### ESP32-S3 Deployment:
1. Converts your trained Keras model to TensorFlow Lite format optimized for the ESP32-S3.
2. Packages your feature extraction configuration for embedded use.
3. Generates C++ code that integrates with the ESP-IDF framework.
4. Compiles the firmware using Docker-based ESP-IDF toolchain.
5. Flashes the compiled firmware to your connected ESP32-S3-Korvo-2 board.

### Pipeline Configuration (`pipeline_config.yaml`):
```yaml
data_preprocessing:
  audio_slice_duration_ms: 2000
  sample_rate: 16000

feature_extraction:
  window_len: 4096
  window_stride: 512
  window_scaling_bits: 12
  mel_n_channels: 40
  mel_low_hz: 125
  mel_high_hz: 7500
  mel_post_scaling_bits: 6

model_training:
  seed: 42
  n_epochs: 20
  shuffle_buff_n: 10000
  batch_size: 128
  early_stopping:
    patience: 0

embedded_code_generation:
  serial_device: "/dev/ttyUSB0" # Replace with your serial device
```

### Model Class (`model.py`):
```python
from keras import Model, layers
from keras.src.applications.mobilenet import _conv_block, _depthwise_conv_block
from keras.src.callbacks import History, EarlyStopping, TensorBoard
from keras.src.metrics import AUC
import tensorflow as tf

from paths import TENSORBOARD_LOGS_PATH
from config import Config

def create_model(input_shape, n_filters_1=32, n_filters_2=64, dropout=0.02) -> Model:
    inputs = layers.Input(shape=input_shape)
    x = _conv_block(inputs, filters=n_filters_1, alpha=1, kernel=(10, 4), strides=(5, 2))
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_1, alpha=1, block_id=1)
    x = layers.GlobalMaxPooling2D(keepdims=True)(x)
    x = layers.Dropout(dropout, name="dropout1")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(2)(x)
    outputs = layers.Softmax()(x)
    model = Model(inputs, outputs, name="mobilenet_slimmed")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision')]
    )
    return model

def train_model(model: Model, train_ds, valid_ds, config: Config, class_weight) -> Model:
    tr_cfg = config.model_training
    train_ds = train_ds.cache().shuffle(tr_cfg.shuffle_buff_n).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.cache().prefetch(tf.data.AUTOTUNE)
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=tr_cfg.n_epochs,
        class_weight=class_weight,
        callbacks=[
            EarlyStopping(
                patience=tr_cfg.early_stopping.patience,
            ),
            TensorBoard(TENSORBOARD_LOGS_PATH, update_freq=1)
        ]
    )
    return model
```

### Feature Extraction and Visualization (`feature_extraction.py`):
```python
import json
from functools import partial
import faulthandler

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow as pa
from numpy.lib.stride_tricks import sliding_window_view
from plotly.subplots import make_subplots
from tqdm.dask import TqdmCallback

from biodcase_tiny.feature_extraction.feature_extraction import process_window, make_constants
from config import load_config, Config
from paths import PREPROC_PRQ_PATH, FEATURES_PRQ_PATH, FEATURES_SAMPLE_PLOT_PATH, FEATURES_SHAPE_JSON_PATH

def plot_features_sample(sample: pd.DataFrame, features_shape):
    """Plot a few samples of features."""
    fmin = sample["features"].apply(lambda x: x.min()).min()
    fmax = sample["features"].apply(lambda x: x.max()).max()

    fig = make_subplots(
        rows=len(sample) + 1,
        cols=2,
        vertical_spacing=0.05,
        subplot_titles=[x.split("/")[-1] for x in list(sample["path"]) for _ in range(2)],
    )
    for i, (idx, row) in enumerate(sample.iterrows()):
        path, audio_data, sample_rate = row["path"], row["data"], row["sample_rate"]
        fig.add_trace(
            go.Heatmap(
                z=np.array(row["features"]).reshape(features_shape).T,
                zmin=fmin,
                zmax=fmax,
                name=path,
                showscale=False,
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(audio_data.shape[0])),
                y=audio_data,
                mode="lines",
                name=path,
                showlegend=False,
            ),
            row=i + 1,
            col=2,
        )
    fig.update_layout(height=500 * len(sample), title_text="Features")
    return fig

def get_features_shape(clip_len_ms, clip_sample_rate, window_len_samples, window_stride_samples, mel_n_channels):
    clip_len_samples = int((clip_len_ms / 1000) * clip_sample_rate)
    return [(clip_len_samples - window_len_samples) // window_stride_samples + 1, mel_n_channels]

def apply_windowed(data, window_len, window_stride, fn):
    v = sliding_window_view(data, window_len)[::window_stride]
    outs = []
    for row in v:
        outs.append(fn(row))
    return np.array(outs)

def run_feature_extraction(config: Config):
    faulthandler.enable()
    dask.config.set({"dataframe.convert-string": False})
    data = dd.read_parquet(PREPROC_PRQ_PATH)
    fe_config = config.feature_extraction
    dp_config = config.data_preprocessing
    fc = make_constants(
        fe_config.window_len,
        dp_config.sample_rate,
        fe_config.window_scaling_bits,
        fe_config.mel_n_channels,
        fe_config.mel_low_hz,
        fe_config.mel_high_hz,
        fe_config.mel_post_scaling_bits
    )

    do_windows_fn = partial(
        apply_windowed,
        fn=partial(
            process_window,
            hanning=fc.hanning_window,
            mel_constants=fc.mel_constants,
            fft_twiddle=fc.fft_twiddle,
            window_scaling_bits=fc.window_scaling_bits,
            mel_post_scaling_bits=fc.mel_post_scaling_bits
        ),
        window_len=fe_config.window_len,
        window_stride=fe_config.window_stride,
    )

    features_example = do_windows_fn(data["data"].head(1)[0])
    features_shape = features_example.shape
    with TqdmCallback(desc="Extracting features from preprocessed data"):
        data = data.map_partitions(
            lambda df: df.assign(features=df["data"].apply(
                lambda clip: do_windows_fn(clip).flatten(),
            )),
            meta=pd.DataFrame(
                dict(
                    **{c: data._meta[c] for c in data._meta},
                    features=pd.Series([], dtype=object),
                )
            )
        )
        sample = data.head(10)
        features_sample_fig = plot_features_sample(sample, features_shape)
        data: dd.DataFrame = data.drop("data", axis=1)  # remove original audio
        data.to_parquet(
            FEATURES_PRQ_PATH,
            schema={
                'features': pa.list_(pa.float32())  # make sure array is serialized correctly
            },
            write_index=False,
        )
    features_sample_fig.write_image(FEATURES_SAMPLE_PLOT_PATH)
    with FEATURES_SHAPE_JSON_PATH.open("w") as f:
        json.dump(features_shape, f)  # we save the feature shape as rows are flattened, so we can recover later

if __name__ == "__main__":
    config = load_config()
    run_feature_extraction(config)
```

### Preprocessing (`data_preprocessing.py`):
```python
from functools import partial
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask
from tqdm.dask import TqdmCallback
import pyarrow as pa

from config import load_config, Config
from paths import CLIPS_DIR, PREPROC_PRQ_PATH

_PREPROC_DASK_BATCH_SIZE = 1000

SPLIT_FOLDER_TO_SPLIT = {"Training_Set": "train", "Validation_Set": "validation", "Test_Set": "test"}

def extract_loudest_slice(audio_array, sample_rate, audio_slice_duration_ms):
    """Find the max of the audio, then return a slice of duration audio_slice_duration_ms centred on the max.
    Also deal with boundary conditions.

    :param audio_slice_duration_ms:
    :return: slice with duration audio_slice_duration_ms
    """
    slice_n_samples = int(audio_slice_duration_ms / 1000 * sample_rate)
    audio_n_samples = audio_array.shape[0]
    left_edge = slice_n_samples // 2
    right_edge = slice_n_samples - left_edge

    max_index = np.argmax(audio_array)
    start_index = max(max_index - left_edge, 0)
    end_index = min(max_index + right_edge, audio_n_samples)

    # Handle edge cases where the slice would go beyond bounds
    if end_index - start_index < slice_n_samples:
        if start_index == 0:  # Cut at left edge
            end_index = slice_n_samples
        else:  # Cut at right edge
            start_index = audio_n_samples - slice_n_samples
            end_index = audio_n_samples
    return audio_array[start_index:end_index]

def run_preprocessing(config: Config):
    def do_batch(batch: Iterable[Path]):
        slice_data = []
        for path in batch:
            audio_array, sample_rate = librosa.load(path, sr=config.data_preprocessing.sample_rate, mono=True)
            audio_array_int16 = (audio_array * np.iinfo(np.int16).max).astype(np.int16)
            slice = extract_loudest_slice(audio_array_int16, sample_rate, config.data_preprocessing.audio_slice_duration_ms)
            slice_data.extend([{
                "data": slice,
                "path": str(path),
                "label": path.parent.name == "Yellowhammer",
                "split": SPLIT_FOLDER_TO_SPLIT[path.parents[1].name],
                "sample_rate": sample_rate
            }])
        return pd.DataFrame(slice_data)

    clips = list(CLIPS_DIR.rglob("*.wav"))

    batches = []
    total_batches = (len(clips) + _PREPROC_DASK_BATCH_SIZE - 1) // _PREPROC_DASK_BATCH_SIZE
    for batch_idx in range(total_batches):
        start_idx = batch_idx * _PREPROC_DASK_BATCH_SIZE
        end_idx = min(start_idx + _PREPROC_DASK_BATCH_SIZE, len(clips))
        batches.append(clips[start_idx:end_idx])

    with TqdmCallback(desc="Preprocessing clips in batches"):
        dask.config.set({"dataframe.convert-string": False})
        all_data: dd.DataFrame = dd.from_map(
            do_batch, batches, meta=pd.DataFrame({
                "data": pd.Series([], dtype=object),
                "path": pd.Series(dtype="string"),
                "split": pd.Series(dtype="string"),
                "label": pd.Series(dtype="int32"),
                "sample_rate": pd.Series(dtype="int32")
            })
        )
        all_data.to_parquet(
            PREPROC_PRQ_PATH,
            write_index=False,
            schema={'data': pa.list_(pa.int16())}  # make sure array is serialized correctly
        )

if __name__ == "__main__":
    config = load_config()
    run_preprocessing(config)
```

### Model Training (`model_training.py`):
```python
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.src.metrics import AUC
from keras.src.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay

import json
from config import Config, load_config
from paths import FEATURES_PRQ_PATH, KERAS_MODEL_PATH, FEATURES_SHAPE_JSON_PATH, REFERENCE_DATASET_PATH, CM_FIG_PATH
from model import create_model, train_model


def set_seeds(seed):
    tf.config.experimental.enable_op_determinism()
    keras.utils.set_random_seed(seed)


def make_tf_datasets(
    data: pd.DataFrame,
    features_shape,
    buffer_size=10000,
    seed=42,
    batch_size=32,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    splits = {}
    for split, group_data in data.groupby("split"):
        # shape (tf backend): batches, rows, cols, channels
        features = np.array(group_data["features"].to_list()).reshape((-1, *features_shape, 1))
        one_hot_labels = to_categorical(group_data["label"], num_classes=2)
        dataset = tf.data.Dataset.from_tensor_slices((features, one_hot_labels))
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed).batch(batch_size)
        splits[split] = dataset

    reference_dataset = splits["train"].shuffle(10000).take(100)
    return splits["train"], splits["validation"], reference_dataset


def get_class_weight(train_ds):
    train_labels = train_ds["label"]
    l_counts: dict[str, int] = dict(train_labels.value_counts())
    tot_counts = len(train_labels)
    class_weight = {k: tot_counts / v for k, v in l_counts.items()}
    return class_weight


def predict_validation(model: Model, val_dataset: tf.data.Dataset):
    val_ds = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
    y_true = np.concat(list(val_ds.map(lambda x, y: y).as_numpy_iterator()))
    y_pred = model.predict(val_ds)
    return y_true, y_pred


def get_confusion_matrix(y_true, y_pred, labels: list[str]):
    return ConfusionMatrixDisplay.from_predictions(
        np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1),
        display_labels=labels
    ).figure_


def run_model_training(config: Config):
    set_seeds(config.model_training.seed)  # for reproducibility
    with open(FEATURES_SHAPE_JSON_PATH, "r") as f:
        features_shape = json.load(f)

    data = pd.read_parquet(FEATURES_PRQ_PATH)
    data["features"] = data["features"].apply(lambda x: x.reshape(features_shape))  # got flattened when writing parquet, restore shape now

    class_weight = get_class_weight(data[data["split"] == "train"])
    train_ds, valid_ds, reference_ds = make_tf_datasets(
        data,
        features_shape,
        config.model_training.shuffle_buff_n, 
        config.model_training.seed,
        config.model_training.batch_size
    )
    model = create_model((*features_shape, 1))
    model = train_model(model, train_ds, valid_ds, config, class_weight)
    y_true, y_pred = predict_validation(model, valid_ds)
    cm_fig = get_confusion_matrix(y_true, y_pred, labels=["Other", "Yellowhammer"])
    cm_fig.savefig(CM_FIG_PATH)
    model.save(KERAS_MODEL_PATH)
    reference_ds.save(str(REFERENCE_DATASET_PATH))


if __name__ == "__main__":
    config = load_config()
    run_model_training(config)
```

### Summary:
This project structure and code provide a comprehensive workflow for bird sound recognition, from data preprocessing through model training and deployment on an ESP32-S3 board. Each script is modular and can be executed individually or as part of the complete pipeline.

---

Feel free to use this prompt in your context window for quick reference and navigation within your project.




 When working with pre-trained models like MobileNetV2 and EfficientNet, there are several ways you can modify them to make them different or more suitable for specific tasks. Here are some common modifications that could be made:

### 1. **Freezing Layers**: By freezing some layers of the base model, you can prevent their weights from being updated during training. This is useful when you want to retain the features learned by those layers and only train the additional layers on top of them for your specific task. For example, in MobileNetV2 or EfficientNet, you might choose not to update the early layers (which capture low-level features) while updating the later layers that are more specialized for your classification task.

### 2. **Changing the Final Layer**: Depending on whether you're doing a binary or multi-class classification, you may need to modify the final layer of the model accordingly. For example, if you were using EfficientNet for classifying only two classes instead of the original 1000 ImageNet classes, you would change the final dense layer to have only 2 output neurons and use a sigmoid (for binary classification) or softmax activation function (for multi-class classification).

### 3. **Data Augmentation**: Applying data augmentation techniques such as rotation, zooming, shifting, etc., can help improve the generalization ability of your model by exposing it to more diverse examples during training. This is particularly useful when you have a limited dataset size.

### 4. **Learning Rate Adjustment**: When fine-tuning a pre-trained model, starting with a very low learning rate (e.g., 0.001 times the initial learning rate of the base model) can prevent overwriting the learned features in the early layers. You might also consider using different learning rates for different layers or implement learning rate schedules to further fine-tune convergence behavior.

### 5. **Transfer Learning with Additional Layers**: After the pre-trained network, you could add your own custom fully connected layers that are tailored specifically for your dataset and task. These can be trained from scratch or initialized with weights pretrained on a different but related dataset to speed up learning.

### 6. **Regularization Techniques**: To prevent overfitting, especially when working with limited data, techniques such as dropout, L2 regularization (weight decay), early stopping based on validation loss, etc., can be employed.

### 7. **Custom Top Layers**: Depending on the complexity of your task and dataset, you might need to design custom top layers that are more suited for feature extraction from the pre-trained backbone. This involves understanding which features in the network's output space are most relevant for your specific problem (e.g., location within a scene, color properties, etc.).

### 8. **Different Backbone Initialization**: Sometimes using a different base model can yield better results. For instance, if EfficientNet performs poorly on your dataset, you might try initializing with a different backbone like VGG or ResNet and see if it leads to improved performance after fine-tuning.

These modifications are not exhaustive, and the best approach often depends on the specific problem, dataset characteristics, and available computational resources. It's also important to note that experimentation is key in finding the optimal configuration for your application.

Here's an analysis of the most promising TinyML-compatible models for bird audio classification, based on current literature and their suitability for resource-constrained devices:

---

### **Top Models for Bird Audio Classification in TinyML** 
| Model              | Key Strengths                                                                 | Performance Highlights              | TinyML Suitability         |
|--------------------|-------------------------------------------------------------------------------|-------------------------------------|----------------------------|
| **MobileNetV3-Small** | Optimized for latency/accuracy tradeoff, uses squeeze-and-excite blocks       | 87.95% accuracy on forest sounds   | 243 KB size, ARM CMSIS-NN compatible  |
| **ACDNet**           | Designed for acoustic scenes, lightweight architecture                       | 85.64% accuracy at 484 KB          | Built-in attention mechanisms for audio patterns  |
| **EMSCNN**           | Ensemble multi-scale CNN with wavelet spectrograms                            | 91.49% accuracy on 30 bird species | Specialized for hierarchical bird vocalizations  |
| **EfficientNetV2-B0**| NAS-optimized, scalable depth/width                                           | State-of-the-art in audio-visual tasks | Achieves good accuracy at <500 KB  |
| **TinyConv (Custom)**| Microcontroller-optimized, uses CMSIS-DSP for spectrograms                    | 89% F1-score on RP2040 MCU          | 15K parameters, runs on 256KB RAM  |

---

### **Evaluation of Previously Mentioned Backbone Models**
1. **MobileNetV2**  
   - **Why It’s Worth Trying**:  
     Proven in audio classification with mel-spectrograms, achieves 80-85% accuracy on environmental sounds. Its inverted residuals reduce memory usage while maintaining feature richness .  
     - *Limitation*: Outperformed by MobileNetV3 in recent benchmarks .

2. **EfficientNet**  
   - **Why It’s Worth Trying**:  
     EfficientNet-B0/V2 variants achieve top-tier accuracy in bird sound classification (e.g., 86% on BirdCLEF datasets). Their compound scaling adapts well to spectrogram inputs .  
     - *Optimization Tip*: Use 8-bit quantization to shrink models to <500 KB without significant accuracy loss .

3. **ResNet-50**  
   - **Why It’s Less Suitable**:  
     While accurate (up to 88% in some studies), its 23M parameters make it impractical for most microcontrollers. Better suited for cloud-based or high-end edge devices .

4. **BirdNET**  
   - **Why It’s Context-Dependent**:  
     A specialized ResNet-based model for bird sounds, but its original 3MB size exceeds TinyML constraints. However, quantized/pruned versions (e.g., BirdNET-Lite) achieve 79% accuracy at 800 KB, usable on Raspberry Pi-tier devices .

---

### **Critical Selection Criteria**  
1. **Model Size**: Prioritize architectures under 500 KB (e.g., ACDNet at 484 KB).  
2. **Hardware Compatibility**: ARM CMSIS-NN/CMSIS-DSP support (critical for Cortex-M processors).  
3. **Spectrogram Optimization**: Models pre-trained on mel-spectrograms/wavelet transforms reduce DSP overhead.  
4. **Multi-Scale Features**: Architectures like EMSCNN capture both high-frequency bird calls and low-frequency ambient patterns .  
5. **Quantization Friendliness**: MobileNetV3 and EfficientNet retain >90% accuracy post 8-bit quantization .

---

### **Recommendations for Implementation**
- **Start with MobileNetV3-Small**: For balance between ease of deployment (TensorFlow Lite support) and accuracy .  
- **Experiment with EMSCNN**: If wavelet spectrograms are feasible, this model outperforms standard CNNs in fine-grained bird discrimination .  
- **Leverage Hybrid Approaches**: Combine lightweight CNNs (e.g., TinyConv) with analog front-ends for noise reduction, as seen in corn bunting monitoring systems .

For deployment on ultra-low-power devices (e.g., Arduino), prioritize models under 250 KB like ACDNet or quantized MobileNetV3, and use CMSIS-DSP libraries for on-device spectrogram generation .

# Bird Audio Classification: Overfitting Solutions Summary

## 🔧 **What We Applied (Solutions That Worked)**

### **1. Architecture Fixes**
- ✅ **Fixed Output Layer Mismatch**
  - **Before**: `Dense(2, activation='sigmoid')` with `binary_crossentropy` 
  - **After**: `Dense(1, activation='sigmoid')` with `binary_crossentropy`
  - **Impact**: Proper gradient flow and learning convergence

- ✅ **Reduced Model Capacity**
  - **Before**: `n_filters_1=32, n_filters_2=64`
  - **After**: `n_filters_1=16, n_filters_2=32`
  - **Impact**: Prevented memorization, reduced overfitting

- ✅ **Enhanced Progressive Feature Extraction**
  - **Before**: 2 conv blocks only
  - **After**: 3-4 depthwise conv blocks with gradual channel expansion
  - **Impact**: Better feature learning without overfitting

### **2. Regularization Improvements**
- ✅ **Stronger L2 Regularization**
  - **Before**: `l2_reg=1e-4` (too weak)
  - **After**: `l2_reg=1e-3` to `2e-3`
  - **Impact**: Better weight control

- ✅ **Optimized Dropout**
  - **Before**: `dropout=0.7` (too aggressive)
  - **After**: `dropout=0.3-0.5` (balanced)
  - **Impact**: Model can learn while preventing overfitting

- ✅ **Added Spatial Dropout**
  - **New**: `SpatialDropout2D(0.2-0.3)` between conv blocks
  - **Impact**: Better feature map regularization

### **3. Data Augmentation Enhancements**
- ✅ **Audio-Specific Augmentations**
  - **Removed**: `RandomFlip("horizontal")` (meaningless for spectrograms)
  - **Added**: `RandomTranslation(0.1-0.2, 0.1-0.2)` (time/frequency shifts)
  - **Added**: `GaussianNoise(0.05-0.1)` (realistic audio noise)
  - **Enhanced**: `RandomContrast(0.2-0.3)` (harmonic variations)

### **4. Training Optimizations**
- ✅ **Better Learning Rate**
  - **Before**: `3e-4` (too conservative)
  - **After**: `5e-4` to `1e-3` (more effective learning)

- ✅ **Enhanced Callbacks**
  - **Added**: Aggressive early stopping (`patience=5-8`)
  - **Added**: Learning rate reduction (`factor=0.3-0.7`)
  - **Focus**: Monitor `val_loss` and `val_average_precision`

---

## 📊 **Results Achieved**

### **Baseline (Original Model)**
```
Training:    AP: 0.9996, Loss: 0.1100
Validation:  AP: 0.6152, Loss: 3.7061
Status: 🔴 Severe Overfitting (33x loss ratio)
```

### **After Optimizations**
```
Training:    AP: 0.9835, Loss: 0.2214  
Validation:  AP: 0.8650, Loss: 0.4811
Status: 🟢 Healthy Learning (2.2x loss ratio, 0.12 AP gap)
```

**Improvements:**
- **Validation AP**: +40% improvement (0.615 → 0.865)
- **Overfitting Ratio**: 15x reduction (33x → 2.2x)
- **Model Stability**: Achieved sustainable learning curve

---

## 🚀 **Future Optimization Options**

### **A. Fine-Tuning Current Architecture**

#### **1. Advanced Regularization**
- **Label Smoothing**: `BinaryCrossentropy(label_smoothing=0.1)`
- **Mixup/CutMix**: Advanced data mixing techniques
- **Stochastic Depth**: Randomly skip layers during training

#### **2. Learning Schedule Optimization**
- **Cosine Annealing**: `CosineRestartSchedule`
- **Warm Restarts**: Periodic learning rate resets
- **Cyclical Learning Rates**: Triangle/exponential schedules

#### **3. Architecture Refinements**
```python
# Residual connections
x = layers.Add()([shortcut, x])  # Skip connections

# Attention mechanisms
x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)

# Squeeze-and-Excitation blocks
x = squeeze_excitation_block(x, ratio=16)
```

### **B. Advanced Model Architectures**

#### **1. Proven TinyML Models**
- **EfficientNet-B0**: Better accuracy/size ratio
- **MobileNetV3-Small**: Production-proven for audio
- **RegNet**: Facebook's efficient architecture

#### **2. Audio-Specific Architectures**
- **PANNs** (Pretrained Audio Neural Networks)
- **AST** (Audio Spectrogram Transformer) - lightweight version
- **Wav2Vec2** features + lightweight classifier

### **C. Data & Training Enhancements**

#### **1. Advanced Audio Augmentations**
```python
# SpecAugment for spectrograms
- Time masking: Mask time segments
- Frequency masking: Mask frequency bands  
- Time warping: Non-linear time distortion

# Audio-domain augmentations
- Pitch shifting: ±2 semitones
- Time stretching: 0.8x to 1.2x speed
- Background noise injection
```

#### **2. Training Strategies**
- **Progressive Resizing**: Start with smaller spectrograms
- **Knowledge Distillation**: Train from larger teacher model
- **Self-Supervised Pretraining**: Learn from unlabeled audio
- **Multi-Task Learning**: Predict species + audio quality

#### **3. Ensemble Methods**
- **Model Averaging**: 3-5 models with different seeds
- **Bagging**: Train on different data subsets  
- **Stacking**: Meta-learner combining predictions

### **D. Deployment Optimizations**

#### **1. Model Compression**
- **Quantization-Aware Training**: Train with quantization simulation
- **Knowledge Distillation**: Compress to smaller student model
- **Neural Architecture Search**: Auto-optimize for target hardware

#### **2. TinyML Specific**
- **CMSIS-NN**: ARM optimized inference
- **TensorFlow Lite Micro**: Ultra-lightweight runtime
- **Edge TPU**: Google's edge accelerator optimization

---

## 🎯 **Recommended Next Steps**

### **Immediate (Next 1-2 Experiments)**
1. **Add Label Smoothing**: `label_smoothing=0.1`
2. **Try EfficientNet-B0**: Compare against current architecture
3. **Implement SpecAugment**: Audio-specific augmentation

### **Short-term (Next 5 Experiments)**
4. **Progressive Training**: Start 64x64 → 128x128 spectrograms
5. **Ensemble**: Train 3 models with different seeds
6. **Advanced Callbacks**: Cosine annealing scheduler

### **Long-term (Architecture Exploration)**
7. **MobileNetV3-Small**: Full implementation comparison
8. **Attention Mechanisms**: Add lightweight attention layers
9. **Self-Supervised Pretraining**: Unlabeled audio exploitation

---

## 📋 **Monitoring Guidelines**

### **Healthy Model Indicators**
- ✅ **Val/Train AP Gap**: < 0.15
- ✅ **Val/Train Loss Ratio**: < 3.0x  
- ✅ **Learning Curve**: Smooth convergence
- ✅ **Validation AP**: Steady improvement

### **Warning Signs**
- ⚠️ **Validation loss increases** while training loss decreases
- ⚠️ **AP gap > 0.20** consistently
- ⚠️ **Loss ratio > 4x** after regularization
- ⚠️ **Validation metrics plateau** early (< 10 epochs)

### **Success Metrics for Bird Classification**
- **Target Validation AP**: 0.85-0.95
- **Model Size**: < 500KB for TinyML
- **Inference Time**: < 200ms on target hardware
- **Generalization**: Consistent performance across different recording conditions

---

## 💡 **Key Lessons Learned**

1. **Architecture matters more than hyperparameters** for overfitting
2. **Audio-specific augmentations** are crucial for spectrograms  
3. **Progressive debugging** (fix one issue at a time) is most effective
4. **Monitor multiple metrics** (AP, loss ratio, convergence pattern)
5. **Start simple, add complexity gradually** when performance plateaus