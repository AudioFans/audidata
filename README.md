# Audidata

Audidata is a toolkit that allows users to easily load audio datasets in less than 5 minutes. Audidata supports music, audio, and speech datasets. Audidata also provides samplers, tokenizers, and transforms. Users can also create their own datasets inside audidata.

## 0. Installation

python >= 3.9 is required.

### Method 1. Download source code (Recomended for developers)

Download source code allows users to edit and create new features for audidata easily. 

```bash
git clone https://github.com/AudioFans/audidata
```

Set environment (for each terminal).

```bash
cd audidata
export PYTHONPATH=`pwd`  # Export environment path so that users can import audidata anywhere in the terminal.
```

### Method 2. Install via pip (Developing)

```python
pip install git+https://github.com/AudioFans/audidata.git@main
```

## 1. Example

Users must download the dataset manually. We provided how the datasets should be structured in each dataset file. Here is an example of loading GTZAN dataset.

```python
from audidata.datasets import GTZAN

root = "/datasets/gtzan"
dataset = GTZAN(root=root, split="train", test_fold=0, sr=16000)
print(dataset[0])
```

Output:

<pre>
{'audio_path': '/datasets/gtzan/genres/blues/blues.00010.au', 
'audio': array([[ 0.11234417,  0.13617763,  0.10609552, ..., -0.06634186, -0.07007345, -0.07359146]], dtype=float32), 
'target': array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32), 
'label': 'blues'}
</pre>

For more examples please see [audidata/examples](https://github.com/AudioFans/audidata/tree/main/examples). For example, users can run the following example script to concatenate multiple datasets:

```python
python examples/test_concat_datasets.py
```

Output:

<pre>
0 dict_keys(['dataset_name', 'audio_path', 'bass', 'drums', 'other', 'vocals', 'accompaniment', 'mixture'])
1 dict_keys(['dataset_name', 'audio_path', 'audio', 'target', 'label'])
2 dict_keys(['dataset_name', 'audio_path', 'bass', 'drums', 'other', 'vocals', 'accompaniment', 'mixture'])
3 dict_keys(['dataset_name', 'audio_path', 'audio', 'target', 'label'])
</pre>

## Repo structure
<pre>
audidata
├── audidata
│   ├── collate
│   │   ├── base.py
│   │   └── ...
│   ├── datasets
│   │   ├── gtzan.py
│   │   └── ...
│   ├── io
│   │   ├── audio.py
│   │   └── ...
│   ├── samplers
│   │   ├── multi_datasets.py
│   │   └── ...
│   ├── tokenizers
│   │   ├── base.py
│   │   └── ...
│   └── transforms
│       ├── midi.py
│       └── ...
├── examples
│   ├── test_concat_datasets.py
│   └── ...
├── LICENSE
├── README.md
└── pyproject.toml

</pre>