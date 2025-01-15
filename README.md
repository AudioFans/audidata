# Audidata

Audidata is a toolkit that allows users to easily load audio datasets. Audidata supports music, audio, and speech datasets. Audidata also provides samplers, tokenizers, and transforms. Users can also create their own datasets, samplers, and transforms based on Audidata.

## 0. Installation

### Method 1.

```bash
pip install audidata
````

### Method 1. Download source code

```bash
git clone https://github.com/AudioFans/audidata
cd audidata
export PYTHONPATH=`pwd`  # Export environment path so that users can import audidata anywhere in the terminal.
```

## 1. Example

Users must download the dataset manually. The datsets should be structured described in dataset files. Here is an example of loading GTZAN dataset.

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
│   ├── transforms
│   │   ├── midi.py
│   │   └── ...
│   └── utils.py
├── examples
│   ├── test_concat_datasets.py
│   └── ...
├── LICENSE
├── README.md
└── pyproject.toml

</pre>

## License

MIT