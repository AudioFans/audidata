# Audidata

Audidata is a toolkit that allows users to load audio datasets in an easy way.

## 0. Installation

### Method 1. Install via pip.

```python
pip install git+https://github.com/AudioFans/audidata.git@main
```

### Method 2. Download source code (Suggested for developers)

Download source code allows users to edit and create new features for audidata easily. 

```bash
git clone https://github.com/AudioFans/audidata
```

```bash
cd audidata
export PYTHONPATH=`pwd`  # Export environment path so that users can import audidata anywhere in the terminal.
```

## Example

Users must download the dataset manually. We provided how the datasets should be structured in each dataset file. Here is an example of loading GTZAN dataset.

```python
from audidata.datasets import GTZAN

root = "/datasets/gtzan"
dataset = GTZAN(root=root, split="train", test_fold=0, sr=16000)
print(dataset[0])
```

For more examples please see [./audidata/examples](https://github.com/AudioFans/audidata/examples)