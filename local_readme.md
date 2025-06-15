## Running staff

In the groot folder, source the groot conda environment, and run the following:
```
python scripts/inference_service.py --server --embodiment_tag new_embodiment --data_config libero
```

In another terminal, source the virtual environment from sim/libero/.venv/bin/activate, and then run the following:
```
python sim/libero/main.py
```
