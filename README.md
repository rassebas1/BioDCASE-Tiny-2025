## Usage
- Modify model.py with your architecture (make sure to compile with optimizer and loss)
- Modify pipeline_config.yaml parameters of feature extraction
- run main.py

## Development

```bash
# while in root folder
flatc --python -o biodcase_tiny/feature_extraction/ schemas/feature_config.fbs --gen-onefile
flatc --cpp -o biodcase_tiny/embedded/firmware/main/ schemas/feature_config.fbs
```