# Stage 7A — Comparator Reproducibility Audit

**SeisBench:** 0.12.3  
**New valid comparators:** 2 → `['PhaseNet-ETHZ', 'PhaseNet-SCEDC']`  
**Proceed GPU batch:** `True`

## Per-model registry

### PhaseNet-STEAD

- official_source: `seisbench.models.PhaseNet.from_pretrained('stead')`
- weight_name: `stead`
- training_dataset: `STEAD`
- possible_confirm_leakage: `False`
- sampling_rate: `100`
- component_order: `ZNE`
- window_length: `SeisBench PhaseNet annotate default`
- preprocessing: `SeisBench annotate + project UTC remap ENZ->ZNE / PSN`
- license: `SeisBench model zoo / original dataset licenses`
- valid_comparator: `True`
- exclusion_reason: `None`
- weight_sha256: `761c46496139e9917a7480ff56f660914124c806511ea794f0005518358b09ec`

### PhaseNet-ETHZ

- official_source: `seisbench.models.PhaseNet.from_pretrained('ethz')`
- weight_name: `ethz`
- training_dataset: `ETHZ`
- possible_confirm_leakage: `False`
- sampling_rate: `100`
- component_order: `ZNE`
- window_length: `SeisBench PhaseNet annotate default`
- preprocessing: `SeisBench annotate + project UTC remap ENZ->ZNE / PSN`
- license: `SeisBench model zoo / original dataset licenses`
- valid_comparator: `True`
- exclusion_reason: `None`
- weight_sha256: `89cb5ff3272ea91df7d6a3c347fdd067f5a6c2aab91aa884c448a93a482cd848`

### PhaseNet-SCEDC

- official_source: `seisbench.models.PhaseNet.from_pretrained('scedc')`
- weight_name: `scedc`
- training_dataset: `SCEDC`
- possible_confirm_leakage: `False`
- sampling_rate: `100`
- component_order: `ZNE`
- window_length: `SeisBench PhaseNet annotate default`
- preprocessing: `SeisBench annotate + project UTC remap ENZ->ZNE / PSN`
- license: `SeisBench model zoo / original dataset licenses`
- valid_comparator: `True`
- exclusion_reason: `None`
- weight_sha256: `c71b8857cde02cd4d0ab6e3d5e429b16ae82cfd9e39928bef9ec229a4a64502c`

### EQTransformer-STEAD

- official_source: `seisbench.models.EQTransformer.from_pretrained('stead')`
- weight_name: `stead`
- training_dataset: `None`
- possible_confirm_leakage: `None`
- sampling_rate: `None`
- component_order: `None`
- window_length: `None`
- preprocessing: `None`
- license: `None`
- valid_comparator: `False`
- exclusion_reason: `official_weight_download_failed_SSLError_network; local cache empty`
- weight_sha256: `None`

### EQTransformer-ETHZ

- official_source: `seisbench.models.EQTransformer.from_pretrained('ethz')`
- weight_name: `None`
- training_dataset: `None`
- possible_confirm_leakage: `None`
- sampling_rate: `None`
- component_order: `None`
- window_length: `None`
- preprocessing: `None`
- license: `None`
- valid_comparator: `False`
- exclusion_reason: `official_weight_download_failed_SSLError_network; local cache empty`
- weight_sha256: `None`

### PhaseNet-instance

- official_source: `None`
- weight_name: `None`
- training_dataset: `None`
- possible_confirm_leakage: `True`
- sampling_rate: `None`
- component_order: `None`
- window_length: `None`
- preprocessing: `None`
- license: `None`
- valid_comparator: `False`
- exclusion_reason: `INSTANCE-trained weights may include confirm events; leakage diagnostic only`
- weight_sha256: `None`

### EQTransformer-instance

- official_source: `None`
- weight_name: `None`
- training_dataset: `None`
- possible_confirm_leakage: `True`
- sampling_rate: `None`
- component_order: `None`
- window_length: `None`
- preprocessing: `None`
- license: `None`
- valid_comparator: `False`
- exclusion_reason: `INSTANCE-trained weights may include confirm events; leakage diagnostic only; weight unavailable locally`
- weight_sha256: `None`

### LFTNet

- official_source: `None`
- weight_name: `None`
- training_dataset: `None`
- possible_confirm_leakage: `None`
- sampling_rate: `None`
- component_order: `None`
- window_length: `None`
- preprocessing: `None`
- license: `None`
- valid_comparator: `False`
- exclusion_reason: `no local official package/checkpoint; incomplete reproducibility gates`
- weight_sha256: `None`

### PhaseNO

- official_source: `None`
- weight_name: `None`
- training_dataset: `None`
- possible_confirm_leakage: `None`
- sampling_rate: `None`
- component_order: `None`
- window_length: `None`
- preprocessing: `None`
- license: `None`
- valid_comparator: `False`
- exclusion_reason: `multi-station input; incompatible with single-station Stage-6 protocol`
- weight_sha256: `None`

### SegPhase

- official_source: `None`
- weight_name: `None`
- training_dataset: `None`
- possible_confirm_leakage: `None`
- sampling_rate: `None`
- component_order: `None`
- window_length: `None`
- preprocessing: `None`
- license: `None`
- valid_comparator: `False`
- exclusion_reason: `region/sampling/training protocol incompatible; no fully compatible official checkpoint verified`
- weight_sha256: `None`

### GreenPhase

- official_source: `None`
- weight_name: `None`
- training_dataset: `None`
- possible_confirm_leakage: `None`
- sampling_rate: `None`
- component_order: `None`
- window_length: `None`
- preprocessing: `None`
- license: `None`
- valid_comparator: `False`
- exclusion_reason: `no compatible official public implementation verified in this environment`
- weight_sha256: `None`

### PhaseNet-DAS

- official_source: `None`
- weight_name: `None`
- training_dataset: `None`
- possible_confirm_leakage: `None`
- sampling_rate: `None`
- component_order: `None`
- window_length: `None`
- preprocessing: `None`
- license: `None`
- valid_comparator: `False`
- exclusion_reason: `DAS modality incompatible`
- weight_sha256: `None`

### PhaseNet+

- official_source: `None`
- weight_name: `None`
- training_dataset: `None`
- possible_confirm_leakage: `None`
- sampling_rate: `None`
- component_order: `None`
- window_length: `None`
- preprocessing: `None`
- license: `None`
- valid_comparator: `False`
- exclusion_reason: `multi-task objectives; not a dedicated picking comparator under our protocol`
- weight_sha256: `None`

