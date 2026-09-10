# Model

Qwen2.5-Omni integration and KV cache quantization for OmniKVQuant and TurboQuant.

- [modeling_qwen2_5_omni.py](modeling_qwen2_5_omni.py): Loads the model, processor, and calibrated V rotations; installs the quantized attention implementation and exposes `OmniKVQuantModel` for generation.
- [__init__.py](__init__.py): Exports `OmniKVQuantModel` from the `model` package.
- [codec/kv_cache.py](codec/kv_cache.py): Selects the method, applies normalization and rotations, and manages cache updates. OmniKVQuant uses modality-local temporal K groups and calibrated V rotations; TurboQuant uses shared Hadamard rotations for K and V.
- [codec/packed_cache.py](codec/packed_cache.py): Stores packed 2-bit indices, norms, and token positions. Manages local K groups and their MXFP4 minimum/span metadata, buffer allocation, and reconstruction.
- [codec/attention.py](codec/attention.py): Connects Qwen attention to the quantized cache and dispatches decode to the CUDA kernel or CPU reference implementation.
- [codec/triton_packed_attention.py](codec/triton_packed_attention.py): Computes CUDA decode attention directly from packed K/V, including tile reconstruction, masking, and softmax reduction.
- [codec/triton_store.py](codec/triton_store.py): Implements CUDA quantization and bit packing for the shared Lloyd-Max codec and local K groups.
- [codec/turboquant.py](codec/turboquant.py): Provides normalized Hadamard matrices and the four-level Gaussian Lloyd-Max codebook used by both methods.
- [codec/modality.py](codec/modality.py): Labels input tokens by modality so the cache can group tokens and select the corresponding rotations.
- [codec/__init__.py](codec/__init__.py): Marks `codec` as a Python package.
