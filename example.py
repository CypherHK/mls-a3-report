from part4.design_methodology import HardwareAwareDesignMethodology

# Initialize
hdm = HardwareAwareDesignMethodology()
hdm._load_performance_data()

# Get recommendation
recommendation = hdm.recommend_configuration(
    platform='arm_cortex_m7',
    objective='memory',
    custom_constraints={'latency_ms': 200, 'memory_mb': 0.6}
)

# Output:
# Model: memory_pruned_trained.tflite
# Quantization: Dynamic Range
# Accuracy: 51.4%
# Latency: 117 ms
# Memory: 0.53 MB