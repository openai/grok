#!/bin/bash
# Quick start script for modular addition grokking experiments

echo "=================================="
echo "Modular Addition Grokking Quick Start"
echo "=================================="
echo ""

# Check if running from project root
if [ ! -f "grok/modular_arithmetic.py" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Test 1: Data generation
echo "[1/4] Testing data generation..."
python -m grok.modular_arithmetic
if [ $? -ne 0 ]; then
    echo "Error: Data generation test failed"
    exit 1
fi
echo "✓ Data generation test passed"
echo ""

# Test 2: Single experiment (quick)
echo "[2/4] Running quick training test (p=59, α=0.3, 1000 steps)..."
python -m grok.train_modular_addition \
    --modulus 59 \
    --train_fraction 0.3 \
    --max_steps 1000 \
    --log_dir logs/quick_test \
    --experiment_name quick_test

if [ $? -ne 0 ]; then
    echo "Error: Training test failed"
    exit 1
fi
echo "✓ Training test passed"
echo ""

# Test 3: Multi-alpha experiment
echo "[3/4] Running multi-alpha experiment (p=59, α∈{0.2,0.4}, 5000 steps)..."
python scripts/run_alpha_experiments.py \
    --modulus 59 \
    --alpha_values 0.2 0.4 \
    --max_steps 5000 \
    --log_dir logs/alpha_test \
    --seeds 0

if [ $? -ne 0 ]; then
    echo "Error: Multi-alpha experiment failed"
    exit 1
fi
echo "✓ Multi-alpha experiment passed"
echo ""

# Test 4: Visualization
echo "[4/4] Generating visualizations..."
python scripts/visualize_grokking.py \
    --log_dir logs/alpha_test \
    --modulus 59

if [ $? -ne 0 ]; then
    echo "Error: Visualization failed"
    exit 1
fi
echo "✓ Visualization passed"
echo ""

echo "=================================="
echo "Quick Start Complete!"
echo "=================================="
echo ""
echo "All tests passed successfully!"
echo ""
echo "Next steps:"
echo "  1. Check logs/alpha_test/plots/ for generated plots"
echo "  2. Run full experiment:"
echo "     python scripts/run_alpha_experiments.py \\"
echo "         --modulus 97 \\"
echo "         --alpha_values 0.1 0.2 0.3 0.4 0.5 \\"
echo "         --max_steps 50000"
echo ""
echo "  3. Read MODULAR_ADDITION_README.md for detailed documentation"
echo ""
