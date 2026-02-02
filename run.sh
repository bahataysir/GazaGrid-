set -e  
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                 GAZAGRID QUANTUM OPTIMIZER             ║"
echo "║          Quantum-Classical Energy Planning for Gaza     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' 
print_status() {echo -e "${BLUE}[STATUS]${NC} $1"}
print_success() {echo -e "${GREEN}[SUCCESS]${NC} $1"}
print_warning() {echo -e "${YELLOW}[WARNING]${NC} $1"}
print_error() {echo -e "${RED}[ERROR]${NC} $1"}
print_status " wait im Checking Python version"
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ $python_version < 3.9 ]]; then
    print_warning "Python $python_version detected. Python 3.9+ recommended."
else
    print_success "Python $python_version ✓"
fi
print_status "wait im Installing dependencies"
if pip install -q -r requirements.txt; then
    print_success "Dependencies installed ✓"
else
    print_error "Failed to install dependencies"
    exit 1
fi
print_status "wait plz!"
if python3 -c "
try:
    import streamlit, qiskit, pandas, numpy, plotly, folium
    print('✓ All core imports successful')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    raise
"; then
    print_success "Health check passed ✓"
else
    print_error "Health check failed"
    exit 1
fi
print_status "Checking data..."
if [ ! -f "gaza_energy_data.csv" ]; then
    print_warning "Data file not found. Generating sample data..."
    if python3 -c "
from data_generator import GazaDataGenerator
print('Generating realistic Gaza energy data...')
gen = GazaDataGenerator(seed=42)
df = gen.generate_realistic_data(45)
df.to_csv('gaza_energy_data.csv', index=False)
print(f'✓ Generated {len(df)} sites')
"; then
        print_success "Data generated ✓"
    else
        print_error "Failed to generate data"
        exit 1
    fi
else
    count=$(wc -l < gaza_energy_data.csv)
    print_success "Found existing data with $((count-1)) sites ✓"
fi
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                     HACKATHON READY                      ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  🎯 5-Minute Demo:                                       ║"
echo "║     1. Generate/load data (sidebar)                      ║"
echo "║     2. Run quantum optimization                          ║"
║     3. Compare algorithms (Results tab)                      ║"
echo "║     4. Show map & impact (Geographic View)               ║"
echo "║     5. Export results (Export tab)                       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  🌐 Dashboard: http://localhost:8501                     ║"
echo "║  🛑 Stop: Press Ctrl+C                                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
# Start Streamlit with clear output
print_status "Starting GazaGrid dashboard..."
echo ""
streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --theme.base "light" \
    --theme.primaryColor "#1a2980" \
    --theme.backgroundColor "#ffffff" \
    --theme.secondaryBackgroundColor "#f0f2f6" \
    --theme.textColor "#262730" \
    --theme.font "sans serif"
