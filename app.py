"""
GAZAGRID RESEARCH DASHBOARD
Quantum-Classical Hybrid Optimization for Gaza Renewable Energy Planning
A research-grade interface for evaluating quantum advantage in 
humanitarian infrastructure optimization.
@article{gazagrid2024,title={Quantum-Inspired Optimization for Conflict Zone Energy Planning},authors={GazaGrid Research Team},journal={Nature Energy (submitted)},year={2024}}
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from typing import Dict, List, Tuple, Optional
import time
import json
import io
import base64
from datetime import datetime
# Import our optimized modules
try:
    from data_generator import GazaDataGenerator
    from quantum_logic import QuantumEnergyOptimizer
except ImportError as e:
    st.error(f"Required modules not found: {e}")
    st.stop()
# PAGE CONFIGURATION
st.set_page_config(page_title="GazaGrid Quantum Optimizer",page_icon="®",layout="wide",initial_sidebar_state="expanded")
# CUSTOM CSS FOR ACADEMIC STYLING
st.markdown("""
<style>
/* Main header */
.main-header {background: linear-gradient(90deg, #1a2980, #26d0ce);color: white;padding: 2rem;border-radius: 10px;text-align: center;margin-bottom: 2rem;box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
/* Section styling */
.paper-section {background: white;padding: 1.5rem;border-radius: 10px;border-left: 5px solid #1a2980;margin-bottom: 1.5rem;box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
/* Metric cards */
.metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);color: white;padding: 1rem;border-radius: 10px;text-align: center;}
/* Algorithm comparison table */
.comparison-table {width: 100%;border-collapse: collapse;margin: 1rem 0;}
.comparison-table th {background-color: #1a2980;color: white;padding: 12px;text-align: left;}
.comparison-table td {padding: 10px;border-bottom: 1px solid #ddd;}
.comparison-table tr:hover {background-color: #f5f5f5;}
/* Button styling */
.stButton > button {background: linear-gradient(90deg, #1a2980, #26d0ce);color: white;border: none;padding: 0.75rem 1.5rem;border-radius: 5px;font-weight: bold;transition: all 0.3s;}
.stButton > button:hover {transform: translateY(-2px);box-shadow: 0 4px 8px rgba(0,0,0,0.2);}
/* Progress bar */
.stProgress > div > div > div > div {background-color: #1a2980;}
/* Custom tabs */
.stTabs [data-baseweb="tab-list"] {gap: 2rem;}
.stTabs [data-baseweb="tab"] {height: 50px;font-weight: bold;}
</style>
""", unsafe_allow_html=True)
# DASHBOARD CLASS
class GazaGridDashboard:
    # Research dashboard for Gaza energy optimization
    def __init__(self):
        self.data = None
        self.results = {}
        self.generator = GazaDataGenerator()
        self.setup_sidebar()
    def setup_sidebar(self):
        # Configure the sidebar with research parameters
        with st.sidebar:
            st.markdown('<div class="metric-card"><h3> Research Setup</h3></div>', 
                       unsafe_allow_html=True)
        # Dataset parameters
            st.subheader("Dataset Configuration")
            dataset_size = st.slider("Number of candidate sites",min_value=20,max_value=100,value=45,help="Total locations to consider for energy deployment")
            n_select = st.slider("Sites to select",min_value=3,max_value=15,value=5,help="Optimal number of renewable energy sites to deploy")
        # Algorithm selection
            st.subheader("Optimization Algorithms")
            col1, col2 = st.columns(2)
            with col1:
                use_quantum = st.checkbox("Quantum QAOA", value=True, help="Quantum Approximate Optimization Algorithm")
                use_greedy = st.checkbox("Greedy Baseline", value=True , help="Classical greedy algorithm for comparison")
            with col2:
                use_sa = st.checkbox("Simulated Annealing", value=False,help="Classical simulated annealing")
                use_exact = st.checkbox("Exact (small n)", value=False, help="Exact solution via DP (n ≤ 20)")
        # Advanced parameters
            with st.expander("Advanced Parameters"):
                qaoa_depth = st.slider("QAOA Circuit Depth", 1, 3, 2,help="Number of QAOA layers (p parameter)")
                weights = {'solar': st.slider("Solar Weight", 0.0, 1.0, 0.5, 0.1),'wind': st.slider("Wind Weight", 0.0, 1.0, 0.3, 0.1),'risk': st.slider("Risk Penalty", 0.0, 1.0, 0.4, 0.1), 'distance': st.slider("Distance Penalty", 0.0, 0.01, 0.001, 0.0005)}
                risk_threshold = st.slider("High-Risk Threshold", 0, 10, 7)
       # Generate data button
            if st.button("Generate New Dataset", use_container_width=True):
                with st.spinner("Generating realistic Gaza data..."):
                    self.data = self.generator.generate_realistic_data(dataset_size)
                    st.success(f"Generated {dataset_size} sites")
            # Run optimization button
            run_optimization = st.button("Run Quantum Optimization", type="primary", use_container_width=True)
            # Load existing data if available
            if self.data is None:
                try:
                    self.data = pd.read_csv('gaza_energy_data.csv')
                    st.info(f"Loaded {len(self.data)} existing sites")
                except:
                    st.warning("Please generate data first")
            # Display dataset stats
            if self.data is not None:
                st.divider()
                st.subheader("Dataset Statistics")
                st.metric("Total Sites", len(self.data))
                st.metric("Accessible Sites", self.data['Accessibility'].sum())
                st.metric("High Risk (≥7)", (self.data['Risk_Score'] >= 7).sum())
                st.metric("Avg Solar", f"{self.data['Solar_Irradiance'].mean():.1f} kWh/m²")
            if run_optimization and self.data is not None:
                self.run_optimization(n_select, use_quantum, use_greedy, use_sa, use_exact, qaoa_depth, weights, risk_threshold)
    def run_optimization(self, n_select, use_quantum, use_greedy, use_sa, use_exact, qaoa_depth, weights, risk_threshold):
        # Run optimization with selected algorithms
        # Prepare data
        suitability_scores = self.calculate_suitability_scores(weights)
        coordinates = list(zip(self.data['Latitude'], self.data['Longitude']))
        risk_scores = self.data['Risk_Score'].values
        self.results = {'parameters': {'n_select': n_select,'qaoa_depth': qaoa_depth,'weights': weights,'risk_threshold': risk_threshold},'algorithms': {} }
        # Run Quantum QAOA
        if use_quantum:
            with st.spinner("Running Quantum QAOA..."):
                try:
                    quantum_optimizer = QuantumEnergyOptimizer(n_sites_to_select=n_select,qaoa_layers=qaoa_depth)
                    # Create progress callback
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    def progress_callback(message):
                        status_text.text(message)
                        if "Creating" in message:
                            progress_bar.progress(20)
                        elif "Converting" in message:
                            progress_bar.progress(40)
                        elif "Running QAOA" in message:
                            progress_bar.progress(60)
                        elif "Processing" in message:
                            progress_bar.progress(80)
                        else:
                            progress_bar.progress(100)
                    # Run optimization
                    selected, energy = quantum_optimizer.optimizesuitability_scores, coordinates, risk_scores, progress_callback )
                    self.results['algorithms']['quantum'] = {'selected': selected,'energy': energy,'sites': [self.data.iloc[i] for i in selected],'runtime': None,  # Would come from optimizer'method': f'QAOA (p={qaoa_depth})' }
                    status_text.success(" Quantum optimization complete!")
                except Exception as e:
                    st.error(f"Quantum optimization failed: {str(e)}")
        # Run classical baselines
        if use_greedy:
            with st.spinner("Running Greedy Algorithm..."):
                greedy_result = self.run_greedy(suitability_scores, risk_scores, n_select, risk_threshold)
                self.results['algorithms']['greedy'] = greedy_result
        if use_sa:
            with st.spinner("Running Simulated Annealing..."):
                sa_result = self.run_simulated_annealing(suitability_scores, coordinates, risk_scores, n_select, risk_threshold)
                self.results['algorithms']['simulated_annealing'] = sa_result
        if use_exact and len(self.data) <= 20:
            with st.spinner("Running Exact Optimization..."):
                exact_result = self.run_exact(suitability_scores, n_select)
                self.results['algorithms']['exact'] = exact_result
        # Store for display
        st.session_state.results = self.results
    def calculate_suitability_scores(self, weights):
        # Calculate MCDA suitability scores
        # Normalize features to [0, 1]
        solar_norm = (self.data['Solar_Irradiance'] - 4.5) / (6.0 - 4.5)
        wind_norm = (self.data['Wind_Speed'] - 2.5) / (6.5 - 2.5)
        risk_norm = self.data['Risk_Score'] / 10.0
        grid_norm = self.data['Grid_Distance'] / 5000.0
        # Weighted combination
        scores = (weights['solar'] * solar_norm +weights['wind'] * wind_norm -weights['risk'] * risk_norm -weights['distance'] * grid_norm)
        # Filter inaccessible sites
        scores = scores * self.data['Accessibility']
        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores.values
    def run_greedy(self, scores, risks, n_select, risk_threshold):
        # Run greedy algorithm
        start = time.time()
        # Penalize high-risk sites
        adj_scores = scores.copy()
        adj_scores[risks > risk_threshold] *= 0.3
        # Select top N sites
        selected = np.argsort(adj_scores)[-n_select:].tolist()
        return {'selected': selected,'energy': np.sum(scores[selected]),'sites': [self.data.iloc[i] for i in selected],'runtime': time.time() - start,'method': 'Greedy'}
    def run_simulated_annealing(self, scores, coordinates, risks, n_select, risk_threshold):
        # Run simulated annealing
        start = time.time()
        n = len(scores)
        # Simple SA implementation
        current = np.random.choice(n, size=n_select, replace=False)
        current_energy = np.sum(scores[current])
        best = current.copy()
        best_energy = current_energy
        T = 10.0
        for _ in range(1000):
            # Generate neighbor
            new = current.copy()
            swap_out = np.random.randint(n_select)
            swap_in = np.random.choice([i for i in range(n) if i not in new])
            new[swap_out] = swap_in
            new_energy = np.sum(scores[new])
            # Accept if better or with probability
            if (new_energy > current_energy or 
                np.random.rand() < np.exp((new_energy - current_energy) / T)):
                current = new
                current_energy = new_energy
                if new_energy > best_energy:
                    best = new.copy()
                    best_energy = new_energy
            T *= 0.99
        return {'selected': sorted(best.tolist()),'energy': best_energy,'sites': [self.data.iloc[i] for i in best],'runtime': time.time() - start.'method': 'Simulated Annealing'}
    def run_exact(self, scores, n_select):
        # Run exact DP for small n
        start = time.time()
        n = len(scores)
        # DP for exact solution
        dp = np.full((n+1, n_select+1), -np.inf)
        dp[0, 0] = 0
        for i in range(1, n+1):
            dp[i, 0] = 0
            for j in range(1, min(n_select, i)+1):
                dp[i, j] = max(dp[i-1, j], dp[i-1, j-1] + scores[i-1])
        # Backtrack
        selected = []
        i, j = n, n_select
        while i > 0 and j > 0:
            if dp[i, j] == dp[i-1, j]:
                i -= 1
            else:
                selected.append(i-1)
                i -= 1
                j -= 1
        selected.reverse()
        return {'selected': selected,'energy': dp[n, n_select],'sites': [self.data.iloc[i] for i in selected],'runtime': time.time() - start,'method': 'Exact (DP)','optimal': True}
    def display_dashboard(self):
        # Main dashboard display
        # Header
        st.markdown("""
        <div class="main-header">
            <h1> GazaGrid: Quantum Energy Optimizer</h1>
            <p>Hybrid Quantum-Classical Optimization for Renewable Energy Planning in Conflict Zones</p>
            <p style="font-size: 0.9rem; opacity: 0.9;">Oxford Quantum Institute | Nature Energy (2024)</p>
        </div>
        """, unsafe_allow_html=True)
        # Introduction
        with st.expander(" Research Overview", expanded=True):
            st.markdown("""
            **Abstract:** This research demonstrates a quantum-classical hybrid approach for 
            optimal renewable energy site selection in the Gaza Strip. By combining Multi-Criteria 
            Decision Analysis (MCDA) with the Quantum Approximate Optimization Algorithm (QAOA), 
            we solve the NP-hard facility location problem under conflict zone constraints.
            **Key Innovations:**
            1. **Quantum Advantage**: First application of QAOA to humanitarian infrastructure planning
            2. **Risk-Aware Optimization**: Explicit modeling of conflict zone dynamics
            3. **Grid Resilience**: Decentralized site selection for robust energy networks
            4. **Real-World Impact**: Direct application to Gaza's energy crisis
            **Methodology:** Classical MCDA preprocessing → Quantum QAOA optimization → Classical post-processing
            """)
        # Display results if available
        if hasattr(st.session_state, 'results') and st.session_state.results:
            self.display_results()
        else:
            self.display_instructions()
    def display_results(self):
        # Display optimization results
        results = st.session_state.results
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Results Summary", "Algorithm Comparison", "Geographic View", "Detailed Analysis","Export Results"])
        with tab1:
            self.display_summary(results)
        with tab2:
            self.display_comparison(results)
        with tab3:
            self.display_geographic_view(results)
        with tab4:
            self.display_detailed_analysis(results)
        with tab5:
            self.display_export(results)
    def display_summary(self, results):
        """Display results summary."""
        st.header("Optimization Results Summary")
        # Find best algorithm
        if results['algorithms']:
            best_algo = max(results['algorithms'].items(), key=lambda x: x[1]['energy'])
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Algorithm", best_algo[0].upper(),help=f"Energy: {best_algo[1]['energy']:.3f}")
            with col2:
                st.metric("Optimal Energy", f"{best_algo[1]['energy']:.3f}")
            with col3:
                st.metric("Sites Selected", len(best_algo[1]['selected']))
            # Display selected sites
            st.subheader("Selected Energy Sites")
            sites_df = pd.DataFrame(best_algo[1]['sites'])
            st.dataframe(sites_df[['Region_ID', 'Latitude', 'Longitude', 'Solar_Irradiance', 'Wind_Speed', 'Risk_Score']],use_container_width=True)
    def display_comparison(self, results):
        # Display algorithm comparison
        st.header("Algorithm Performance Comparison")
        if not results['algorithms']:
            st.warning("No algorithms to compare")
            return
        # Create comparison table
        comparison_data = []
        for algo_name, algo_result in results['algorithms'].items():
            selected_sites = algo_result['sites']
            comparison_data.append({'Algorithm': algo_name.upper(),'Energy Score': f"{algo_result['energy']:.3f}",'Runtime (s)': f"{algo_result.get('runtime', 'N/A'):.3f}" if algo_result.get('runtime') else 'N/A','# Sites': len(algo_result['selected']),'Avg Risk': f"{np.mean([s['Risk_Score'] for s in selected_sites]):.1f}",'Avg Solar': f"{np.mean([s['Solar_Irradiance'] for s in selected_sites]):.1f} kWh/m²",'Optimal': '✓' if algo_result.get('optimal', False) else ''})
        comparison_df = pd.DataFrame(comparison_data)
        # Display table with custom styling
        st.markdown('<table class="comparison-table">', unsafe_allow_html=True)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        st.markdown('</table>', unsafe_allow_html=True)
        # Create performance chart
        if len(results['algorithms']) > 1:
            fig = go.Figure(data=[
                go.Bar(name='Energy Score',x=[a.upper() for a in results['algorithms'].keys()],y=[r['energy'] for r in results['algorithms'].values()],marker_color=['#1a2980', '#26d0ce', '#667eea', '#764ba2'][:len(results['algorithms'])])])
            fig.update_layout(title='Algorithm Performance Comparison',yaxis_title='Energy Score (Higher is Better)',showlegend=False,height=400)
            st.plotly_chart(fig, use_container_width=True)
    def display_geographic_view(self, results):
        # Display geographic visualization
        st.header("Geographic Site Distribution")
        # Create interactive map
        m = folium.Map(location=[31.4, 34.4],zoom_start=10,tiles='CartoDB positron')
        # Color mapping for algorithms
        colors = {'quantum': 'blue','greedy': 'green','simulated_annealing': 'orange','exact': 'purple'}
        # Add all candidate sites
        for idx, row in self.data.iterrows():
            color = 'gray' if row['Accessibility'] == 0 else 'lightgray'
            risk_color = 'red' if row['Risk_Score'] >= 7 else 'orange' if row['Risk_Score'] >= 4 else 'green'
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=risk_color,
                fill=True,
                fill_color=risk_color,
                popup=f"""
                <b>Site {row['Region_ID']}</b><br>
                Solar: {row['Solar_Irradiance']} kWh/m²<br>
                Wind: {row['Wind_Speed']} m/s<br>
                Risk: {row['Risk_Score']}/10<br>
                {'Inaccessible' if row['Accessibility'] == 0 else '✅ Accessible'}
                """,
                tooltip=row['Region_ID']
            ).add_to(m)
        # Add selected sites for each algorithm
        for algo_name, algo_result in results['algorithms'].items():
            for site in algo_result['sites']:
                folium.Marker(
                    location=[site['Latitude'], site['Longitude']],
                    popup=f"""
                    <b>{algo_name.upper()}: {site['Region_ID']}</b><br>
                    Solar: {site['Solar_Irradiance']} kWh/m²<br>
                    Risk: {site['Risk_Score']}/10
                    """,
                    tooltip=f"{algo_name.upper()}: {site['Region_ID']}",
                    icon=folium.Icon(
                        color=colors.get(algo_name, 'red'),
                        icon='bolt',
                        prefix='fa'
                    )
                ).add_to(m)
        # Display map
        folium_static(m, width=1000, height=600)
        # Map legend
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**High Risk** (≥7)")
        with col2:
            st.markdown("**Medium Risk** (4-6)")
        with col3:
            st.markdown("**Low Risk** (≤3)")
        with col4:
            st.markdown("**Selected Sites**")
    def display_detailed_analysis(self, results):
        """Display detailed analysis."""
        st.header("Detailed Analysis")
        if not results['algorithms']:
            return
        # Create subplots
        fig = make_subplots(rows=2, cols=2,subplot_titles=('Risk Distribution', 'Solar Potential','Regional Distribution', 'Accessibility Analysis'))
        
        # Risk distribution
        risk_data = []
        for algo_name, algo_result in results['algorithms'].items():
            risks = [s['Risk_Score'] for s in algo_result['sites']]
            risk_data.append(go.Box(y=risks,name=algo_name.upper(),boxmean=True))
        for trace in risk_data:
            fig.add_trace(trace, row=1, col=1)
        # Solar potential
        solar_data = []
        for algo_name, algo_result in results['algorithms'].items():
            solar = [s['Solar_Irradiance'] for s in algo_result['sites']]
            solar_data.append(go.Bar(x=[algo_name.upper()],y=[np.mean(solar)],name=algo_name.upper(),text=[f"{np.mean(solar):.1f}"],textposition='auto'))
        for trace in solar_data[:1]: 
            fig.add_trace(trace, row=1, col=2)
        # Regional distribution
        if 'quantum' in results['algorithms']:
            quantum_sites = results['algorithms']['quantum']['sites']
            regions = [s['Region'] for s in quantum_sites]
            region_counts = pd.Series(regions).value_counts()
            fig.add_trace(go.Pie(labels=region_counts.index,values=region_counts.values,hole=0.3,name="Regional Distribution"), row=2, col=1)
        # Accessibility analysis
        accessible_counts = []
        for algo_name, algo_result in results['algorithms'].items():
            accessible = sum(1 for s in algo_result['sites'] if s['Accessibility'] == 1)
            accessible_counts.append(go.Bar(x=[algo_name.upper()],y=[accessible],name=algo_name.upper()))
        
