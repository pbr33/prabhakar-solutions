import streamlit as st

class ScenarioModelingTab:
    def __init__(self, symbol, market_data, ui_components):
        self.symbol = symbol
        self.market_data = market_data
        self.ui = ui_components
    
    def render(self):
        st.markdown("## 🎭 Scenario Modeling & Monte Carlo Analysis")
        st.markdown("*Multiple probability-weighted future scenarios for strategic planning*")
        
        scenarios = ScenarioEngine.generate_scenarios(symbol, market_data)
        
        # Scenario probability chart
        scenario_names = [name.split(' ')[1] + ' ' + name.split(' ')[2] for name in scenarios.keys()]
        probabilities = [scenario['probability'] for scenario in scenarios.values()]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        fig_prob = go.Figure(data=[
            go.Bar(x=scenario_names, y=probabilities, marker_color=colors,
                   text=[f"{p}%" for p in probabilities], textposition='auto')
        ])
        fig_prob.update_layout(
            title="📊 Scenario Probability Distribution",
            xaxis_title="Scenarios",
            yaxis_title="Probability (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # Scenario details table
        st.markdown("### 📋 Detailed Scenario Analysis")
        
        for scenario_name, details in scenarios.items():
            with st.expander(f"{scenario_name} - {details['probability']}% Probability", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**🎯 Target Price:** ${details['target_price']:.2f}")
                    st.markdown(f"**⏰ Timeframe:** {details['timeframe']}")
                    st.markdown(f"**📈 Expected Return:** {((details['target_price'] / market_data['Close'].iloc[-1]) - 1) * 100:+.1f}%")
                    
                    st.markdown("**🔥 Key Catalysts:**")
                    for catalyst in details['catalysts']:
                        st.markdown(f"• {catalyst}")
                    
                    st.markdown(f"**📊 Conditions:** {details['conditions']}")
                    st.markdown(f"**⚠️ Risk Factors:** {details['risk_factors']}")
                
                with col2:
                    # Mini probability gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = details['probability'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Probability"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': colors[list(scenarios.keys()).index(scenario_name)]},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgray"},
                                {'range': [25, 75], 'color': "gray"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50}
                        }
                    ))
                    fig_gauge.update_layout(height=200)
                    st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Monte Carlo simulation button
        st.markdown("---")
        if st.button("🎲 Run Monte Carlo Simulation", key="monte_carlo"):
            with st.spinner("Running 10,000 simulations..."):
                # Simulate Monte Carlo
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Generate simulation results
                current_price = market_data['Close'].iloc[-1]
                simulation_results = np.random.normal(1.05, 0.15, 10000) * current_price
                
                fig_monte = go.Figure(data=[go.Histogram(x=simulation_results, nbinsx=50)])
                fig_monte.update_layout(
                    title="📈 Monte Carlo Price Distribution (90 days)",
                    xaxis_title="Price ($)",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig_monte, use_container_width=True)
                
                # Simulation statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Price", f"${np.mean(simulation_results):.2f}")
                with col2:
                    st.metric("95% VaR", f"${np.percentile(simulation_results, 5):.2f}")
                with col3:
                    st.metric("Upside (95%)", f"${np.percentile(simulation_results, 95):.2f}")
                with col4:
                    st.metric("Std Dev", f"${np.std(simulation_results):.2f}")