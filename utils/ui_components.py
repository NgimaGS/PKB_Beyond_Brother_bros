import streamlit as st

def inject_custom_css():
    """
    Injects the Global Professional UI Styling (SaaS Standard).
    This keeps the main app clean and separates styling from logic.
    """
    st.markdown("""
        <style>
        /* Import Inter Font - The gold standard for modern UI */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Global Reset & Base Styles */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0e1117; /* Deep Slate Blue-Grey */
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #e0e6ed;
            line-height: 1.6;
        }

        /* Typography Hierarchy */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: #f8fafc;
            letter-spacing: -0.01em;
        }

        p, div, span, label {
            font-weight: 400;
            color: #cbd5e1;
        }

        /* Layout Expansion & Scroll Optimization */
        footer { visibility: hidden; }
        .block-container { 
            padding-top: 4rem !important; 
            padding-bottom: 1rem !important; 
            max-width: 1400px !important; 
            margin: 0 auto; 
        }
        
        /* Ensure only the chat container scrolls internally */
        html, body, [data-testid="stAppViewContainer"] {
            overflow: hidden;
        }

        /* Sidebar - Ingestion Hub Hub Styling */
        section[data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #30363d;
            width: 320px !important;
        }

        /* Result Cards - Crisp & Technical */
        .nexus-card {
            background: #1f2937; /* Gray 800 */
            border: 1px solid #374151;
            border-radius: 10px;
            padding: 1.25rem;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            transition: transform 0.2s ease, border-color 0.2s;
        }
        .nexus-card:hover {
            border-color: #3b82f6;
            transform: translateY(-2px);
        }

        /* Metric Tags */
        .match-tag {
            background: rgba(16, 185, 129, 0.1); 
            color: #34d399; 
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        /* Metadata Labels */
        .meta-label {
            font-size: 0.75rem;
            color: #94a3b8; 
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Dynamic Engine Badges */
        .engine-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 12px;
            letter-spacing: 0.02em;
        }
        .badge-ml { background: rgba(59, 130, 246, 0.1); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.2); }
        .badge-dl { background: rgba(139, 92, 246, 0.1); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.2); }

        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0e1117; }
        ::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #4b5563; }

        /* File Sidebar UI Refinement */
        .parallax-wrapper {
            width: 100%;
            overflow: hidden;
            position: relative;
            background: #1f2937;
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 6px;
            border: 1px solid #374151;
            cursor: help;
        }
        
        .parallax-text {
            white-space: nowrap;
            display: inline-block;
            font-size: 12px;
            color: #cbd5e1;
            transition: color 0.2s;
        }

        .parallax-wrapper:hover .parallax-text {
            color: #3b82f6;
            animation: scroll-text 8s linear infinite;
        }

        @keyframes scroll-text {
            0% { transform: translateX(0); }
            100% { transform: translateX(-50%); }
        }
        </style>
        """, unsafe_allow_html=True)

def render_header(num_docs, engine_mode, status_text):
    """Renders the global application header with active engine badges."""
    engine_badge = f"<span class='engine-badge badge-{'ml' if engine_mode == 'Machine Learning' else 'dl'}'>{engine_mode} Active</span>"
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem; margin-top: -1rem;">
            {engine_badge}
            <h2 style="margin-bottom: 0.25rem; font-size: 24px; color: white;">NLP Semantic Workspace</h2>
            <p style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                Status: {status_text} • Active Intelligence: {num_docs} Documents
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_sidebar_branding():
    """Renders the top branding element in the sidebar."""
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="font-size: 20px; margin-bottom: 5px; color: white;">💠 NLP Engine</h2>
            <p style="font-size: 11px; color: #64748b; font-weight: 500; letter-spacing: 0.05em;">HYBRID INTELLIGENCE CORE</p>
        </div>
    """, unsafe_allow_html=True)

def render_token_report(input_tokens, eval_tokens, total_tokens):
    """Renders a technical token consumption report in the UI."""
    st.markdown(f"""
        <div style="display: flex; gap: 10px; margin-top: 10px; border-top: 1px solid #30363d; padding-top: 10px;">
            <div style="flex: 1; text-align: center;">
                <p class="meta-label" style="font-size: 9px;">Input Tokens</p>
                <p style="font-size: 12px; font-weight: 600; color: #60a5fa;">{input_tokens}</p>
            </div>
            <div style="flex: 1; text-align: center;">
                <p class="meta-label" style="font-size: 9px;">Output Tokens</p>
                <p style="font-size: 12px; font-weight: 600; color: #34d399;">{eval_tokens}</p>
            </div>
            <div style="flex: 1; text-align: center;">
                <p class="meta-label" style="font-size: 9px;">Total Cycle</p>
                <p style="font-size: 12px; font-weight: 600; color: #f8fafc;">{total_tokens}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

def get_plotly_template():
    """Returns a customized 'Dark Nebula' theme for Plotly 3D charts."""
    import plotly.graph_objects as go
    
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', color='#94a3b8'),
            scene=dict(
                xaxis=dict(gridcolor='#334155', zerolinecolor='#334155', showbackground=False),
                yaxis=dict(gridcolor='#334155', zerolinecolor='#334155', showbackground=False),
                zaxis=dict(gridcolor='#334155', zerolinecolor='#334155', showbackground=False),
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
    )

def render_spatial_inspector(fig, galaxy_stats=None):
    """
    Renders a robust HTML/JS component using official Plotly serialization
    to ensure the 3D map appears on all browsers, including the Knowledge Inspector.
    """
    import plotly.io as pio
    import streamlit.components.v1 as components
    import warnings

    # Explicitly suppress the 2026 deprecation notice to keep the user console clean
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*st.components.v1.html.*")

    # Use Official Plotly HTML generation for maximum stability
    # This includes the full Plotly library and sets up the DIV correctly.
    base_html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
    
    html_content = f"""
    <div style="display:flex; height:700px; background:#0d1117; border-radius:12px; overflow:hidden; border:1px solid #30363d; font-family:sans-serif;">
        <div id="inspector-pane" style="flex:1; background:rgba(15, 23, 42, 0.7); border-right:1px solid #30363d; padding:20px; color:#e6edf3; overflow-y:auto; backdrop-filter:blur(10px);">
            <div style="font-size:10px; text-transform:uppercase; letter-spacing:0.1em; color:#58a6ff; font-weight:600; margin-bottom:15px; border-bottom:1px solid rgba(88, 166, 255, 0.2); padding-bottom:8px;">Knowledge Inspector</div>
            <div id="inspector-content" style="font-size:13px; line-height:1.6; color:#8b949e;">
                Hover over a point in the 3D space to inspect knowledge metadata.
            </div>
            
            <div id="galaxy-intelligence" style="margin-top:auto; padding-top:20px; border-top:1px solid #30363d; font-size:11px;">
                <div style="font-size:9px; text-transform:uppercase; color:#58a6ff; letter-spacing:0.05em; margin-bottom:10px;">
                    { 'Total Universe Index' if galaxy_stats and 'galaxy_map' in galaxy_stats else 'Galaxy Intelligence' }
                </div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-bottom:12px;">
                    <div style="background:#161b22; padding:8px; border-radius:4px; border:1px solid #30363d;">
                        <div style="color:#6e7681; font-size:8px; text-transform:uppercase;">Entities</div>
                        <div style="color:#f0f6fc; font-size:14px; font-weight:600;">{galaxy_stats.get('docs', 'N/A') if galaxy_stats else 'N/A'}</div>
                    </div>
                    <div style="background:#161b22; padding:8px; border-radius:4px; border:1px solid #30363d;">
                        <div style="color:#6e7681; font-size:8px; text-transform:uppercase;">Fragments</div>
                        <div style="color:#f0f6fc; font-size:14px; font-weight:600;">{galaxy_stats.get('segments', 'N/A') if galaxy_stats else 'N/A'}</div>
                    </div>
                </div>
                
                { 
                  f'''<div style="font-size:9px; text-transform:uppercase; color:#58a6ff; letter-spacing:0.05em; margin:15px 0 8px 0;">Galaxy Directory</div>
                      <div style="max-height:150px; overflow-y:auto; border:1px solid #30363d; border-radius:4px; background:#0d1117; padding:5px;">
                          {" ".join([f'<div style="padding:4px 8px; border-bottom:1px solid #161b22; font-size:10px; color:#c9d1d9;"><span style="color:#238636; font-weight:600;">G{k}:</span> {", ".join(v)}</div>' for k, v in galaxy_stats['galaxy_map'].items()])}
                      </div>''' 
                  if galaxy_stats and 'galaxy_map' in galaxy_stats else 
                  f'''<div style="color:#8b949e; line-height:1.4;">
                        <span style="color:#58a6ff;">●</span> Semantic Core: {", ".join(galaxy_stats['topics']) if galaxy_stats and galaxy_stats.get('topics') else 'Global Index'}
                      </div>'''
                }
            </div>
        </div>
        <div id="chart-wrapper" style="flex:3; position:relative;">
            {base_html}
        </div>
    </div>
    <script>
        (function() {{
            function initHoverHook() {{
                const chartDiv = document.querySelector('.plotly-graph-div');
                if (!chartDiv) {{
                    setTimeout(initHoverHook, 100);
                    return;
                }}
                
                chartDiv.on('plotly_hover', function(data){{
                    const point = data.points[0];
                    const custom = point.customdata || [];
                    
                    const source = custom[1] || 'Unknown';
                    const page = custom[2] || 'N/A';
                    const snippet = custom[3] || 'No snippet available.';
                    const galaxy = custom[4] || '0';
                    
                    document.getElementById('inspector-content').innerHTML = `
                        <div style="margin-bottom:20px;">
                            <div style="font-size:9px; color:#58a6ff; margin-bottom:5px; text-transform:uppercase;">Origin Source</div>
                            <div style="font-size:14px; color:#f0f6fc; font-weight:500;">${{source}}</div>
                        </div>
                        <div style="margin-bottom:20px;">
                            <div style="font-size:9px; color:#58a6ff; margin-bottom:5px; text-transform:uppercase;">Context</div>
                            <div style="font-size:13px; color:#c9d1d9;">Page ${{page}} // Galaxy: ${{galaxy}}</div>
                        </div>
                        <div>
                            <div style="font-size:9px; color:#58a6ff; margin-bottom:5px; text-transform:uppercase;">Semantic Fragment</div>
                            <div style="background:#161b22; padding:12px; border-radius:6px; font-size:13px; font-style:italic; border-left:2px solid #238636; line-height:1.6; color:#8b949e;">
                                "${{snippet}}..."
                            </div>
                        </div>
                    `;
                }});
            }}
            initHoverHook();
        }})();
    </script>
    """
    # Using v1.html to ensure secure Javascript execution for the hover-inspector hook
    components.html(html_content, height=720)
