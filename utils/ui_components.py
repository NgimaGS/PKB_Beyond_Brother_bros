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

        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0e1117; }
        ::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #4b5563; }
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
