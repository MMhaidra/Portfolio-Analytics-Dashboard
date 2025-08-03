# app.py

import streamlit as st
import pathlib
from dataclasses import dataclass, asdict, field
from datetime import datetime
import pandas as pd
import fitz  # PyMuPDF
# import matplotlib.pyplot as plt # Not used in current viz
# import seaborn as sns # Not used in current viz
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import numpy as np # Not used in current viz
import tempfile
import os
# from io import BytesIO # Not used for temp file handling here
import re

# === Configuration ===
st.set_page_config(
    page_title="Portfolio Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Data Model ===
@dataclass(slots=True, frozen=True)
class Position:
    stock: str
    quantity: float
    price: float
    prmp: float # Purchase Reference Market Price
    statement_date: pd.Timestamp
    amount: float = field(init=False)
    gain: float = field(init=False)
    perf_pct: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'amount', self.quantity * self.price)
        object.__setattr__(self, 'gain', (self.price - self.prmp) * self.quantity)
        object.__setattr__(self, 'perf_pct', (self.price - self.prmp) / self.prmp if self.prmp != 0 else 0)

# French month mapping for date parsing
MONTHS = {
    'janvier':1, 'janv':1, 'fevrier':2, 'février':2, 'fevr':2, 'févr':2,
    'mars':3, 'avril':4, 'avr':4, 'mai':5, 'juin':6,
    'juillet':7, 'juil':7, 'aout':8, 'août':8, 'septembre':9, 'sept':9,
    'octobre':10, 'oct':10, 'novembre':11, 'nov':11, 'decembre':12, 'décembre':12, 'dec':12, 'déc':12
}

def _parse_statement_date(page) -> datetime:
    """Extracts the statement date from a PDF page."""
    txt = page.get_text()
    patterns = [
        r'(\d{1,2})\s+([a-zA-Zéûüçàè\.]+)\s+(\d{4})',
        r'le\s+(\d{1,2})\s+([a-zA-Zéûüçàè\.]+)\s+(\d{4})'
    ]
    for pattern in patterns:
        m = re.search(pattern, txt)
        if m:
            day, month_str, year = m.groups()
            # Normalize unicode characters (e.g., remove accents)
            mo = "".join([ch for ch in month_str.lower().strip('.') if ch.isalnum()])
            month = MONTHS.get(mo, 1)
            return datetime(int(year), month, int(day))
    return datetime.now() # Fallback

def safe_float_convert(value_str):
    """Safely converts a string to a float, handling common formatting issues."""
    if value_str is None or (isinstance(value_str, str) and value_str.strip() == ''):
        return None
    clean_str = str(value_str).strip().replace(',', '.').replace(' ', '').replace('\xa0', '')
    if '%' in clean_str:
        clean_str = clean_str.replace('%', '')
        try:
            return float(clean_str) / 100
        except ValueError:
            return None
    try:
        return float(clean_str)
    except (ValueError, TypeError):
        return None

def is_wafabourse_pdf(doc):
    """Checks if the PDF appears to be from Wafabourse by looking for specific text."""
    # Check first few pages for characteristic text
    for i in range(min(2, len(doc))):
        page = doc.load_page(i)
        text = page.get_text().lower()
        # Look for common Wafabourse terms or structure indicators
        if any(keyword in text for keyword in ['wafabourse', 'portefeuille', 'valeur', 'quantité']):
            return True
    return False

def parse_portfolio_pdfs(uploaded_file): # Accept single file now
    """Parses portfolio data from an uploaded PDF file."""
    all_positions = []
    if not uploaded_file:
        return pd.DataFrame()

    # Save uploaded file to a temporary location to be read by PyMuPDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        doc = fitz.open(tmp_path)
        
        # Validate PDF source
        if not is_wafabourse_pdf(doc):
            st.error("❌ The uploaded PDF does not appear to be a valid Wafabourse portfolio statement. Please check the file and try again.")
            return pd.DataFrame() # Return empty DataFrame on invalid PDF

        for page in doc:
            tables = page.find_tables()
            if not tables:
                continue
            stmt_date = _parse_statement_date(page)
            for t in tables:
                df = t.to_pandas()
                if df.shape[1] < 7: # Ensure enough columns
                    continue
                for idx, row in df.iterrows():
                    stock_val = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
                    # Filter out headers or totals
                    if stock_val.lower() in ['valeur', 'totaux'] or len(stock_val) < 2:
                        continue
                    qty_raw, price_raw, prmp_raw = row.iloc[2], row.iloc[3], row.iloc[6]
                    quantity = safe_float_convert(qty_raw)
                    price = safe_float_convert(price_raw)
                    prmp = safe_float_convert(prmp_raw)
                    # Validate data
                    if all(x is not None and x > 0 for x in [quantity, price, prmp]):
                        all_positions.append(Position(
                            stock=stock_val, quantity=quantity, price=price,
                            prmp=prmp, statement_date=pd.Timestamp(stmt_date)
                        ))
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
        return pd.DataFrame()
    finally:
        # Ensure the temporary file is always deleted
        os.unlink(tmp_path)

    if not all_positions:
        # If no data was found even in a valid PDF, show a warning
        # Note: doc is closed, so re-checking is tricky. Just show a general warning.
        st.warning("⚠️ No valid portfolio data could be extracted from the PDF. The format might be different or empty.")
        return pd.DataFrame()
    return pd.DataFrame([asdict(p) for p in all_positions])

def generate_visualizations(df_sorted):
    """Generates all visualizations for the dashboard."""
    if df_sorted.empty:
        return {}

    df_viz = df_sorted.copy()
    df_viz['weight'] = df_viz['amount'] / df_viz['amount'].sum()
    df_viz['abs_gain'] = df_viz['gain'].abs()

    # --- 1. PORTFOLIO ALLOCATION PIE CHART ---
    pie_fig = go.Figure(data=[go.Pie(
        labels=[stock[:15] for stock in df_viz['stock']],
        values=df_viz['weight'],
        hole=0.4,
        textinfo='label+percent',
        textposition='auto',
        marker=dict(colors=px.colors.qualitative.Set3 + px.colors.qualitative.Pastel),
        hovertemplate='<b>%{label}</b><br>Weight: %{percent}<br>Amount: %{customdata:,.0f} MAD<extra></extra>',
        customdata=df_viz['amount']
    )])
    pie_fig.update_layout(
        title="Portfolio Allocation (All Holdings)",
        font=dict(size=10),
        height=500,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
    )

    # --- 2. PERFORMANCE WATERFALL CHART ---
    waterfall_fig = go.Figure(go.Waterfall(
        name="Performance", orientation="v",
        measure=["relative"] * len(df_viz) + ["total"],
        x=[stock[:10] for stock in df_viz['stock']] + ["Total P&L"],
        y=list(df_viz['gain']) + [df_viz['gain'].sum()],
        text=[f"{x:+,.0f}" for x in df_viz['gain']] + [f"{df_viz['gain'].sum():+,.0f}"],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "green"}},
        decreasing={"marker": {"color": "red"}},
        totals={"marker": {"color": "blue"}}
    ))
    waterfall_fig.update_layout(
        title="Portfolio P&L Waterfall Analysis",
        xaxis_tickangle=-45, height=500,
        xaxis_title="Stocks", yaxis_title="Gain/Loss (MAD)"
    )

    # --- 3. TREEMAP VISUALIZATION ---
    treemap_fig = px.treemap(
        df_viz, path=[px.Constant("Portfolio"), 'stock'], values='amount',
        color='perf_pct', color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
        title="Portfolio Treemap (Size=Amount, Color=Performance)",
        hover_data=['quantity', 'price', 'prmp']
    )
    treemap_fig.update_layout(height=500)

    # --- 4. INTERACTIVE DASHBOARD SUMMARY (Subplots) ---
    dashboard_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance Distribution', 'Amount vs Gain', 
                       'Weight Distribution', 'Gain per MAD Invested'),
        vertical_spacing=0.12, horizontal_spacing=0.1
    )
    # Performance histogram
    dashboard_fig.add_trace(go.Histogram(x=df_viz['perf_pct']*100, name="Performance %", nbinsx=15,
                                        marker_color='lightblue', opacity=0.7), row=1, col=1)
    # Amount vs Gain scatter
    dashboard_fig.add_trace(go.Scatter(
        x=df_viz['amount'], y=df_viz['gain'], mode='markers',
        text=[f"{stock}<br>Perf: {perf*100:+.1f}%" for stock, perf in zip(df_viz['stock'], df_viz['perf_pct'])],
        marker=dict(size=10, color=df_viz['perf_pct'], colorscale='RdYlGn', line=dict(width=1, color='black')),
        name="Amount vs Gain",
        hovertemplate='<b>%{text}</b><br>Amount: %{x:,.0f}<br>Gain: %{y:+,.0f}<extra></extra>'
    ), row=1, col=2)
    # Weight distribution
    dashboard_fig.add_trace(go.Bar(
        x=[stock[:8] for stock in df_viz['stock']], y=df_viz['weight']*100,
        name="Weight %", marker_color='lightcoral',
        text=[f"{w:.1f}%" for w in df_viz['weight']*100], textposition='outside'
    ), row=2, col=1)
    # Efficiency (Gain per MAD invested)
    efficiency = (df_viz['gain'] / df_viz['amount']) * 100
    dashboard_fig.add_trace(go.Bar(
        x=[stock[:8] for stock in df_viz['stock']], y=efficiency,
        name="Efficiency %", marker_color='gold',
        text=[f"{e:+.1f}%" for e in efficiency], textposition='outside'
    ), row=2, col=2)
    dashboard_fig.update_layout(height=800, title_text="Portfolio Dashboard Overview", showlegend=False)
    dashboard_fig.update_xaxes(tickangle=45, row=2, col=1)
    dashboard_fig.update_xaxes(tickangle=45, row=2, col=2)

    # --- 5. TOP WINNERS & LOSERS ---
    winners = df_viz[df_viz['perf_pct'] > 0].sort_values('perf_pct', ascending=True)
    losers = df_viz[df_viz['perf_pct'] < 0].sort_values('perf_pct', ascending=False)

    winners_fig = go.Figure()
    if not winners.empty:
        winners_fig.add_trace(go.Bar(
            y=[stock[:15] for stock in winners['stock']], x=winners['perf_pct'] * 100,
            orientation='h', marker_color='#2E8B57',
            text=[f"{p*100:.1f}%" for p in winners['perf_pct']], textposition='auto'
        ))
        winners_fig.update_layout(
            title='Top Winners', xaxis_title='Performance (%)', yaxis_title='Stock',
            height=400 + len(winners) * 20, margin=dict(l=150)
        )

    losers_fig = go.Figure()
    if not losers.empty:
        losers_fig.add_trace(go.Bar(
            y=[stock[:15] for stock in losers['stock']], x=losers['perf_pct'] * 100,
            orientation='h', marker_color='#DC143C',
            text=[f"{p*100:.1f}%" for p in losers['perf_pct']], textposition='auto'
        ))
        losers_fig.update_layout(
            title='Underperformers', xaxis_title='Performance (%)', yaxis_title='Stock',
            height=400 + len(losers) * 20, margin=dict(l=150)
        )
    else:
        losers_fig.add_annotation(
            x=0.5, y=0.5, text="🎉 No Losing Positions!", font=dict(size=20, color="green"),
            showarrow=False, xref="paper", yref="paper"
        )
        losers_fig.update_layout(title='Underperformers', height=400)

    # --- 6. RISK-RETURN SCATTER PLOT ---
    risk_return_fig = go.Figure()
    risk_return_fig.add_trace(go.Scatter(
        x=df_viz['weight'] * 100, y=df_viz['perf_pct'] * 100,
        mode='markers+text', text=[stock[:8] for stock in df_viz['stock']],
        textposition="top center", textfont=dict(size=10),
        marker=dict(
            size=df_viz['amount'] / 50, # Bubble size by investment amount
            color=['#2E8B57' if x > 0 else '#DC143C' for x in df_viz['perf_pct']],
            opacity=0.7, line=dict(width=1, color='black')
        ),
        hovertemplate='<b>%{text}</b><br>Weight: %{x:.1f}%<br>Performance: %{y:.1f}%<br>Amount: %{customdata:,.0f} MAD<extra></extra>',
        customdata=df_viz['amount']
    ))
    risk_return_fig.add_hline(y=0, line_dash="dash", line_color="black")
    risk_return_fig.update_layout(
        title='Risk-Return Analysis (Bubble size = Investment amount)',
        xaxis_title='Portfolio Weight (%)', yaxis_title='Performance (%)',
        height=500
    )

    # --- 7. PORTFOLIO COMPOSITION BY VALUE RANGES ---
    def categorize_holding(amount):
        if amount >= 3000: return "🔥 Large (≥3K)"
        elif amount >= 1000: return "🟡 Medium (1K-3K)"
        else: return "🔵 Small (<1K)"
    df_viz['size_category'] = df_viz['amount'].apply(categorize_holding)
    
    value_fig = px.pie(df_viz, names='size_category', values='amount',
                      title='Holdings by Value Distribution',
                      color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    # For count distribution, we need to count the number of rows per category
    count_data = df_viz['size_category'].value_counts().reset_index()
    count_data.columns = ['size_category', 'count']
    count_fig = px.pie(count_data, names='size_category', values='count', # Use 'count' column
                      title='Holdings by Count Distribution',
                      color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])

    # --- 8. PORTFOLIO ANALYTICS SUMMARY TABLE ---
    total_invested = (df_viz['amount'] - df_viz['gain']).sum()
    roi = (df_viz['gain'].sum() / total_invested) * 100 if total_invested > 0 else 0
    winning_positions = len(df_viz[df_viz['gain'] > 0])
    losing_positions = len(df_viz[df_viz['gain'] < 0])
    
    # Prepare metrics data ensuring clean types for Arrow compatibility
    # Create a DataFrame with only string/object columns to avoid Arrow issues
    metrics_data = {
        "Metric": [
            "Total Positions", "Current Value", "Total Invested", 
            "Unrealized P&L", "ROI", "Winning Positions",
            "Losing Positions", "Best Performer", "Worst Performer",
            "Portfolio Concentration"
        ],
        "Value": [
            str(len(df_viz)),
            f"{df_viz['amount'].sum():,.0f} MAD",
            f"{total_invested:,.0f} MAD",
            f"{df_viz['gain'].sum():+,.0f} MAD",
            f"{roi:+.1f}%",
            f"{winning_positions}/{len(df_viz)} ({winning_positions/len(df_viz)*100:.0f}%)",
            f"{losing_positions}/{len(df_viz)} ({losing_positions/len(df_viz)*100:.0f}%)",
            f"{df_viz.loc[df_viz['perf_pct'].idxmax(), 'stock']} ({df_viz['perf_pct'].max()*100:+.1f}%)",
            f"{df_viz.loc[df_viz['perf_pct'].idxmin(), 'stock']} ({df_viz['perf_pct'].min()*100:+.1f}%)",
            f"Top 5 = {df_viz['weight'].head(5).sum()*100:.1f}%"
        ]
    }
    metrics_df = pd.DataFrame(metrics_data)

    # --- 9. PORTFOLIO SUMMARY TABLE ---
    portfolio_df = pd.DataFrame({
        "Stock": [row['stock'][:19] for _, row in df_sorted.iterrows()],
        "Qty": [f"{row['quantity']:.0f}" for _, row in df_sorted.iterrows()],
        "Price": [f"{row['price']:.2f}" for _, row in df_sorted.iterrows()],
        "Amount": [f"{row['amount']:,.0f}" for _, row in df_sorted.iterrows()],
        "PRMP": [f"{row['prmp']:.2f}" for _, row in df_sorted.iterrows()],
        "Gain/Loss": [f"{'🟢' if row['gain'] >= 0 else '🔴'} {row['gain']:+,.0f}" for _, row in df_sorted.iterrows()],
        "Perf %": [f"{'🟢' if row['perf_pct'] >= 0 else '🔴'} {row['perf_pct']*100:+.1f}%" for _, row in df_sorted.iterrows()]
    })

    return {
        "pie": pie_fig, "waterfall": waterfall_fig, "treemap": treemap_fig,
        "dashboard": dashboard_fig, "winners": winners_fig, "losers": losers_fig,
        "risk_return": risk_return_fig, "value_distribution": value_fig,
        "count_distribution": count_fig, "metrics_df": metrics_df, # Pass the string-only DF
        "portfolio_df": portfolio_df,
        "df_viz": df_viz # Pass df_viz for insights
    }

# === Main App Logic ===
st.title("📊 Portfolio Analytics Dashboard")
st.markdown("""
Upload your **single** Wafabourse portfolio statement (PDF format) to analyze your investment performance. 
The dashboard will extract positions and generate comprehensive visualizations.
""")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Portfolio Statement")
    # Changed to accept only one file
    uploaded_file = st.file_uploader(
        "Upload a single PDF portfolio statement", 
        type=["pdf"], accept_multiple_files=False # Single file only
    )
    st.divider()
    st.header("About This Dashboard")
    st.markdown("""
    This dashboard is designed to help you analyze your Wafabourse portfolio statements.

    **Key Features:**

    *   **Data Extraction:** Automatically parses your PDF statement to extract stock holdings, quantities, prices, and purchase costs.
    *   **Performance Analysis:** Visualizes your portfolio's unrealized profit & loss (P&L) through various charts, identifying top winners and underperformers.
    *   **Risk Assessment:** Analyzes the risk-return profile of your investments, showing how performance relates to the weight of each holding in your portfolio.
    *   **Portfolio Composition:** Breaks down your portfolio by the value and number of holdings, giving you a clear picture of diversification.
    *   **Actionable Insights:** Provides summarized metrics and textual analysis to help you understand your portfolio's health and make informed decisions.

    **How to Use:**

    1.  Obtain your portfolio statement PDF from the Wafabourse platform.
    2.  Upload the PDF using the file uploader in the sidebar.
    3.  Explore the different tabs to view visualizations and insights.
    4.  Download the processed data as a CSV for further analysis if needed.

    """)

# Main content
if not uploaded_file:
    st.info("Please upload a single PDF portfolio statement to get started.")
    # Show example
    st.subheader("Example Output")
    st.image("https://via.placeholder.com/800x400?text=Portfolio+Dashboard+Visualizations", 
             caption="Sample portfolio analysis visualizations")
else:
    with st.spinner("Processing PDF file..."):
        df = parse_portfolio_pdfs(uploaded_file) # Pass single file
        if df.empty:
            # Error message is handled inside parse_portfolio_pdfs or implicitly if empty
            pass # Do nothing, error/warning already shown
        else:
            df_sorted = df.sort_values('amount', ascending=False)
            total_amount = df_sorted['amount'].sum()
            total_gain = df_sorted['gain'].sum()
            total_invested = total_amount - total_gain

            # Display statement date
            statement_date = df['statement_date'].iloc[0].strftime('%d %b %Y')
            st.success(f"✅ Successfully processed portfolio statement dated {statement_date}")

            # Show summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Positions", len(df))
            col2.metric("Current Value", f"{total_amount:,.0f} MAD")
            col3.metric("Total Invested", f"{total_invested:,.0f} MAD")
            col4.metric("Unrealized P&L", f"{total_gain:+,.0f} MAD", 
                       delta=f"{(total_gain/total_invested*100):+.1f}%" if total_invested > 0 else "N/A")

            # Generate visualizations
            viz = generate_visualizations(df_sorted)

            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview", "📈 Performance Analysis", 
                "📉 Risk Analysis", "💎 Portfolio Composition",
                "📋 Detailed Data"
            ])

            with tab1:
                st.subheader("Portfolio Summary")
                # Use the display-friendly DataFrame
                st.dataframe(viz["portfolio_df"], use_container_width=True, hide_index=True)
                st.subheader("Key Metrics")
                # Use the display-friendly DataFrame (strings only)
                st.dataframe(viz["metrics_df"], use_container_width=True, hide_index=True)
                st.subheader("Portfolio Allocation")
                st.plotly_chart(viz["pie"], use_container_width=True, key="tab1_pie")
                st.subheader("Portfolio Treemap")
                st.plotly_chart(viz["treemap"], use_container_width=True, key="tab1_treemap")

            with tab2:
                st.subheader("Performance Waterfall")
                st.plotly_chart(viz["waterfall"], use_container_width=True, key="tab2_waterfall")
                
                # --- Textual Insights for Performance ---
                df_v = viz["df_viz"]
                top_3_winners = df_v.nlargest(3, 'perf_pct')[['stock', 'perf_pct']]
                top_3_losers = df_v.nsmallest(3, 'perf_pct')[['stock', 'perf_pct']]
                avg_perf = df_v['perf_pct'].mean() * 100
                perf_insight = f"""
                **Performance Insights:**
                
                *   Your portfolio's average performance is **{avg_perf:.2f}%**.
                *   The top 3 performing stocks are:
                    1. {top_3_winners.iloc[0]['stock']} ({top_3_winners.iloc[0]['perf_pct']*100:+.1f}%)
                    2. {top_3_winners.iloc[1]['stock']} ({top_3_winners.iloc[1]['perf_pct']*100:+.1f}%)
                    3. {top_3_winners.iloc[2]['stock']} ({top_3_winners.iloc[2]['perf_pct']*100:+.1f}%)
                *   The 3 underperforming stocks are:
                    1. {top_3_losers.iloc[0]['stock']} ({top_3_losers.iloc[0]['perf_pct']*100:+.1f}%)
                    2. {top_3_losers.iloc[1]['stock']} ({top_3_losers.iloc[1]['perf_pct']*100:+.1f}%)
                    3. {top_3_losers.iloc[2]['stock']} ({top_3_losers.iloc[2]['perf_pct']*100:+.1f}%)
                """
                st.markdown(perf_insight)
                
                st.subheader("Top Performers & Underperformers")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(viz["winners"], use_container_width=True, key="tab2_winners")
                with col2:
                    st.plotly_chart(viz["losers"], use_container_width=True, key="tab2_losers")
                st.subheader("Performance Distribution")
                st.plotly_chart(viz["dashboard"], use_container_width=True, key="tab2_dashboard")

            with tab3:
                st.subheader("Risk-Return Analysis")
                st.plotly_chart(viz["risk_return"], use_container_width=True, key="tab3_risk_return")
                
                # --- Textual Insights for Risk ---
                df_v = viz["df_viz"]
                # Find largest holding by amount
                largest_holding = df_v.loc[df_v['amount'].idxmax()]
                # Find highest weight holding
                highest_weight_holding = df_v.loc[df_v['weight'].idxmax()]
                # Find best performer
                best_perf = df_v.loc[df_v['perf_pct'].idxmax()]
                # Find worst performer
                worst_perf = df_v.loc[df_v['perf_pct'].idxmin()]
                
                risk_insight = f"""
                **Risk Insights:**

                *   Your largest single investment is **{largest_holding['stock']}** worth **{largest_holding['amount']:,.0f} MAD** ({largest_holding['weight']*100:.1f}% of portfolio).
                *   The stock with the highest portfolio weight is **{highest_weight_holding['stock']}** at **{highest_weight_holding['weight']*100:.1f}%**.
                *   **{best_perf['stock']}** is your best performer with a **{best_perf['perf_pct']*100:+.1f}%** return.
                *   **{worst_perf['stock']}** is your worst performer with a **{worst_perf['perf_pct']*100:+.1f}%** return.
                
                Consider if your largest holdings align with your risk tolerance and if the weight of your worst performers is significant.
                """
                st.markdown(risk_insight)

                st.subheader("Portfolio Composition by Size")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(viz["value_distribution"], use_container_width=True, key="tab3_value_dist")
                with col2:
                    st.plotly_chart(viz["count_distribution"], use_container_width=True, key="tab3_count_dist")

            with tab4:
                st.subheader("Portfolio Allocation")
                st.plotly_chart(viz["pie"], use_container_width=True, key="tab4_pie") # Key added
                
                # --- Textual Insights for Composition ---
                df_v = viz["df_viz"]
                top_5_value = df_v.nlargest(5, 'amount')
                top_5_value_sum = top_5_value['amount'].sum()
                top_5_weight = top_5_value_sum / total_amount * 100

                size_counts = df_v['size_category'].value_counts()
                most_common_size = size_counts.idxmax()
                most_common_size_count = size_counts.max()

                comp_insight = f"""
                **Composition Insights:**

                *   The top 5 holdings by value account for **{top_5_weight:.1f}%** of your total portfolio value ({top_5_value_sum:,.0f} MAD / {total_amount:,.0f} MAD).
                *   You have the most holdings in the **'{most_common_size}'** category ({most_common_size_count} holdings).
                
                A high concentration in the top holdings might indicate less diversification. The distribution by count shows where you have the most individual positions.
                """
                st.markdown(comp_insight)

                st.subheader("Portfolio Composition by Size")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(viz["value_distribution"], use_container_width=True, key="tab4_value_dist") # Key added
                with col2:
                    st.plotly_chart(viz["count_distribution"], use_container_width=True, key="tab4_count_dist") # Key added
                st.subheader("Risk-Return Analysis")
                st.plotly_chart(viz["risk_return"], use_container_width=True, key="tab4_risk_return") # Key added

            with tab5:
                st.subheader("Raw Portfolio Data")
                st.dataframe(df_sorted, use_container_width=True)
                csv = df_sorted.to_csv(index=False)
                st.download_button(
                    label="Download Portfolio Data as CSV",
                    data=csv, file_name="portfolio_data.csv", mime="text/csv"
                )
