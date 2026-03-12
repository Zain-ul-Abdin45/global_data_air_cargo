"""
GlobalData Lead Pipeline Dashboard
------------------------------------
Run:  streamlit run dashboard.py
Deps: pip install streamlit plotly pandas openpyxl
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="GlobalData Lead Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lodige Brand Colours ──────────────────────────────────────
LODIGE_BLUE      = "#003F7D"
LODIGE_LIGHT     = "#0066CC"
LODIGE_ACCENT    = "#0099CC"
LODIGE_SILVER    = "#E8EEF5"
LODIGE_WHITE     = "#FFFFFF"
CHART_COLORS     = [LODIGE_BLUE, LODIGE_LIGHT, LODIGE_ACCENT,
                    "#0B5394", "#3D85C8", "#6FA8DC", "#9FC5E8", "#CFE2F3"]

# ── Custom CSS ────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* Global */
  html, body, [class*="css"] {{
    font-family: 'Segoe UI', Arial, sans-serif;
  }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background: {LODIGE_BLUE};
  }}
  section[data-testid="stSidebar"] * {{
    color: {LODIGE_WHITE} !important;
  }}
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stMultiSelect label,
  section[data-testid="stSidebar"] .stRadio label {{
    color: {LODIGE_WHITE} !important;
    font-weight: 600;
  }}

  /* Header bar */
  .dash-header {{
    background: linear-gradient(135deg, {LODIGE_BLUE} 0%, {LODIGE_LIGHT} 100%);
    padding: 1.4rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .dash-header h1 {{
    color: {LODIGE_WHITE};
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0;
  }}
  .dash-header p {{
    color: rgba(255,255,255,0.75);
    font-size: 0.82rem;
    margin: 0.2rem 0 0;
  }}
  .dash-badge {{
    background: rgba(255,255,255,0.15);
    color: white;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.3);
  }}

  /* KPI cards */
  .kpi-card {{
    background: {LODIGE_WHITE};
    border: 1px solid #D0DFF0;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    border-left: 5px solid {LODIGE_BLUE};
    box-shadow: 0 2px 8px rgba(0,63,125,0.08);
  }}
  .kpi-card .kpi-val {{
    font-size: 2.4rem;
    font-weight: 800;
    color: {LODIGE_BLUE};
    line-height: 1;
    margin-bottom: 0.2rem;
  }}
  .kpi-card .kpi-label {{
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #6B8CB5;
  }}
  .kpi-card .kpi-sub {{
    font-size: 0.78rem;
    color: #8FAAC8;
    margin-top: 0.3rem;
  }}
  .kpi-card-teal {{ border-left-color: {LODIGE_ACCENT}; }}
  .kpi-card-teal .kpi-val {{ color: {LODIGE_ACCENT}; }}
  .kpi-card-mid {{ border-left-color: {LODIGE_LIGHT}; }}
  .kpi-card-mid .kpi-val {{ color: {LODIGE_LIGHT}; }}

  /* Section headers */
  .section-title {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {LODIGE_LIGHT};
    margin-bottom: 0.5rem;
    margin-top: 0.3rem;
  }}

  /* Table */
  .stDataFrame {{
    border: 1px solid #D0DFF0 !important;
    border-radius: 8px !important;
  }}

  /* Upload area */
  .uploadedFile {{
    background: {LODIGE_SILVER} !important;
  }}

  /* Divider */
  hr {{ border-color: #D0DFF0; }}

  /* Hide streamlit branding */
  #MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ── Seniority classifier ──────────────────────────────────────
SENIORITY_MAP = {
    "C-Suite":    ["ceo","cto","coo","cfo","chief","president"],
    "VP":         ["vp ","vice president","v.p"],
    "Director":   ["director","head of","head,"],
    "Manager":    ["manager","manger","managing"],
    "Consultant": ["consultant","advisor","adviser"],
    "Engineer":   ["engineer","engineering","technical","mechanic"],
    "Founder":    ["founder","co-founder","owner"],
    "Sales":      ["sales","business development","account","bd "],
    "Intern":     ["intern","trainee","student","graduate"],
}

def classify(title):
    if pd.isna(title): return "Other"
    t = str(title).lower().strip()
    for band, kws in SENIORITY_MAP.items():
        if any(k in t for k in kws): return band
    return "Other"


# ── Excel serial date fix ─────────────────────────────────────
def fix_excel_date(val):
    try:
        n = float(str(val))
        if 40000 < n < 60000:
            from datetime import timedelta
            return (datetime(1899, 12, 30) + timedelta(days=int(n))).strftime("%Y-%m-%d")
    except: pass
    return val


# ── Data loader ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file, dtype=str)
        else:
            df = pd.read_excel(file, dtype=str)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return pd.DataFrame()

    # Fix dates
    if "Date" in df.columns:
        df["Date"] = df["Date"].apply(fix_excel_date)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop nulls
    if "City" in df.columns:
        df = df.drop(columns=["City"])

    # Dedup
    if "_row_hash" in df.columns:
        df = df.drop_duplicates(subset=["_row_hash"]).drop(columns=["_row_hash"])

    # Year / Month
    if "Date" in df.columns:
        df["Year"]       = df["Date"].dt.year.astype("Int64")
        df["Month"]      = df["Date"].dt.month.astype("Int64")
        df["Month_Name"] = df["Date"].dt.strftime("%b")
        df["YearMonth"]  = df["Date"].dt.to_period("M").astype(str)

    # Seniority
    if "Seniority" not in df.columns or df["Seniority"].isna().all():
        df["Seniority"] = df["Job Title"].apply(classify)

    # Decision maker flag
    df["Is_DM"] = df["Seniority"].isin(["C-Suite","VP","Director", "Founder"])

    return df


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://www.lodige.com/fileadmin/templates/lodige/imgs/logo.svg",
             width=140, use_container_width=False)
    st.markdown("---")
    st.markdown("### 📂 Load Data")
    uploaded = st.file_uploader(
        "Upload leads_cleaned.csv or .xlsx",
        type=["csv","xlsx"],
        help="Upload your cleaned leads file"
    )
    st.markdown("---")

    if uploaded:
        df_raw = load_data(uploaded)
        if df_raw.empty:
            st.stop()

        # Year filter
        years = sorted(df_raw["Year"].dropna().unique().tolist())
        year_options = ["All Years"] + [str(int(y)) for y in years]
        selected_year = st.radio("📅 Year", year_options, index=0)

        # Seniority filter
        all_seniority = sorted(df_raw["Seniority"].dropna().unique().tolist())
        selected_seniority = st.multiselect(
            "👤 Seniority",
            all_seniority,
            default=all_seniority,
        )

        # Country filter
        all_countries = sorted(df_raw["Country"].dropna().unique().tolist())
        selected_countries = st.multiselect(
            "🌍 Country",
            all_countries,
            default=all_countries,
        )

        st.markdown("---")
        st.markdown(f"<small style='color:rgba(255,255,255,0.5)'>GlobalData Lead Pipeline<br>Internal Use Only</small>",
                    unsafe_allow_html=True)
    else:
        st.info("Upload a file to begin.")
        st.stop()


# ── Apply filters ─────────────────────────────────────────────
df = df_raw.copy()
if selected_year != "All Years":
    df = df[df["Year"] == int(selected_year)]
if selected_seniority:
    df = df[df["Seniority"].isin(selected_seniority)]
if selected_countries:
    df = df[df["Country"].isin(selected_countries)]

# ── Unique leads (dedup by Lead + Email) ─────────────────────
# df       = all records including repeat visits (used for Total Records KPI only)
# df_unique = one row per person (used for all charts, analysis and table)
_dedup_cols = [c for c in ["Lead", "Email"] if c in df.columns]
df_unique = df.drop_duplicates(subset=_dedup_cols).copy() if _dedup_cols else df.copy()


# ── Header ────────────────────────────────────────────────────
year_label = selected_year if selected_year != "All Years" else "2025 & 2026"
st.markdown(f"""
<div class="dash-header">
  <div>
    <h1>📊 Lead Pipeline Dashboard</h1>
    <p>GlobalData · {year_label} · Internal Review</p>
  </div>
  <div class="dash-badge">{len(df):,} records in view</div>
</div>
""", unsafe_allow_html=True)


# ── KPI Row ───────────────────────────────────────────────────
total       = len(df)                          # all records incl. repeat visits
unique_n    = len(df_unique)                   # one row per person
repeat_n    = total - unique_n                 # repeat visit records
dm_count    = df_unique["Is_DM"].sum()         # DMs from unique leads only
dm_pct      = round(dm_count / unique_n * 100) if unique_n else 0
countries_n = df_unique["Country"].nunique()
companies_n = df_unique["Company"].nunique() if "Company" in df_unique.columns else "—"

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-val">{total:,}</div>
      <div class="kpi-label">Total Records</div>
      <div class="kpi-sub">All visits incl. repeats</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""<div class="kpi-card kpi-mid">
      <div class="kpi-val" style="color:{LODIGE_LIGHT}">{unique_n:,}</div>
      <div class="kpi-label">Unique Leads</div>
      <div class="kpi-sub">{repeat_n} repeat visit record{"s" if repeat_n != 1 else ""}</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-val">{dm_count}</div>
      <div class="kpi-label">Decision Makers</div>
      <div class="kpi-sub">{dm_pct}% of unique leads</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""<div class="kpi-card kpi-teal">
      <div class="kpi-val" style="color:{LODIGE_ACCENT}">{countries_n}</div>
      <div class="kpi-label">Countries</div>
      <div class="kpi-sub">{companies_n} companies</div>
    </div>""", unsafe_allow_html=True)

with k5:
    y2025 = len(df_unique[df_unique["Year"] == 2025])
    st.markdown(f"""<div class="kpi-card kpi-mid">
      <div class="kpi-val" style="color:{LODIGE_LIGHT}">{y2025}</div>
      <div class="kpi-label">2025 Unique</div>
      <div class="kpi-sub">Distinct contacts</div>
    </div>""", unsafe_allow_html=True)

with k6:
    y2026 = len(df_unique[df_unique["Year"] == 2026])
    st.markdown(f"""<div class="kpi-card kpi-teal">
      <div class="kpi-val" style="color:{LODIGE_ACCENT}">{y2026}</div>
      <div class="kpi-label">2026 Unique</div>
      <div class="kpi-sub">Distinct contacts</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Row 2: Monthly Trend + Seniority ─────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown('<div class="section-title">Monthly Lead Volume</div>', unsafe_allow_html=True)

    if "YearMonth" in df_unique.columns:
        monthly = (
            df_unique.groupby(["YearMonth","Year"])
            .size().reset_index(name="Count")
            .sort_values("YearMonth")
        )
        monthly["Year"] = monthly["Year"].astype(str)

        fig_trend = px.bar(
            monthly, x="YearMonth", y="Count", color="Year",
            color_discrete_map={"2025": LODIGE_BLUE, "2026": LODIGE_ACCENT},
            labels={"YearMonth":"Month","Count":"Leads","Year":"Year"},
            barmode="group",
        )
        fig_trend.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font_family="Segoe UI",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=10, b=10, l=0, r=0),
            xaxis=dict(showgrid=False, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor="#EEF2F8", tickfont=dict(size=10)),
            height=280,
        )
        fig_trend.update_traces(marker_line_width=0)
        st.plotly_chart(fig_trend, use_container_width=True)


with col_right:
    st.markdown('<div class="section-title">Seniority Breakdown</div>', unsafe_allow_html=True)

    sen = df_unique["Seniority"].value_counts().reset_index()
    sen.columns = ["Seniority","Count"]

    fig_sen = px.bar(
        sen, x="Count", y="Seniority", orientation="h",
        color="Seniority",
        color_discrete_sequence=CHART_COLORS,
    )
    fig_sen.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Segoe UI",
        showlegend=False,
        margin=dict(t=10, b=10, l=0, r=0),
        xaxis=dict(showgrid=True, gridcolor="#EEF2F8", tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, tickfont=dict(size=10), categoryorder="total ascending"),
        height=280,
    )
    fig_sen.update_traces(marker_line_width=0)
    st.plotly_chart(fig_sen, use_container_width=True)


# ── Row 3: Geography + Company ────────────────────────────────
col_geo, col_comp = st.columns(2)

with col_geo:
    st.markdown('<div class="section-title">Geographic Distribution</div>', unsafe_allow_html=True)

    geo = df_unique["Country"].value_counts().reset_index()
    geo.columns = ["Country","Count"]

    fig_geo = px.bar(
        geo.head(10), x="Count", y="Country", orientation="h",
        color="Count",
        color_continuous_scale=[[0, LODIGE_SILVER],[1, LODIGE_BLUE]],
    )
    fig_geo.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Segoe UI",
        coloraxis_showscale=False,
        margin=dict(t=10, b=10, l=0, r=0),
        xaxis=dict(showgrid=True, gridcolor="#EEF2F8", tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, tickfont=dict(size=10), categoryorder="total ascending"),
        height=300,
    )
    fig_geo.update_traces(marker_line_width=0)
    st.plotly_chart(fig_geo, use_container_width=True)


with col_comp:
    st.markdown('<div class="section-title">Top Companies</div>', unsafe_allow_html=True)

    if "Company" in df_unique.columns:
        comp = df_unique["Company"].value_counts().reset_index()
        comp.columns = ["Company","Count"]

        fig_comp = px.pie(
            comp.head(8), names="Company", values="Count",
            color_discrete_sequence=CHART_COLORS,
            hole=0.55,
        )
        fig_comp.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font_family="Segoe UI",
            legend=dict(font=dict(size=10), orientation="v"),
            margin=dict(t=10, b=10, l=0, r=10),
            height=300,
        )
        fig_comp.update_traces(textfont_size=10, textposition="outside")
        st.plotly_chart(fig_comp, use_container_width=True)


# ── Row 4: Lead Table ─────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">Lead Records</div>', unsafe_allow_html=True)

# Search
search = st.text_input(
    "🔍 Search by name, company, country or job title",
    placeholder="e.g. Director, United States, Lodige...",
    label_visibility="collapsed",
)

# Build display table — unique leads only
display_cols = [c for c in ["Lead","Job Title","Seniority","Company","Country","Date","Year"]
                if c in df_unique.columns]
df_display = df_unique[display_cols].copy()

if "Date" in df_display.columns:
    df_display["Date"] = df_display["Date"].dt.strftime("%Y-%m-%d")

# Apply search
if search:
    mask = df_display.apply(
        lambda col: col.astype(str).str.contains(search, case=False, na=False)
    ).any(axis=1)
    df_display = df_display[mask]

st.markdown(
    f"<small style='color:#6B8CB5'>Showing <b>{len(df_display):,}</b> unique leads "
    f"({total - unique_n} repeat visit records excluded from table)</small>",
    unsafe_allow_html=True
)

st.dataframe(
    df_display.sort_values("Date", ascending=False) if "Date" in df_display.columns else df_display,
    use_container_width=True,
    height=380,
    hide_index=True,
    column_config={
        "Lead":      st.column_config.TextColumn("Name", width="medium"),
        "Job Title": st.column_config.TextColumn("Job Title", width="medium"),
        "Seniority": st.column_config.TextColumn("Seniority", width="small"),
        "Company":   st.column_config.TextColumn("Company", width="medium"),
        "Country":   st.column_config.TextColumn("Country", width="small"),
        "Date":      st.column_config.TextColumn("Date", width="small"),
        "Year":      st.column_config.TextColumn("Year", width="small"),
    }
)

# Download button
csv_out = df_display.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ Download filtered data as CSV",
    data=csv_out,
    file_name=f"leads_filtered_{selected_year.replace(' ','_')}.csv",
    mime="text/csv",
)
