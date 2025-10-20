# streamlit_app.py
# GlobalData — Air Cargo radar with contextual "Quick relevancy slices"
# ---------------------------------------------------------------
# - File uploader (CSV/XLSX)
# - Air Cargo filter + optional extra keyword
# - Region/Country filter: Europe + Asia + six Arab countries
# - Stage/value filters; de-duplication by Ultimate_ProjectId
# - Keyword highlighter (Match_Fields / Match_Excerpt)
# - Lödige relevancy scoring + top-N export
# - Notes (import/export) in the table tab
# - Contextual "Quick relevancy slices" per tab via quick_slices(df_src, ...)
# ---------------------------------------------------------------

import io
import re
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="GlobalData – Air Cargo Projects Radar (Contextual Slices)", layout="wide")

# ---------------------------- Constants ----------------------------
ARAB_COUNTRIES = ["Saudi Arabia", "Oman", "United Arab Emirates", "Kuwait", "Qatar", "Bahrain"]
ASIA_REGION_ALIASES = {"Asia", "Asia-Pacific"}
EUROPE_REGION_ALIASES = {"Europe"}
MEA_REGION_ALIASES = {"Middle East", "Middle East and Africa", "MEA"}

SECTOR_COLUMNS = [
    "Primary_Sector","Primary_Sector_Level_1","Primary_Sector_Level_2","Secondary_Sector",
    "Project_Name","Project_Overview","Project_Scope","Project_Attributes","Project_Background"
]

STAGE_ORDER = [
    "Announced","Study","Planning","Pre-Design","Design","Pre-Tender","Tender","EPC Award",
    "Execution","Renovation","On Hold","Completed","Canceled","Parent","Sub","Sub-Sub"
]
STAGE_RANK = {s:i for i,s in enumerate(STAGE_ORDER)}

# ---- Lödige relevancy keywords ----
LODIGE_KEYWORDS = {
    "ULD handling": [
        r"\bULD\b", r"\bunit load device", r"pallet(?:\s|-)and(?:\s|-)container",
        r"pallet(?:\s|-)mover", r"slave pallet", r"workstation", r"powered roller deck",
        r"ball mat", r"castor deck", r"dolly docks?"
    ],
    "ETV / Stacker Crane / ASRS": [
        r"\betv\b", r"elevating transfer vehicle", r"stacker crane",
        r"\bASRS\b", r"automated storage(?:\s|-)and(?:\s|-)retrieval", r"high bay storage",
        r"racking system", r"storage system"
    ],
    "Conveyors & Sortation": [
        r"conveyor", r"sorter", r"diverter", r"tilt tray", r"merge", r"accumulation"
    ],
    "Automation & Control": [
        r"inventory control system", r"\bics\b", r"scada", r"\bplc\b", r"warehouse control system",
        r"\bwcs\b", r"warehouse management system", r"\bwms\b"
    ],
    "Vehicles & Robotics": [
        r"\bagv\b", r"automated guided vehicle", r"autonomous vehicle", r"transfer vehicle"
    ],
    "Air cargo terminal scope": [
        r"cargo terminal", r"air cargo (?:hub|terminal|village)", r"cool chain", r"\bpharma\b",
        r"build(?:ing)? (?:unit|uld) storage", r"freight terminal"
    ]
}

LODIGE_STAGE_WEIGHT = {
    "Execution": 1.0, "EPC Award": 0.95, "Tender": 0.9, "Pre-Tender": 0.85,
    "Design": 0.8, "Pre-Design": 0.75, "Planning": 0.7, "Study": 0.6,
    "Announced": 0.55, "Renovation": 0.8, "On Hold": 0.2, "Completed": 0.1, "Canceled": 0.0
}

# ---------------------------- Helpers ----------------------------
def load_df(uploaded):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
    return pd.read_csv(uploaded)

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace("\n"," ").replace("  "," ") for c in df.columns]
    return df

def ensure_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["Last_Update","Announce_Date","Start_Date","End_Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "Project_Value_USDm" in df.columns:
        df["Project_Value_USDm"] = pd.to_numeric(df["Project_Value_USDm"], errors="coerce")
    for qc, out in [
        ("Project_Announcement_Quarter","Announcement_Year"),
        ("Construction_Start_Quarter","Start_Year"),
        ("Project_End_Quarter","End_Year"),
    ]:
        if qc in df.columns:
            df[out] = df[qc].astype(str).str.extract(r"(\d{4})", expand=False).astype("Int64")
    for c in ["Ultimate_ProjectId","Id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["Region","Country","Project_Stage","Primary_Sector","Primary_Sector_Level_1",
              "Primary_Sector_Level_2","Secondary_Sector"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def find_matches_and_excerpt(row: pd.Series, base_pat, extra_pat=None, excerpt_len=120):
    matched = []
    excerpt = ""
    for c in SECTOR_COLUMNS:
        if c not in row.index:
            continue
        text = str(row[c] if pd.notna(row[c]) else "")
        if base_pat.search(text) and (extra_pat.search(text) if extra_pat else True):
            matched.append(c)
            if not excerpt:
                m = (extra_pat.search(text) if extra_pat else base_pat.search(text))
                if not m:
                    m = base_pat.search(text)
                if m:
                    s = max(0, m.start()-40); e = min(len(text), m.end()+80)
                    excerpt = text[s:e].replace("\n"," ")
    return ", ".join(matched), excerpt

def air_cargo_mask_and_marks(df: pd.DataFrame, keywords: str = ""):
    df = df.copy()
    base_pat = re.compile(r"air\s*cargo", re.I)
    extra_pat = re.compile(re.escape(keywords), re.I) if keywords.strip() else None

    sector_hit = pd.Series(False, index=df.index)
    for c in [col for col in SECTOR_COLUMNS if col in df.columns]:
        sector_hit = sector_hit | df[c].astype(str).str.contains(base_pat, na=False)

    mask = sector_hit
    if extra_pat:
        kw_hit = pd.Series(False, index=df.index)
        for c in [col for col in SECTOR_COLUMNS if col in df.columns]:
            kw_hit = kw_hit | df[c].astype(str).str.contains(extra_pat, na=False)
        mask = mask & kw_hit

    match_cols, excerpts = [], []
    for idx, row in df.iterrows():
        if mask.loc[idx]:
            mcols, ex = find_matches_and_excerpt(row, base_pat, extra_pat)
            match_cols.append(mcols); excerpts.append(ex)
        else:
            match_cols.append(""); excerpts.append("")
    df["Match_Fields"] = match_cols
    df["Match_Excerpt"] = excerpts
    return mask, df

def region_country_mask(df: pd.DataFrame, regions_sel, countries_sel):
    rcol = "Region" if "Region" in df.columns else None
    ccol = "Country" if "Country" in df.columns else None
    mask = pd.Series(True, index=df.index)
    if rcol:
        mask = mask & df[rcol].isin(set(regions_sel))
    if ccol and countries_sel:
        mask = mask | df[ccol].isin(countries_sel)
    return mask

def dedupe_by_project(df: pd.DataFrame):
    if "Ultimate_ProjectId" not in df.columns:
        return df
    d = df.copy()
    d["_stage_rank"] = d.get("Project_Stage", pd.Series([""]*len(d))).map(lambda x: STAGE_RANK.get(str(x), -1))
    if "Last_Update" not in d.columns: d["Last_Update"] = pd.NaT
    if "Project_Value_USDm" not in d.columns: d["Project_Value_USDm"] = np.nan
    d = d.sort_values(by=["Ultimate_ProjectId","_stage_rank","Last_Update","Project_Value_USDm"],
                      ascending=[True, False, False, False])
    d = d.drop_duplicates(subset=["Ultimate_ProjectId"], keep="first")
    return d.drop(columns=["_stage_rank"], errors="ignore")

def safe_groupby(df, by, metrics):
    present = [c for c in by if c in df.columns]
    if not present:
        return pd.DataFrame()
    agg = df.groupby(present, dropna=False).agg(metrics).reset_index()
    cnt_like = [k for k,v in metrics.items() if v in ("count","nunique")]
    if cnt_like:
        agg = agg.sort_values(by=cnt_like, ascending=False)
    return agg

def make_download(df: pd.DataFrame, filename: str, label="⬇️ Download CSV"):
    if df is None or df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=filename, mime="text/csv")

def parse_ids(raw: str):
    if not raw.strip():
        return set()
    tokens = re.split(r"[\s,;]+", raw.strip())
    ids = set()
    for t in tokens:
        try:
            ids.add(int(float(t)))
        except:
            pass
    return ids

# ---- Lödige scoring helpers ----
def lodige_relevancy_block(text: str):
    if not isinstance(text, str) or not text:
        return {}, 0, ""
    cat_hits = {}
    total = 0
    first_excerpt = ""
    lower = text.lower()
    for cat, patterns in LODIGE_KEYWORDS.items():
        count_cat = 0
        for pat in patterns:
            m = re.search(pat, lower, flags=re.I)
            if m:
                count_cat += 1; total += 1
                if not first_excerpt:
                    s = max(0, m.start()-50); e = min(len(text), m.end()+80)
                    first_excerpt = text[s:e].replace("\n", " ")
        if count_cat:
            cat_hits[cat] = count_cat
    return cat_hits, total, first_excerpt

def lodige_score_row(row: pd.Series):
    text_cols = [c for c in [
        "Project_Attributes","Project_Scope","Project_Overview","Project_Background",
        "Project_Name","Secondary_Sector","Primary_Sector_Level_2","Primary_Sector"
    ] if c in row.index]
    agg_hits = {}; raw_hits = 0; excerpt = ""; fields = []
    for c in text_cols:
        cat_hits, hits, ex = lodige_relevancy_block(str(row.get(c, "")))
        if hits:
            fields.append(c); raw_hits += hits
            for k,v in cat_hits.items():
                agg_hits[k] = agg_hits.get(k, 0) + v
            if not excerpt and ex:
                excerpt = ex
    cat_count = len(agg_hits)
    base = min(100, 18*cat_count + 6*np.sqrt(max(raw_hits-1, 0)))
    stage = str(row.get("Project_Stage", "")).strip()
    stage_w = LODIGE_STAGE_WEIGHT.get(stage, 0.6)
    air_bonus = 8 if "air cargo" in (
        (str(row.get("Primary_Sector","")) + " " +
         str(row.get("Primary_Sector_Level_2","")) + " " +
         str(row.get("Secondary_Sector",""))).lower()
    ) else 0
    score = float(np.clip(base * stage_w + air_bonus, 0, 100))
    return round(score, 1), "; ".join(sorted(agg_hits.keys())), ", ".join(fields), excerpt

def rank_for_preview(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["_stage_rank"] = d.get("Project_Stage", pd.Series([""]*len(d))).map(lambda x: STAGE_RANK.get(str(x), -1))
    if "Lodige_Score" not in d.columns: d["Lodige_Score"] = np.nan
    if "Last_Update" not in d.columns: d["Last_Update"] = pd.NaT
    if "Project_Value_USDm" not in d.columns: d["Project_Value_USDm"] = np.nan
    d = d.sort_values(
        by=["Lodige_Score","_stage_rank","Last_Update","Project_Value_USDm"],
        ascending=[False, True, False, False]
    )
    return d.drop(columns=["_stage_rank"], errors="ignore")

def download_xlsx(df: pd.DataFrame, filename: str, label="⬇️ Download Excel"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="TopN")
    st.download_button(label, buf.getvalue(), file_name=filename,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---- Contextual Quick Slices ----
def quick_slices(df_src: pd.DataFrame, title="Quick relevancy slices", height=250):
    st.markdown("---")
    st.markdown(f"### {title}")
    c1, c2, c3 = st.columns(3)
    with c1:
        if {"Region","Country","Project_Stage","Ultimate_ProjectId"}.issubset(df_src.columns):
            gb = (df_src.groupby(["Region","Country","Project_Stage"])
                        ["Ultimate_ProjectId"].nunique()
                        .reset_index()
                        .sort_values("Ultimate_ProjectId", ascending=False))
            st.dataframe(gb, use_container_width=True, height=height)
    with c2:
        if {"Region","Country","Project_Value_USDm"}.issubset(df_src.columns):
            gb2 = (df_src.groupby(["Region","Country"])["Project_Value_USDm"]
                      .sum().reset_index()
                      .sort_values("Project_Value_USDm", ascending=False))
            st.dataframe(gb2, use_container_width=True, height=height)
    with c3:
        if {"Latest_Momentum_Score","Project_Stage"}.issubset(df_src.columns):
            gb3 = (df_src.groupby("Project_Stage")["Latest_Momentum_Score"]
                      .mean().reset_index()
                      .sort_values("Latest_Momentum_Score", ascending=False))
            st.dataframe(gb3, use_container_width=True, height=height)

# ---------------------------- Sidebar ----------------------------
st.sidebar.title("Controls")

uploaded = st.sidebar.file_uploader("Upload GlobalData export (CSV/XLSX)", type=["csv","xlsx","xls"])

regions_sel = st.sidebar.multiselect(
    "Regions to include",
    default=sorted(EUROPE_REGION_ALIASES | ASIA_REGION_ALIASES),
    options=sorted(EUROPE_REGION_ALIASES | ASIA_REGION_ALIASES | MEA_REGION_ALIASES),
    help="Europe & Asia by default. MEA Arab countries are force-included via the list below."
)

countries_sel = st.sidebar.multiselect(
    "Always include these Arab countries:",
    options=ARAB_COUNTRIES,
    default=ARAB_COUNTRIES
)

keywords = st.sidebar.text_input("Extra keyword (optional)", value="", help="Narrow further (e.g., logistics, terminal).")

min_value, max_value = st.sidebar.slider("Project value (US$ m)", 0, 100000, (0, 100000), step=50)

stage_options = STAGE_ORDER
stage_filter = st.sidebar.multiselect("Project stages to include (empty = all)", options=stage_options)

on_radar_ids_text = st.sidebar.text_area(
    "Paste 'on radar' Ultimate_ProjectId list",
    value="",
    help="Comma/space/newline separated"
)

# Notes persistence
st.sidebar.markdown("---")
st.sidebar.caption("Notes for projects")
notes_upload = st.sidebar.file_uploader("Import notes CSV (Ultimate_ProjectId,Notes)", type=["csv"], key="notes_upload")
if "notes_map" not in st.session_state:
    st.session_state.notes_map = {}

# ---------------------------- Load & Filter ----------------------------
df = load_df(uploaded)
if df is None:
    st.info("Upload your GlobalData file to begin.")
    st.stop()

df = ensure_dtypes(normalize_cols(df))

# Air Cargo + keyword highlighting
mask_air, df_marks = air_cargo_mask_and_marks(df, keywords=keywords)

# Region/Country filter
mask_geo = region_country_mask(df_marks, regions_sel=regions_sel, countries_sel=countries_sel)

df_f = df_marks[mask_air & mask_geo].copy()

# Stage filter
if stage_filter and "Project_Stage" in df_f.columns:
    df_f = df_f[df_f["Project_Stage"].isin(stage_filter)]

# Value filter
if "Project_Value_USDm" in df_f.columns:
    df_f = df_f[(df_f["Project_Value_USDm"].fillna(0) >= min_value) &
                (df_f["Project_Value_USDm"].fillna(0) <= max_value)]

# De-duplicate by Ultimate_ProjectId
dedupe_enabled = st.toggle("De-duplicate by Ultimate_ProjectId", value=True,
                           help="Keeps best row per project by stage→last update→value.")
if dedupe_enabled:
    df_f = dedupe_by_project(df_f)

# On-Radar flag
radar_ids = parse_ids(on_radar_ids_text)
if "Ultimate_ProjectId" in df_f.columns:
    df_f["On_Radar"] = df_f["Ultimate_ProjectId"].astype("Int64").isin(radar_ids)
else:
    df_f["On_Radar"] = False

# KPIs
left, mid, right = st.columns(3)
with left:
    st.metric("Projects (filtered)", f"{len(df_f):,}")
with mid:
    total_val = df_f["Project_Value_USDm"].sum() if "Project_Value_USDm" in df_f.columns else np.nan
    st.metric("Total Value (US$ m)", f"{total_val:,.0f}" if pd.notna(total_val) else "—")
with right:
    st.metric("On-Radar (count)", f"{int(df_f['On_Radar'].sum())}")

st.caption("Matches columns are visible in **Match_Fields**; quick snippet in **Match_Excerpt**.")

# ---------------------------- Tabs ----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Group-by Explorer",
    "Map",
    "Time & Stages",
    "Owners / Funding",
    "Table, Notes & Export",
    "Match Details",
    "Lödige Relevancy"
])

# ---- Tab 1: Group-by Explorer ----
with tab1:
    st.subheader("Preset aggregations")
    presets = {
        "Region → Country → Stage (count)": (["Region","Country","Project_Stage"], {"Ultimate_ProjectId":"nunique"}),
        "Region → Country (projects & value)": (["Region","Country"], {"Ultimate_ProjectId":"nunique","Project_Value_USDm":"sum"}),
        "Country → City (count)": (["Country","City"], {"Ultimate_ProjectId":"nunique"}),
        "Stage by Region (count)": (["Project_Stage","Region"], {"Ultimate_ProjectId":"nunique"}),
    }
    choice = st.selectbox("Aggregation", options=list(presets.keys()))
    by, metrics = presets[choice]
    agg = safe_groupby(df_f, by, metrics)
    st.dataframe(agg, use_container_width=True)
    make_download(agg, f"agg_{re.sub(r'[^A-Za-z0-9]+','_',choice)}.csv")

    st.markdown("#### Build your own")
    by_custom = st.multiselect("Group by columns", options=list(df_f.columns), default=["Region","Country"])
    metric_target = st.selectbox("Metric column", options=[c for c in df_f.columns if c != "On_Radar"], index=0)
    metric_fn = st.selectbox("Aggregation", options=["count","nunique","sum","mean","median","min","max"], index=1)
    custom = safe_groupby(df_f, by_custom, {metric_target: metric_fn})
    st.dataframe(custom, use_container_width=True)
    make_download(custom, "custom_groupby.csv")

    # Contextual slices for Tab 1
    quick_slices(df_f, title="Quick relevancy slices (current filters)")

# ---- Tab 2: Map ----
with tab2:
    st.subheader("Project Map")
    map_df = df_f.copy()
    if {"Latitude","Longitude"}.issubset(map_df.columns):
        map_df = map_df.dropna(subset=["Latitude","Longitude"])
        tip_cols = [c for c in ["Ultimate_ProjectId","Project_Name","Country","City","Project_Stage","Project_Value_USDm","On_Radar"] if c in map_df.columns]
        st.map(map_df.rename(columns={"Latitude":"lat","Longitude":"lon"})[["lat","lon"] + tip_cols])
        st.caption("Tip: hover pins to see details.")
    else:
        st.info("Latitude/Longitude not found in file.")

# ---- Tab 3: Time & Stages ----
# ---- Tab 3: Time & Stages ----
with tab3:
    st.subheader("Projects over time — by announcement, start, and completion")

    # ---- controls for this tab
    metric_choice = st.radio(
        "Metric", ["Projects (count)", "Total Value (US$ m)"],
        horizontal=True, index=0
    )
    stack_by = st.selectbox(
        "Stack by (optional)",
        ["None", "Project_Stage", "Region"],
        index=1 if "Project_Stage" in df_f.columns else 0,
        help="Stack bars to see distribution by Stage or Region."
    )

    # Helper: build a tidy per-year dataframe for the selected metric
    def _year_pivot(df, year_col, stack_dim=None):
        d = df.copy()
        if year_col not in d.columns:
            return pd.DataFrame(columns=[year_col, "Metric"] + ([stack_dim] if stack_dim else []))
        d = d.dropna(subset=[year_col])
        d[year_col] = d[year_col].astype("Int64")
        if d.empty:
            return d[[year_col]]

        if metric_choice.startswith("Projects"):
            if stack_dim and stack_dim in d.columns:
                g = d.groupby([year_col, stack_dim])["Ultimate_ProjectId"].nunique().reset_index(name="Metric")
            else:
                g = d.groupby([year_col])["Ultimate_ProjectId"].nunique().reset_index(name="Metric")
        else:
            # Total Value
            if "Project_Value_USDm" not in d.columns:
                d["Project_Value_USDm"] = np.nan
            if stack_dim and stack_dim in d.columns:
                g = d.groupby([year_col, stack_dim])["Project_Value_USDm"].sum().reset_index(name="Metric")
            else:
                g = d.groupby([year_col])["Project_Value_USDm"].sum().reset_index(name="Metric")
        return g.sort_values(year_col)

    # Build three datasets
    stack_dim = None if stack_by == "None" else stack_by
    ann_df = _year_pivot(df_f, "Announcement_Year", stack_dim)
    start_df = _year_pivot(df_f, "Start_Year", stack_dim)
    end_df = _year_pivot(df_f, "End_Year", stack_dim)

    # Helper: chart builder
    def _year_chart(tdf, year_col, title):
        if tdf is None or tdf.empty or year_col not in tdf.columns:
            st.info(f"No data for {title.lower()}.")
            return
        if stack_dim and stack_dim in tdf.columns:
            ch = (
                alt.Chart(tdf)
                .mark_bar()
                .encode(
                    x=alt.X(f"{year_col}:O", title="Year"),
                    y=alt.Y("Metric:Q", title=metric_choice),
                    color=alt.Color(f"{stack_dim}:N", title=stack_dim),
                    tooltip=list(tdf.columns)
                )
                .properties(height=300, title=title)
            )
        else:
            ch = (
                alt.Chart(tdf)
                .mark_bar()
                .encode(
                    x=alt.X(f"{year_col}:O", title="Year"),
                    y=alt.Y("Metric:Q", title=metric_choice),
                    tooltip=list(tdf.columns)
                )
                .properties(height=300, title=title)
            )
        st.altair_chart(ch, use_container_width=True)

    # Layout: three columns on wide screens, two rows on narrow automatically
    c1, c2, c3 = st.columns(3)
    with c1:
        _year_chart(ann_df, "Announcement_Year", "Announcement Year (pipeline creation)")
    with c2:
        _year_chart(start_df, "Start_Year", "Construction Start Year (execution ramp)")
    with c3:
        _year_chart(end_df, "End_Year", "Completion / End Year (deliveries)")

    st.caption(
        "Tip: **Announcement** reflects pipeline creation; **Start** reflects near-term execution; "
        "**End** approximates delivery timing. Use the **Stack by** control to see distribution by Stage or Region, "
        "and switch the **Metric** to total value for commercial sizing."
    )

# ---- Tab 4: Owners / Funding ----
with tab4:
    st.subheader("Owners, Funding, Contractors")
    cols = st.columns(3)
    with cols[0]:
        if "Project_Owner" in df_f.columns:
            own = df_f.groupby("Project_Owner")["Ultimate_ProjectId"].nunique().reset_index().sort_values("Ultimate_ProjectId", ascending=False).head(20)
            st.write("Top Owners (by # projects)")
            st.dataframe(own, use_container_width=True)
    with cols[1]:
        if {"Funding_Status","Funding_Mode"}.issubset(df_f.columns):
            fund = df_f.groupby(["Funding_Status","Funding_Mode"])["Ultimate_ProjectId"].nunique().reset_index().sort_values("Ultimate_ProjectId", ascending=False)
            st.write("Funding (status × mode)")
            st.dataframe(fund, use_container_width=True)
    with cols[2]:
        if "Main_Contractor" in df_f.columns:
            mc = df_f.groupby("Main_Contractor")["Ultimate_ProjectId"].nunique().reset_index().sort_values("Ultimate_ProjectId", ascending=False).head(20)
            st.write("Top Main Contractors")
            st.dataframe(mc, use_container_width=True)

# ---- Tab 5: Table, Notes & Export ----
with tab5:
    st.subheader("Filtered projects")

    # Merge imported notes into session map
    if notes_upload is not None:
        try:
            ndf = pd.read_csv(notes_upload)
            for _, r in ndf.iterrows():
                pid = r.get("Ultimate_ProjectId")
                note = r.get("Notes","")
                if pd.notna(pid):
                    st.session_state.notes_map[int(pid)] = str(note)
            st.success(f"Imported {len(ndf)} notes.")
        except Exception as e:
            st.error(f"Failed to import notes: {e}")

    # Attach Notes column from session
    if "Ultimate_ProjectId" in df_f.columns:
        df_f["Notes"] = df_f["Ultimate_ProjectId"].map(lambda x: st.session_state.notes_map.get(int(x) if pd.notna(x) else x, ""))

    base_default_cols = [c for c in ["Ultimate_ProjectId","Project_Name","Region","Country","City","Project_Stage",
                                     "Project_Value_USDm","Funding_Status","Funding_Mode","Project_Url","On_Radar",
                                     "Match_Fields","Match_Excerpt","Notes"] if c in df_f.columns]
    show_cols = st.multiselect("Columns to display", options=list(df_f.columns), default=base_default_cols)

    # Editable table with notes
    edit_df = st.data_editor(df_f[show_cols], use_container_width=True, height=520, key="edit_table")

    # Persist edited notes back into session
    if "Ultimate_ProjectId" in edit_df.columns and "Notes" in edit_df.columns:
        for _, r in edit_df[["Ultimate_ProjectId","Notes"]].iterrows():
            pid = r["Ultimate_ProjectId"]; note = r["Notes"]
            if pd.notna(pid):
                st.session_state.notes_map[int(pid)] = str(note) if pd.notna(note) else ""

    colA, colB = st.columns(2)
    with colA:
        make_download(edit_df, "filtered_air_cargo_projects.csv", label="⬇️ Download filtered table (with notes)")
    with colB:
        if "Ultimate_ProjectId" in edit_df.columns:
            notes_only = edit_df[["Ultimate_ProjectId","Notes"]].copy()
            notes_only = notes_only[notes_only["Notes"].astype(str).str.len() > 0]
            make_download(notes_only, "project_notes.csv", label="⬇️ Download notes only")

# ---- Tab 6: Match Details ----
with tab6:
    st.subheader("Why each project matched ‘Air Cargo’ / keyword")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Matched Fields**")
        st.dataframe(df_f[["Ultimate_ProjectId","Project_Name","Match_Fields"]].sort_values("Match_Fields"), use_container_width=True, height=360)
    with cols[1]:
        st.markdown("**Match Excerpts**")
        st.dataframe(df_f[["Ultimate_ProjectId","Project_Name","Match_Excerpt"]], use_container_width=True, height=360)

# ---- Tab 7: Lödige Relevancy ----
with tab7:
    st.subheader("Lödige Industries Relevancy")

    # Compute score if not present
    lodige_cols = ["Lodige_Score","Lodige_Categories","Lodige_Match_Fields","Lodige_Excerpt"]
    if not set(lodige_cols).issubset(df_f.columns):
        df_f[lodige_cols] = df_f.apply(lambda r: pd.Series(lodige_score_row(r)), axis=1)
        df_f["Lodige_Score"] = df_f["Lodige_Score"].astype(float).round(1)

    # Top candidates table
    topN = st.slider("Show top N candidates", 10, 200, 50, step=10)
    shortlist_cols = [c for c in [
        "Ultimate_ProjectId","Project_Name","Region","Country","City",
        "Project_Stage","Project_Value_USDm","On_Radar",
        "Lodige_Score","Lodige_Categories","Lodige_Match_Fields","Lodige_Excerpt",
        "Project_Url"
    ] if c in df_f.columns]
    top_tbl = df_f.sort_values("Lodige_Score", ascending=False).head(topN)[shortlist_cols]
    st.dataframe(top_tbl, use_container_width=True, height=420)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Score distribution**")
        if "Lodige_Score" in df_f.columns:
            d = df_f[["Lodige_Score"]].dropna()
            chart = alt.Chart(d).mark_bar().encode(
                x=alt.X("Lodige_Score:Q", bin=alt.Bin(maxbins=25), title="Score"),
                y=alt.Y("count()", title="Projects")
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)
    with c2:
        st.markdown("**Score by stage**")
        if {"Lodige_Score","Project_Stage"}.issubset(df_f.columns):
            t = df_f.groupby("Project_Stage")["Lodige_Score"].mean().reset_index()
            chart = alt.Chart(t).mark_bar().encode(
                x=alt.X("Project_Stage:N", sort="-y"),
                y=alt.Y("Lodige_Score:Q", title="Avg score")
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)

    # Category heat
    st.markdown("**Which product families are being signalled?**")
    cat_counts = (
        df_f.assign(_cats=df_f["Lodige_Categories"].fillna(""))
           .assign(_cat_list=lambda d: d["_cats"].str.split(r"\s*;\s*"))
    )
    expl = []
    for _, r in cat_counts.iterrows():
        pid = r.get("Ultimate_ProjectId")
        for c in (r.get("_cat_list") or []):
            if c:
                expl.append({"Ultimate_ProjectId": pid, "Category": c})
    if expl:
        catdf = pd.DataFrame(expl)
        heat = catdf.groupby("Category")["Ultimate_ProjectId"].nunique().reset_index(name="Projects")
        chart = alt.Chart(heat).mark_bar().encode(
            x=alt.X("Projects:Q"),
            y=alt.Y("Category:N", sort="-x")
        ).properties(height=280)
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(heat.sort_values("Projects", ascending=False), use_container_width=True, height=280)

    # Filters & export
    min_score = st.slider("Minimum Lödige score to include", 0, 100, 60, step=5)
    exp_df = df_f[df_f["Lodige_Score"] >= min_score]
    st.markdown(f"**Exportable shortlist ≥ {min_score}**")
    exp_cols = [c for c in shortlist_cols if c in exp_df.columns]
    exp = exp_df[exp_cols].copy()
    st.dataframe(exp, use_container_width=True, height=300)
    make_download(exp, f"lodige_shortlist_min{min_score}.csv", label="⬇️ Download shortlist CSV")

    # Contextual slices for Tab 7 (use the min-score filtered view)
    quick_slices(exp_df, title=f"Slices (Lödige score ≥ {min_score})")
