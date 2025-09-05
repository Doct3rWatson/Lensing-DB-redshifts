# interactive_lenses_exact.py
# Requires: pip install bokeh pandas numpy

import pandas as pd
import numpy as np
from urllib.parse import quote, unquote
from datetime import datetime
import subprocess, os
from html import unescape

from bokeh.plotting import figure, output_file, save
from bokeh.models import (
    ColumnDataSource, HoverTool, Slider, CustomJS,
    CDSView, BooleanFilter, Div, Label, Span, Range1d, Spacer,
    Legend, LegendItem, FixedTicker, InlineStyleSheet, TapTool, OpenURL

)
from bokeh.layouts import column, row
from bokeh.events import DocumentReady
from bokeh.resources import INLINE, CDN
from bokeh.embed import file_html


VERSION = "v0.1.0" 
BUILD_DATE = datetime.now().strftime("%Y-%m-%d")
COPY_YEAR = datetime.now().year

# ---- Helpers ---- 
def _svg_quasar(size=14, stroke="#000"):
    s = size
    m = s/2
    pad = 2
    return f"""
<svg width="{s}" height="{s}" viewBox="0 0 {s} {s}" style="vertical-align:middle">
  <!-- plus -->
  <line x1="{pad}" y1="{m}" x2="{s-pad}" y2="{m}" stroke="{stroke}" stroke-width="1.8"/>
  <line x1="{m}" y1="{pad}" x2="{m}" y2="{s-pad}" stroke="{stroke}" stroke-width="1.8"/>
  <!-- x -->
  <line x1="{pad}" y1="{pad}" x2="{s-pad}" y2="{s-pad}" stroke="{stroke}" stroke-width="1.8"/>
  <line x1="{s-pad}" y1="{pad}" x2="{pad}" y2="{s-pad}" stroke="{stroke}" stroke-width="1.8"/>
</svg>""".strip()

def _svg_circle(size=14, stroke="#000", fill="none"):
    s = size
    m = s/2
    r = m - 2
    return f"""
<svg width="{s}" height="{s}" viewBox="0 0 {s} {s}" style="vertical-align:middle">
  <circle cx="{m}" cy="{m}" r="{r}" stroke="{stroke}" stroke-width="1.8" fill="{fill}"/>
</svg>""".strip()

def _svg_star(size=14, stroke="#000", fill="#000"):
    # Simple 5-point star polygon centered in the box
    s=size
    pts="7,1 9,5 13,5 10,7.8 11.5,12 7,9.7 2.5,12 4,7.8 1,5 5,5"
    return f"""
<svg width="{s}" height="{s}" viewBox="0 0 14 14" style="vertical-align:middle">
  <polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>
</svg>""".strip()

def build_hist_legend_html(q_tot, g_tot, u_tot, q_year, g_year, u_year):
    return rf"""
<div style="font-size:1.2em; margin-bottom:4px;"><b>Histogram Legend</b></div>
<div style="display:flex; flex-direction:column; gap:6px; font-size:1em;">
  <div style="display:flex; align-items:center; gap:8px;">
    <span style="display:inline-block; width:14px; height:14px; background:{HIST_COLS['Quasar']}; border:1px solid #666;"></span>
    $$\text{{Quasar, N}} = {q_tot}$$
  </div>
  <div style="display:flex; align-items:center; gap:8px;">
    <span style="display:inline-block; width:14px; height:14px; background:{HIST_COLS['Galaxy']}; border:1px solid #666;"></span>
    $$\text{{Galaxy, N}} = {g_tot}$$
  </div>
  <div style="display:flex; align-items:center; gap:8px;">
    <span style="display:inline-block; width:14px; height:14px; background:{HIST_COLS['Unknown']}; border:1px solid #666;"></span>
    $$\text{{Unknown, N}} = {u_tot}$$
  </div>
</div>
"""

def style_ticks(fig,
                label_size="1.2em",
                major_len=10, minor_len=5,
                line_width=1.5,
                label_standoff=5):
    # Apply to all axes on the figure
    for ax in fig.xaxis + fig.yaxis:
        # label font
        ax.major_label_text_font_size = label_size
        ax.major_label_text_font='serif'
        ax.major_label_standoff = label_standoff  # gap from axis to labels

        # tick mark lengths (outside vs inside the frame)
        ax.major_tick_out = major_len
        ax.major_tick_in  = 0                    # usually keep inside at 0
        ax.minor_tick_out = minor_len
        ax.minor_tick_in  = 0

        # tick line thickness
        ax.major_tick_line_width = line_width
        ax.minor_tick_line_width = max(1, line_width - 0.5)


# ---- Input data ----
CSV_PATH = "spec_conf_lens_db.csv"

# ---- Styling / mapping to match colors and markers ----
COLOR_MAP = {
    'CLASS': "#88CCEE",
    'SLACS': "#44AA99",
    'BELLS': "#117733",
    'SL2S': "#999933",
    'CASSOWARY': '#DDCC77',
    'AGEL': "#332288",
    'Serendipitous': '#882255',
    'Others': 'gray',
}

## ---- Some constants ----
QUASAR_TAGS = {'GAL-QSO', 'GRP-QSO', 'CLUST-QSO', 'QUASAR'}
GAL_TAGS    = {'GAL-GAL', 'QSO-GAL', 'GRP-GAL', 'CLUST-GAL', 'GALAXY'}
HIST_COLS = {"Quasar": "#777777", "Galaxy": "#000000", "Unknown": "#CCCCCC"}  # grey / black / white
stack_cols = ["Quasar", "Galaxy", "Unknown"]
stack_colors = [HIST_COLS[c] for c in stack_cols]

# ---- Load / prepare ----
df = pd.read_csv(CSV_PATH)

df.loc[df['survey'].eq('AGEL'), 'kind'] = 'GALAXY'

# Ensure columns exist
need_cols = ["survey","kind","z_def","z_src","year"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in {CSV_PATH}: {missing}")

# Kind buckets to match plotting branches
def classify_kind(k):
    if pd.isna(k) or k == '' or k == 'Unknown' or k == 'XRAY-CLUST':
        return "Unknown"
    k = str(k).strip().upper()
    if k in QUASAR_TAGS:
        return "Quasar"
    return "Galaxy"

df["kind_class"] = df["kind"].apply(classify_kind)

# Survey colors (+ Others)
df["color"]  = df["survey"].map(lambda s: COLOR_MAP.get(s, COLOR_MAP["Others"]))
df["is_agel"] = (df["survey"] == "AGEL")

# Coerce numerics
df["z_def"] = pd.to_numeric(df["z_def"], errors="coerce")
df["z_src"] = pd.to_numeric(df["z_src"], errors="coerce")
df["year"]  = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["z_def","z_src","year"]).reset_index(drop=True)
df["year"] = df["year"].astype(int)

if "paper" not in df.columns:
    raise ValueError("DataFrame needs a 'paper' column to make clickable points.")
ADS_BASE = "https://ui.adsabs.harvard.edu/abs/"
def normalize_bibcode(bib):
    if pd.isna(bib):
        return None
    s = str(bib).strip()
    # Convert HTML / LaTeX forms to raw text
    s = unescape(s).replace(r"\&", "&")
    # Collapse any existing percent-encoding (handles %26, %2526, etc.)
    for _ in range(3):
        new_s = unquote(s)
        if new_s == s:
            break
        s = new_s
    return s

def bibcode_to_ads_url(bib):
    s = normalize_bibcode(bib)
    if s is None:
        return None
    # Encode as a path segment but KEEP existing % so we don't re-encode %26 -> %2526
    path = quote(s, safe="%")
    return f"{ADS_BASE}{path}/abstract"

df["ads_url"] = df["paper"].apply(bibcode_to_ads_url) #ads_base + df["paper"].astype(str).map(lambda s: quote(s, safe="")) + "/abstract"

min_year = max(1979, int(df["year"].min()))
max_year = max(2025, int(df["year"].max()))
years_all = np.arange(min_year, max_year + 1, 1)
initial_year = max_year

# ---- Precompute per-year counts and cumulative fractions (for top/middle panels) ----
# Per-year counts by kind (for the middle panel)
per_year = (
    df.groupby(["year","kind_class"])
      .size()
      .unstack(fill_value=0)
      .reindex(years_all, fill_value=0)
      .rename_axis("year")
      .reset_index()
)
# Ensure all columns exist
for col in ["Quasar","Galaxy","Unknown"]:
    if col not in per_year.columns:
        per_year[col] = 0

# Cumulative totals for top panel, then normalize to fractions
cum = per_year[["Quasar","Galaxy","Unknown"]].cumsum()
cum_total = cum.sum(axis=1).replace(0, np.nan)
frac = (cum.T / cum_total).T.fillna(0.0)  # rows are years

# Sources for histograms (top = normalized fractions, middle = raw per-year counts)
# --- arrays we will show in hover tooltips ---
q_counts = per_year["Quasar"].to_numpy()
g_counts = per_year["Galaxy"].to_numpy()
u_counts = per_year["Unknown"].to_numpy()

q_cum = np.cumsum(q_counts)
g_cum = np.cumsum(g_counts)
u_cum = np.cumsum(u_counts)

# TOP panel source: fractions + (for hover) per-year counts and cumulative totals
top_src = ColumnDataSource(dict(
    year=years_all,
    Quasar=frac["Quasar"].to_numpy(),
    Galaxy=frac["Galaxy"].to_numpy(),
    Unknown=frac["Unknown"].to_numpy(),
    Quasar_count=q_counts, Galaxy_count=g_counts, Unknown_count=u_counts,
    Quasar_cum=q_cum,     Galaxy_cum=g_cum,     Unknown_cum=u_cum,
    alpha=(years_all <= initial_year).astype(float)
))

# MIDDLE panel source: per-year counts + cumulative totals
mid_src = ColumnDataSource(dict(
    year=years_all,
    Quasar=q_counts, Galaxy=g_counts, Unknown=u_counts,
    Quasar_cum=q_cum, Galaxy_cum=g_cum, Unknown_cum=u_cum,
    alpha=(years_all <= initial_year).astype(float)
))


# ---- Main scatter source and four BooleanFilter views (match marker logic) ----
df["mask"] = (df["year"] <= initial_year)  # updated via JS

src = ColumnDataSource(df)

# Initial boolean masks (will be overwritten by JS on slider move)
bq = list((df["mask"] & (df["kind_class"]=="Quasar")).astype(bool))
bg = list((df["mask"] & (df["kind_class"]=="Galaxy") & (~df["is_agel"])).astype(bool))
bu = list((df["mask"] & (df["kind_class"]=="Unknown") & (~df["is_agel"])).astype(bool))
bau = list((df["mask"] & df["is_agel"] & (df["kind_class"]=="Unknown")).astype(bool))
ba = list((df["mask"] & df["is_agel"] & (df["kind_class"]!="Unknown")).astype(bool))

v_quasar  = CDSView(filter=BooleanFilter(bq))
v_galaxy  = CDSView(filter=BooleanFilter(bg))
v_unknown = CDSView(filter=BooleanFilter(bu))
v_agel    = CDSView(filter=BooleanFilter(ba))
v_agelu    = CDSView(filter=BooleanFilter(bau))



# ---- Figures ----
mathjax_loader = Div(text="""
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
""")
year_marker_top  = Span(location=initial_year, dimension="height",
                        line_color="black", line_dash="dashed", line_width=2)
year_marker_mid  = Span(location=initial_year, dimension="height",
                        line_color="black", line_dash="dashed", line_width=2)
year_marker_tick = Span(location=initial_year, dimension="height",
                        line_color="black", line_dash="dashed", line_width=2)
histo_heights = 140
histo_widths = 650
mainax_height = 400
mainax_width = 640

# Top panel: normalized cumulative stacked bars (N_frac_total)
p_top = figure(frame_height=histo_heights, frame_width=histo_widths, 
               tools="pan,xwheel_zoom,box_zoom,reset,save",
               toolbar_location=None)
# p_top = figure(height=histo_heights, sizing_mode="stretch_width", 
#                tools="pan,xwheel_zoom,box_zoom,reset,save", toolbar_location=None)
p_top.title.text = r"$$\text{Cumulative fractional total of source types}$$"
p_top.xaxis.axis_label = r"$$\text{Year}$$"
p_top.yaxis.axis_label = r"N$$_{\text{frac.\ total}}$$"
top_rends = p_top.vbar_stack(
    stack_cols, x="year", width=0.9, source=top_src,
    color=stack_colors, fill_alpha="alpha",
    line_color="#666666", line_alpha="alpha", line_width=1
    )
p_top.y_range = Range1d(0, 1)
p_top.add_layout(year_marker_top)
p_top.yaxis.axis_label_text_font_size = '1.7em'
p_top.xaxis.axis_label_text_font_size = '1.3em'
p_top.title.text_font_size='1em'



# Middle panel: per-year stacked counts
p_mid = figure(frame_height=histo_heights, frame_width=histo_widths, 
               tools="pan,xwheel_zoom,box_zoom,reset,save", 
               toolbar_location=None)
# p_mid = figure(height=histo_heights, sizing_mode="stretch_width", 
#                tools="pan,xwheel_zoom,box_zoom,reset,save", toolbar_location=None)
p_mid.title.text = r"N$$_{\text{systems}}\ \text{ per year (stacked by source kind)}$$"
p_mid.xaxis.axis_label = r"$$\text{Year}$$"
p_mid.yaxis.axis_label = r"N$$_{\text{systems}}$$"
mid_rends = p_mid.vbar_stack(
    stack_cols, x="year", width=0.9, source=mid_src,
    color=stack_colors, fill_alpha="alpha",
    line_color="#666666", line_alpha="alpha", line_width=1
)
p_mid.y_range.start = 0
p_mid.add_layout(year_marker_mid)
p_mid.yaxis.axis_label_text_font_size = '1.7em'
p_mid.xaxis.axis_label_text_font_size = '1.3em'
p_mid.title.text_font_size='1em'

# # Bottom panel: z_src vs z_def (exact axes limits)
p = figure(
    frame_width=mainax_width, frame_height=mainax_height,  # <- inner plotting area size
    tools= "pan,wheel_zoom,box_zoom,reset,save,hover", 
    toolbar_location="left", 
    x_range=(0.0, 1.4), y_range=(0.0, 8.5))
# p = figure(height=mainax_height, sizing_mode="stretch_width", 
#            tools="pan,wheel_zoom,box_zoom,reset,save,hover", toolbar_location="left", 
#            x_range=(0.0, 1.4), y_range=(0.0, 8.5))
p.min_border_left = 80  # make space for y-axis labels; fixed so we can align the slider
p.title.text = r"$$\text{Spectroscopically Confirmed Gravitational Lensing Systems}$$"
p.xaxis.axis_label = r"$$z_{\text{deflector}}$$"
p.yaxis.axis_label = r"$$z_{\text{source}}$$"
p.yaxis.axis_label_text_font_size = '2.5em'
p.xaxis.axis_label_text_font_size = '2.5em'
p.title.text_font_size='1.2em'
FW = int(p.frame_width)        # one source of truth for inner width
# FH = mainax_height
LEFT = int(p.min_border_left)    # one source of truth for left indent

header = Div(
    sizing_mode="stretch_width",
    text=rf"""
<div class="header">
  <span>$$\text{{Lensing systems with spectroscopically confirmed source and deflector}}$$</span>
</div>""",
    stylesheets=[InlineStyleSheet(css="""
      .header{
        font-family: serif; font-size: 2em; color:#444;
        border-bottom:1px solid #ddd; padding:10px 0; margin-top:12px;
        display:flex; gap:12px; flex-wrap:wrap; justify-content:center;
      }
      .header a { color:inherit; text-decoration:underline; }
    """)]
)

footer = Div(
    sizing_mode="stretch_width",
    text=f"""
<div class="footer">
  <span>© {COPY_YEAR} Courtney B. Watson | AGEL Team. All rights reserved.</span>
  <span>· Build {BUILD_DATE} · {VERSION}</span>
  <span>· Source: <a href="https://github.com/Doct3rWatson/Lensing-DB-redshifts" target="_blank" rel="noopener">GitHub</a></span>
  <span>· <a href="mailto:courtney.watson@cfa.harvard.edu">courtney.watson@cfa.harvard.edu</a></span>
</div>""",
    stylesheets=[InlineStyleSheet(css="""
      .footer{
        font-family: serif; font-size: 1em; color:#444;
        border-top:1px solid #ddd; padding:10px 0; margin-top:12px;
        display:flex; gap:12px; flex-wrap:wrap; justify-content:center;
      }
      .footer a { color:inherit; text-decoration:underline; }
    """)]
)

# Quasar = plus + times overlay (two renderers)
rq1 = p.scatter(x="z_def", y="z_src", source=src, view=v_quasar,
                size=12, marker="cross", line_color="color", fill_alpha=0.0, line_alpha=0.9)
rq2 = p.scatter(x="z_def", y="z_src", source=src, view=v_quasar,
                size=12, marker="x",     line_color="color", fill_alpha=0.0, line_alpha=0.9)
# Galaxy (non-AGEL): filled circles with outline
rg  = p.scatter(x="z_def", y="z_src", source=src, view=v_galaxy,
                size=10, marker="circle", fill_color="color", line_color="color",
                fill_alpha=0.7, line_alpha=0.9)
# Unknown: hollow circles
ru  = p.scatter(x="z_def", y="z_src", source=src, view=v_unknown,
                size=10, marker="circle", fill_alpha=0.0, line_color="color", line_alpha=0.9)
# AGEL: big star
ra  = p.scatter(x="z_def", y="z_src", source=src, view=v_agel,
                size=16, marker="star", fill_color="color", line_color="color",
                fill_alpha=0.7, line_alpha=0.9)
rau = p.scatter(x="z_def", y="z_src", source=src, view=v_agelu,
                size=16, marker="star", fill_color="color", line_color="color",
                fill_alpha=0.7, line_alpha=0.9)

# Hover (attach to all bottom renderers)
hover = p.select_one(HoverTool)
hover.tooltips = [
    ("Survey", "@survey"),
    ("Source type", "@kind_class"),
    ("Year", "@year"),
    ("z_def", "@z_def{0.000}"),
    ("z_src", "@z_src{0.000}")
]
hover.renderers = [rq1, rq2, rg, ru, ra, rau] 

for i, st in enumerate(stack_cols):
    p_top.add_tools(HoverTool(
        renderers=[top_rends[i]],
        tooltips=[
            ("Year", "@year{int}"),
            ("Source type", st),
            ("Fraction", f"@{st}{{0.00%}}"),
            ("This year", f"@{st}_count"),
            ("Cumulative", f"@{st}_cum"),
        ]
    ))

# --- MIDDLE: per-year counts (show counts + running totals) ---
for i, st in enumerate(stack_cols):
    p_mid.add_tools(HoverTool(
        renderers=[mid_rends[i]],
        tooltips=[
            ("Year", "@year{int}"),
            ("Source type", st),
            ("This year", f"@{st}"),
            ("Cumulative", f"@{st}_cum"),
        ]
    ))

# Dynamic text annotations: "Year Y" and "N_systems="
# N_systems_this_year = (# rows where year == slider value)
PAD = 40  # px inside the frame
label_year = Label(
    x=p.frame_width - PAD - PAD,            # right edge of the plotting area
    y=p.frame_height - PAD,           # top edge of the plotting area
    x_units="screen", y_units="screen",
    text=rf"$$\text{{Year }}{initial_year}$$",
    text_font='computer modern',
    text_font_size="1.5em",
    text_align="center",               # anchor text's right edge at x
    text_baseline="top",              # anchor text's top at y
    background_fill_color="white",    
    background_fill_alpha=0.6
)
label_nsys = Label(
    x=p.frame_width - PAD - PAD,
    y=p.frame_height - PAD - 30,      # a line below the first (adjust as needed)
    x_units="screen", y_units="screen",
    text=rf"N$$_{{\text{{systems}}}} = {len(df)}$$",
    text_font_size="1.2em",
    text_align="center",
    text_baseline="top",
    background_fill_color="white",
    background_fill_alpha=0.6
)
# label_year = Label(
#     x=p.x_range.end, y=p.y_range.end,
#     x_units="data", y_units="data",
#     x_offset=-PAD, y_offset=-PAD,    # pixel offsets inward
#     text=rf"$$\text{{Year }}{initial_year}$$",
#     text_font_size="18pt",
#     text_align="right", text_baseline="top",
#     background_fill_color="white", background_fill_alpha=0.6
# )
# label_nsys = Label(
#     x=p.x_range.end, y=p.y_range.end,
#     x_units="data", y_units="data",
#     x_offset=-PAD, y_offset=-(PAD+25),
#     text=rf"N$$_{{\text{{systems}}}} = {len(df)}$$",
#     text_font_size="14pt",
#     text_align="right", text_baseline="top",
#     background_fill_color="white", background_fill_alpha=0.6
# )

p.add_layout(label_year)
p.add_layout(label_nsys)

tap = TapTool(renderers=[rq1, rq2, rg, ru, ra, rau])
tap.callback = OpenURL(url='@ads_url', same_tab=False)  
p.add_tools(tap)
p.toolbar.active_tap = tap    # make it active by default


## ---- Build Legends ----
# Initial values at initial_year
py = per_year.set_index("year")
q_tot0 = int(py.loc[py.index <= initial_year, "Quasar"].sum())
g_tot0 = int(py.loc[py.index <= initial_year, "Galaxy"].sum())
u_tot0 = int(py.loc[py.index <= initial_year, "Unknown"].sum())
q_yr0  = int(py.loc[initial_year, "Quasar"]) if initial_year in py.index else 0
g_yr0  = int(py.loc[initial_year, "Galaxy"]) if initial_year in py.index else 0
u_yr0  = int(py.loc[initial_year, "Unknown"]) if initial_year in py.index else 0
mid_total0 = q_tot0 + g_tot0 + u_tot0


MID_PAD = 30  # px offset from the top-left
label_mid_total = Label(
    x=MID_PAD, y=p_mid.frame_height - MID_PAD + 10,
    x_units="screen", y_units="screen",
    text=rf"$$\text{{N}}_{{\text{{cumulative}}}} = {mid_total0}$$",
    text_font_size="1.2em",
    text_align="left", text_baseline="top",
    background_fill_color="white", background_fill_alpha=0.6
)
# label_mid_total = Label(
#     x=1979, y=70,
#     x_units="data", y_units="data",
#     text=rf"$$\text{{N}}_{{\text{{cumulative}}}} = {mid_total0}$$",
#     text_font_size="14pt",
#     text_align="left", text_baseline="top",
#     background_fill_color="white", background_fill_alpha=0.6
# )
p_mid.add_layout(label_mid_total)
# p_mid.min_border_top = max(getattr(p_mid, "min_border_top", 0) or 0, 24)


hist_legend_div = Div(width=360, margin=(0,0,10,0),
                      text=build_hist_legend_html(q_tot0,g_tot0,u_tot0,q_yr0,g_yr0,u_yr0))
legend_div = Div(width=360)


def initial_counts_html(cur_year: int) -> str:
    sub = df.loc[df["year"] <= cur_year]

    # Survey counts (same as before)
    parts = []
    for k in ["CLASS","SLACS","BELLS","SL2S","CASSOWARY","AGEL","Serendipitous"]:
        n = int((sub["survey"] == k).sum())
        col = COLOR_MAP[k]
        parts.append(
            rf"<div style='font-size:1em;'><span style='display:inline-block;width:12px;height:12px;background:{col};margin-right:8px;'></span>$$\text{{{k}, N }}={n}$$</div>"
        )
    n_others = int((~sub["survey"].isin(list(COLOR_MAP.keys())[:-1])).sum())
    parts.append(
        "<div style='font-size:1em;'><span style='display:inline-block;width:12px;height:12px;"
        "background:gray;margin-right:8px;'></span>"
        rf"$$\text{{Other surveys, N}} ={n_others}$$</div>"
    )

    # Marker key with actual glyphs
    quasar_svg  = _svg_quasar()
    galaxy_svg  = _svg_circle(fill="#000")
    unknown_svg = _svg_circle(fill="none")

    marker_key_html = (
        "<hr style='margin:8px 0'/>"
        "<div><b>Marker key</b></div>"
        f"<div style='display:flex;align-items:center;gap:8px'>{quasar_svg}<span>Quasar source</span></div>"
        f"<div style='display:flex;align-items:center;gap:8px'>{galaxy_svg}<span>Galaxy source</span></div>"
        f"<div style='display:flex;align-items:center;gap:8px'>{unknown_svg}<span>Unknown source</span></div>"
    )

    return (
        "<div style='font-size:1.2em'><b>Surveys</b></div>"
        + "".join(parts)
        + marker_key_html
    )

legend_div.text = initial_counts_html(initial_year)

# ---- Slider + axis labels ----
start = int(np.ceil(min_year/5.0)*5)        # e.g., 1980 if min_year <= 1980
end   = int(np.ceil(max_year/5.0)*5)        # e.g., 2025 or 2030 depending on data
tick_years = list(range(start, end+1, 5))

YEARS = [int(y) for y in years_all]  # pass to JS
year_slider = Slider(
    start=min_year, end=max_year, value=int(initial_year),   
    step=1, title=r"$$\text{Step through years}$$", show_value=False,
    width=FW+5, bar_color="#777777"
)

# year_slider = Slider(
#     start=min_year, end=max_year, value=int(initial_year), step=1,
#     title=r"$$\text{Year}$$", show_value=False,
#     bar_color="#777777",
#     sizing_mode="stretch_width"
# )

TICK_STRIP = 12          # thin plotting stripe
LABEL_ROOM = 44          # room for rotated labels (try 36–48)
tick_fig = figure(
    frame_width=FW, frame_height=12, toolbar_location=None,
    x_range=(start-0.5, end+0.5), y_range=(0, 1),
)
tick_fig.min_border_left  = 0       
tick_fig.min_border_right = 12
tick_fig.min_border_top   = 0
tick_fig.min_border_bottom= 0
# tick_fig = figure(
#     sizing_mode="stretch_width",
#     height=12 + LABEL_ROOM,          # 12px stripe + ~44px for the rotated labels
#     frame_height=12,
#     toolbar_location=None,
#     x_range=Range1d(start - 0.5, end + 0.5),   # <-- padding on both sides
#     y_range=(0, 1),
# )
# tick_fig.min_border_left  = 10       
# tick_fig.min_border_right = 0#12
# tick_fig.min_border_top   = 0
# tick_fig.min_border_bottom= LABEL_ROOM
tick_fig.yaxis.visible = False
tick_fig.grid.visible  = False
tick_fig.outline_line_color = None
tick_fig.xaxis.ticker = FixedTicker(ticks=tick_years, minor_ticks=YEARS)
tick_fig.xaxis.major_label_orientation = 45
tick_fig.xaxis.major_label_overrides = {y: rf"$${y}$$" for y in tick_years}
tick_fig.xaxis.major_label_text_font_size = "1.4em"
tick_fig.xaxis.major_label_standoff = 6  # small gap from axis line to labels
tick_fig.xaxis.major_tick_out = 8        # keep visible ticks
tick_fig.add_layout(year_marker_tick)

style_ticks(p_top,  label_size="1.2em", major_len=8,  minor_len=4, line_width=1.2)
style_ticks(p_mid,  label_size="1.2em", major_len=8,  minor_len=4, line_width=1.2)
style_ticks(p,      label_size="1.4em", major_len=10, minor_len=5, line_width=1.5)

_marker_key_html = (
    "<hr style='margin:8px 0'/>"
    "<div><b>Marker key</b></div>"
    f"<div style='display:flex;align-items:center;gap:8px'>{_svg_quasar()}<span>Quasar source</span></div>"
    f"<div style='display:flex;align-items:center;gap:8px'>{_svg_circle(fill='#000')}<span>Galaxy source</span></div>"
    f"<div style='display:flex;align-items:center;gap:8px'>{_svg_circle()}<span>Unknown source</span></div>"
)
# ---- JS callback: updates views, bars' alpha, labels, right legend ----
callback = CustomJS(
    args=dict(
        year_span_top=year_marker_top,
        year_span_mid=year_marker_mid,
        tick_marker=year_marker_tick,
        slider=year_slider,
        src=src, vq=v_quasar, vg=v_galaxy, vu=v_unknown, va=v_agel, vau=v_agelu,
        top_src=top_src, mid_src=mid_src,
        label_year=label_year, label_nsys=label_nsys, label_mid_total=label_mid_total,
        legend_div=legend_div, hist_legend_div=hist_legend_div,
        color_map=COLOR_MAP,
        marker_key_html=_marker_key_html
    ),
    code=r"""
    const yr = slider.value;
    const data = src.data;
    const n = data['year'].length;

    // Build the four boolean masks
    const bq = new Array(n);
    const bg = new Array(n);
    const bu = new Array(n);
    const ba = new Array(n);
    const bau = new Array(n);

    let n_this_year = 0;

    for (let i = 0; i < n; i++) {
        const include = (data['year'][i] <= yr);
        data['mask'][i] = include;

        const is_agel = data['is_agel'][i];
        const kind = data['kind_class'][i];

        bq[i] = include && (kind === 'Quasar');
        ba[i] = include && is_agel && (kind !== 'Unknown');
        bau[i] = include && is_agel && (kind === 'Unknown');
        bg[i] = include && (kind === 'Galaxy') && !is_agel;
        bu[i] = include && (kind === 'Unknown') && !is_agel;

        if (data['year'][i] === yr) n_this_year++;
    }

    vq.filter.booleans = bq;
    vg.filter.booleans = bg;
    vu.filter.booleans = bu;
    va.filter.booleans = ba;
    vau.filter.booleans = bau;

    vq.change.emit(); vg.change.emit(); vu.change.emit(); va.change.emit(); vau.change.emit();
    src.change.emit();

    // Top & middle bars: alpha 1 for <= yr, 0 for > yr
    const yrs_top  = top_src.data['year'];
    const alpha_top = top_src.data['alpha'];
    const yrs_mid  = mid_src.data['year'];
    const alpha_mid = mid_src.data['alpha'];

    for (let i = 0; i < yrs_top.length; i++) {
      alpha_top[i] = (yrs_top[i] <= yr) ? 1.0 : 0.0;
      alpha_mid[i] = (yrs_mid[i] <= yr) ? 1.0 : 0.0;
    }
    top_src.change.emit();
    mid_src.change.emit();


    // Update labels on the main plot
    label_year.text = String.raw`$$\text{Year }${yr}$$`;
    label_nsys.text = String.raw`$$\text{N}_{\text{systems}} = ${n_this_year}$$`;

    // Recompute survey counts ≤ yr for right legend
    const counts = {};
    for (let i = 0; i < n; i++) {
        if (data['year'][i] <= yr) {
            const s = data['survey'][i];
            counts[s] = (counts[s] || 0) + 1;
        }
    }
    // Explicit order like your legend:
    const order = ['CLASS','SLACS','BELLS','SL2S','CASSOWARY','AGEL','Serendipitous'];
    let html = "<div style='font-size:1.2em'><b>Surveys</b></div>";
    for (const k of order) {
        const c = counts[k] || 0;
        const col = color_map[k];
        html += String.raw`<div style='font-size:1em;'><span style='display:inline-block;width:12px;height:12px;background:${col};margin-right:8px;'></span>$$\text{${k}, N} = ${c}$$</div>`;
}
    // Others
    let others = 0;
    for (const k in counts) {
        if (!order.includes(k)) others += counts[k];
    }
    html += String.raw`<div style='font-size:1em;'><span style='display:inline-block;width:12px;height:12px;background:gray;margin-right:8px;'></span>$$\text{Other surveys, N} = ${others}$$</div>`;
    legend_div.text = html + marker_key_html;
    if (year_span_top) year_span_top.location = yr;
    if (year_span_mid) year_span_mid.location = yr;
    if (tick_marker)   tick_marker.location   = yr;

    // -------- recompute running totals from mid_src --------
    const q = mid_src.data['Quasar'];
    const g = mid_src.data['Galaxy'];
    const u = mid_src.data['Unknown'];
    let q_tot=0, g_tot=0, u_tot=0;
    for (let i=0; i<yrs_mid.length; i++) {
      if (yrs_mid[i] <= yr) { q_tot += q[i]; g_tot += g[i]; u_tot += u[i]; }
    }

    // Update the middle-panel label in the upper-left
    const n_total = q_tot + g_tot + u_tot;
    label_mid_total.text = String.raw`$$\text{N}_{\text{cumulative}} = ${n_total}$$`;
    // keep it pinned to top-left of data area
    //label_mid_total.x_units = "data";
    //label_mid_total.y_units = "data";
    //label_mid_total.x = p_mid.x_range.start;
    //label_mid_total.y = p_mid.y_range.end;
    //label_mid_total.change.emit();


    // Update the right-column histogram legend text
    hist_legend_div.text = String.raw`
    <div style="font-size:1.2em; margin-bottom:4px;"><b>Histogram Legend</b></div>
    <div style="display:flex; flex-direction:column; gap:6px; font-size:1em;">
    <div style="display:flex; align-items:center; gap:8px;">
        <span style="display:inline-block; width:14px; height:14px; background:#777777; border:1px solid #666;"></span>
        $$\text{Quasar, N }  = ${q_tot}$$
    </div>
    <div style="display:flex; align-items:center; gap:8px;">
        <span style="display:inline-block; width:14px; height:14px; background:#000000; border:1px solid #666;"></span>
        $$\text{Galaxy, N }  = ${g_tot}$$
    </div>
    <div style="display:flex; align-items:center; gap:8px;">
        <span style="display:inline-block; width:14px; height:14px; background:#CCCCCC; border:1px solid #666;"></span>
        $$\text{Unknown, N }  = ${u_tot}$$
    </div>
    </div>`;

    function typesetDiv(divModel) {
        if (window.MathJax && MathJax.typesetPromise) {
            const view = Bokeh.index[divModel.id];
            if (view && view.el) MathJax.typesetPromise([view.el]).catch(()=>{});
        }
    }
    typesetDiv(legend_div);
    typesetDiv(hist_legend_div);
"""
)

year_slider.js_on_event(
    DocumentReady,
    CustomJS(args=dict(
        legend_div=legend_div, hist_legend_div=hist_legend_div, slider=year_slider
    ), code="""
      if (window.MathJax && MathJax.typesetPromise) {
        const els = [];
        [legend_div, hist_legend_div, slider].forEach(m => {
          const v = Bokeh.index[m.id]; if (v && v.el) els.push(v.el);
        });
        MathJax.typesetPromise(els).catch(()=>{});
      }
    """)
)

year_slider.js_on_change("value", callback)


# ---- Layout & output ----
# spacer1 = Spacer(width=1, height=histo_heights-100)

# GAP = histo_heights + 20
# spacer = Spacer(width=1, height=GAP)

# legs = column(
#     mathjax_loader,
#     spacer1,
#     hist_legend_div,   
#     spacer,            
#     legend_div)

# left_indent = Spacer(width=LEFT, height=1)
# slider_row = row(Spacer(width=LEFT), year_slider, sizing_mode="stretch_width")
# ticks_row = row(Spacer(width=LEFT), tick_fig, sizing_mode="stretch_width")
# gap_under_ticks = Spacer(height=6)

# left_stack = column(p_top, 
#                     p_mid, 
#                     p, 
#                     slider_row, 
#                     ticks_row, 
#                     sizing_mode="stretch_width")

# layout = column(row(left_stack, legs, sizing_mode="stretch_width"), footer, sizing_mode="stretch_width")

spacer1 = Spacer(width=1, height=p_top.frame_height-20)

GAP = p_top.frame_height + p_mid.frame_height + 8   
spacer = Spacer(width=1, height=GAP)

left_indent = Spacer(width=LEFT, height=1)
slider_row  = row(Spacer(width=LEFT-10, height=1), year_slider)
ticks_row   = row(left_indent, tick_fig)

legs = column(
    mathjax_loader,
    spacer1,
    hist_legend_div,   
    spacer,            
    legend_div,        
    width=360,
)
left_stack = column(
    p_top,
    p_mid,
    p,
    slider_row,
    ticks_row
)
layout = column(row(left_stack, legs), footer)


datestamp = datetime.now().strftime("%Y%m%d")   # e.g., 20250903
latest_fn = "index.html"
versioned_fn = f"index_{datestamp}.html"

save(layout, filename=latest_fn, title="Confirmed Gravitational Lenses — Interactive", resources=INLINE)
save(layout, filename=versioned_fn, title="Confirmed Gravitational Lenses — Interactive", resources=INLINE)
# save(layout, filename=versioned_fn, title="Confirmed Gravitational Lenses — Interactive", resources=CDN)

html = file_html(layout, resources=INLINE, title="Confirmed Gravitational Lenses — Interactive")
html = html.replace(
    "</head>",
    f'<meta name="author" content="Courtney B. Watson">\n'
    f'<meta name="copyright" content="© {COPY_YEAR} Courtney B. Watson | AGEL Team">\n'
    "</head>"
)

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {{ bokeh_css }}
  {{ bokeh_js }}
  <style>
    body { margin: 0; }
    .page { max-width: 1200px; margin: 0 auto; padding: 12px; }
  </style>
</head>
<body>
  <div class="page">
    {{ plot_div|safe }}
  </div>
  {{ plot_script|safe }}
</body>
</html>
"""

html = file_html(layout, INLINE, title="Lenses", template=TEMPLATE)
with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)
with open(f"index_{datestamp}.html", "w", encoding="utf-8") as f:
    f.write(html)


