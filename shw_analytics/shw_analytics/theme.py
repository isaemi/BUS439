"""Visual style for the whole product, following the UBC brand: UBC Blue with the official
secondary blues, white space, and Whitney where it is licensed (Arial otherwise, UBC's
required alternate).

Colours come from the UBC colour guide (brand.ubc.ca). All colours and fonts live here:
the dashboards, the chat graphics, reports and the app chrome all read from this file.

The UBC logo is NOT drawn here. If SHW provides an official logo file, save it as
assets/ubc-logo.svg (or .png) and it appears in the header; never redraw or recolour it.
"""
import base64
from pathlib import Path

# ---- UBC brand palette ---------------------------------------------------------------
UBC_BLUE = "#002145"        # primary (PMS 282)
BLUE_2935 = "#0055B7"       # secondary blues, darkest to lightest
BLUE_3005 = "#0077C8"
BLUE_2995 = "#00A7E1"
BLUE_298 = "#40B4E5"
BLUE_297 = "#6EC4E8"
BLUE_2975 = "#97D4E9"
UBC_GREY = "#2F5D7C"        # secondary (PMS 5405), used for the unit name
GREY_LIGHT = "#C3D0DB"      # UBC grey series (PMS 5435)

# Whitney is UBC's typeface but is licensed; it shows where installed, Arial otherwise.
FONT = '"Whitney SSm A", "Whitney SSm B", Whitney, Arial, Helvetica, sans-serif'
FONT_URL = ""               # no web font to load; see font_link()

# ---- roles (what the rest of the code uses) -----------------------------------------
NAVY = UBC_BLUE             # titles and big numbers
FOCUS = BLUE_2935           # the group in focus, primary actions, links
OVERALL = GREY_LIGHT        # the comparison (everyone)
QUIET = "#9AA9BB"           # context in charts: within noise, not flagged
TEXT = "#14233B"
MUTED = "#4F5F78"
GRID = "#E4EAF1"
PAGE = "#F4F6F9"
CARD = "#FFFFFF"
BORDER = "#D8E0EA"

# Ranked bars fade from strong to pale: the six official blues, then two tints.
RAMP = [BLUE_2935, BLUE_3005, BLUE_2995, BLUE_298, BLUE_297, BLUE_2975, "#BFE3F1", "#DDF0F8"]

# Donut slices, strongest first.
DONUT = [UBC_BLUE, BLUE_2935, BLUE_3005, BLUE_2995, BLUE_297, BLUE_2975]

UP = "#1E8E5A"              # change in the good direction
DOWN = "#C8372D"            # change in the bad direction
ALERT_BG = {"red": "#FBEBE9", "amber": "#FDF4E1", "info": "#E6F1FA"}
ALERT_FG = {"red": "#B83228", "amber": "#C98A10", "info": BLUE_2935}

APP_NAME = "Student Feedback Assistant"
UNIT_NAME = "UBC Student Health and Wellbeing"
ASSETS = Path(__file__).resolve().parent.parent / "assets"


def font_link():
    """<link> for a web font, or nothing when the font stack needs none."""
    return f'<link href="{FONT_URL}" rel="stylesheet">' if FONT_URL else ""


def altair_config():
    """Altair/Vega-Lite config applied to every chart."""
    return {
        "background": "transparent",
        "font": FONT,
        "view": {"stroke": None},
        "axis": {"labelColor": MUTED, "labelFontSize": 12, "titleColor": MUTED, "titleFontSize": 12,
                 "titleFontWeight": "normal", "gridColor": GRID, "domain": False, "ticks": False,
                 "labelPadding": 6},
        "axisX": {"grid": False},
        "legend": {"orient": "top", "direction": "horizontal", "title": None, "labelColor": TEXT,
                   "labelFontSize": 12, "symbolType": "circle", "symbolSize": 110},
        "bar": {"cornerRadiusEnd": 2},
        "line": {"strokeWidth": 2.5},
        "point": {"filled": True, "size": 45},
        "text": {"font": FONT, "fontSize": 12, "color": TEXT},
    }


# ---- app chrome ----------------------------------------------------------------------

def _logo():
    """The official UBC logo, only if SHW has supplied the file. Used as given."""
    for name, mime in (("ubc-logo.svg", "image/svg+xml"), ("ubc-logo.png", "image/png")):
        f = ASSETS / name
        if f.exists():
            data = base64.b64encode(f.read_bytes()).decode()
            return f'<img class="shw-logo" alt="UBC" src="data:{mime};base64,{data}">'
    return ""


def header_html(note=""):
    """University bar, unit banner and app name, in the order UBC sites use."""
    note_html = f'<p class="shw-note">{note}</p>' if note else ""
    return f"""
<header class="shw-header">
  <div class="shw-ubc">{_logo()}<span>The University of British Columbia</span></div>
  <div class="shw-unit">
    <div class="shw-unit-name">{UNIT_NAME}</div>
    <div class="shw-app-name">{APP_NAME}</div>
    {note_html}
  </div>
  <div class="shw-rule"></div>
</header>"""


def footer_html():
    return ('<footer class="shw-footer">Prototype built for UBC Student Health and Wellbeing. '
            'Not an official UBC website. Answers use de-identified data only.</footer>')


def app_css():
    """Styles for the Streamlit page. Selectors use Streamlit's data-testid hooks, with plain
    element selectors as a fallback, so a Streamlit update degrades to defaults, not breakage."""
    return f"""<style>
/* Brand font on text only. Icons are drawn with their own icon font and must keep it. */
html, body, .stApp, p, li, label, h1, h2, h3, h4, h5, h6, input, textarea, select,
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"], button[data-baseweb="tab"] p, .stButton > button p {{ font-family: {FONT}; }}
.stApp {{ background: {PAGE}; color: {TEXT}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.4rem; max-width: 1280px; }}

/* header: university bar, unit banner, rule */
.shw-header {{ border-radius: 10px; overflow: hidden; border: 1px solid {BORDER}; margin: 0 0 1.1rem;
               background: {CARD}; }}
.shw-ubc {{ background: {UBC_BLUE}; color: #fff; display: flex; align-items: center; gap: 14px;
            padding: 11px 24px; font-size: 12.5px; font-weight: 700; letter-spacing: .2em;
            text-transform: uppercase; }}
.shw-logo {{ height: 30px; width: auto; display: block; }}
.shw-unit {{ padding: 22px 24px 20px; background: linear-gradient(180deg, #E8EDF4 0%, #F7F9FC 100%); }}
.shw-unit-name {{ color: {UBC_GREY}; font-size: 30px; font-weight: 700; line-height: 1.15; }}
.shw-app-name {{ color: {UBC_BLUE}; font-size: 18px; font-weight: 700; margin-top: 6px; }}
.shw-note {{ color: {MUTED}; font-size: 14px; margin: 8px 0 0; max-width: 80ch; line-height: 1.5; }}
.shw-rule {{ height: 4px; background: linear-gradient(90deg, {BLUE_2935}, {BLUE_2995} 55%, {BLUE_2975}); }}
.shw-footer {{ color: {MUTED}; font-size: 13px; border-top: 1px solid {BORDER}; margin-top: 2.5rem;
               padding: 14px 0 6px; }}

/* type */
h1, h2, h3, h4 {{ color: {UBC_BLUE}; letter-spacing: 0; }}
a {{ color: {BLUE_2935}; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: {MUTED}; }}

/* tabs */
[data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
button[data-baseweb="tab"] {{ color: {MUTED}; font-weight: 600; padding: 10px 16px; }}
button[data-baseweb="tab"]:hover {{ color: {UBC_BLUE}; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {UBC_BLUE}; }}
[data-baseweb="tab-highlight"] {{ background: {BLUE_2935}; height: 3px; }}

/* buttons */
.stButton > button, .stDownloadButton > button {{ border: 1px solid {UBC_BLUE}; color: {UBC_BLUE};
    background: #fff; border-radius: 6px; font-weight: 600; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ background: {UBC_BLUE}; color: #fff;
    border-color: {UBC_BLUE}; }}
.stButton > button[kind="primary"] {{ background: {BLUE_2935}; border-color: {BLUE_2935}; color: #fff; }}
.stButton > button[kind="primary"]:hover {{ background: {UBC_BLUE}; border-color: {UBC_BLUE}; }}
button:focus-visible, input:focus-visible, textarea:focus-visible, [role="tab"]:focus-visible {{
    outline: 3px solid {BLUE_2995}; outline-offset: 2px; }}

/* chat */
[data-testid="stChatMessage"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 14px 18px; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ background: #EAF2FA;
    border-color: #CFE0F1; }}
[data-testid="stChatInput"] textarea {{ color: {TEXT}; }}
[data-testid="stChatInput"] {{ border: 1px solid {GREY_LIGHT}; border-radius: 10px; }}

/* cards: expanders, metrics, tables */
[data-testid="stExpander"] details {{ border: 1px solid {BORDER}; border-radius: 8px; background: {CARD}; }}
[data-testid="stExpander"] summary {{ color: {UBC_BLUE}; font-weight: 600; }}
[data-testid="stMetric"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 14px 18px; }}
[data-testid="stMetricValue"] {{ color: {UBC_BLUE}; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: {CARD}; border-color: {BORDER}; border-radius: 10px; }}
.shw-gtitle {{ color: {UBC_BLUE}; font-size: 18px; font-weight: 700; line-height: 1.3; }}
.shw-gscope {{ color: {MUTED}; font-size: 14px; line-height: 1.45; margin-top: 3px; }}
.shw-num {{ color: {MUTED}; font-size: 13px; font-weight: 700; letter-spacing: .04em; }}
.shw-tag {{ display: inline-block; background: {ALERT_BG['info']}; color: {BLUE_2935}; border-radius: 10px;
            padding: 1px 10px; font-size: 12px; font-weight: 700; margin-left: 8px; }}
.shw-tag.new {{ background: {ALERT_BG['amber']}; color: #8A5E00; }}
.shw-detail {{ color: {TEXT}; font-size: 15px; line-height: 1.55; margin: 6px 0 2px; }}
.shw-small {{ color: {MUTED}; font-size: 13px; line-height: 1.5; margin: 4px 0 0; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: #fff; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: {UBC_BLUE}; }}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ border-top: 4px solid {UBC_BLUE}; }}

@media (max-width: 720px) {{
  .shw-unit-name {{ font-size: 23px; }}
  .shw-ubc {{ letter-spacing: .12em; font-size: 11.5px; padding: 10px 16px; }}
  .shw-unit {{ padding: 16px; }}
}}
</style>"""
