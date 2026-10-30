"""Visual style for dashboards, matching the SHW dashboard mock-up: navy headings,
a strong blue for the focus group, a pale blue-grey for the comparison, white
rounded cards on a light background.

All colours and the font live here. When SHW's official design rules (colours,
typeface) are applied, this is the only file that needs to change."""

FONT = '"Source Sans 3", "Source Sans Pro", "Helvetica Neue", Arial, sans-serif'
FONT_URL = "https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap"

NAVY = "#0C2350"        # titles and big numbers
FOCUS = "#2D5BAA"       # the group in focus (e.g. Faculty of Arts)
OVERALL = "#C8D3E4"     # the comparison (everyone)
TEXT = "#1D2B44"
MUTED = "#5E6B82"
GRID = "#E6EBF3"
PAGE = "#F2F5FA"
CARD = "#FFFFFF"
BORDER = "#E3E8F0"

# Ranked bars fade from strong to pale, like "Top reasons for visit".
RAMP = ["#2D5BAA", "#4470B8", "#5E86C6", "#7C9DD3", "#9BB4DF", "#B7C9E9", "#D0DCF1", "#E1E9F6"]

# Donut slices, strongest first (like Undergraduate / Graduate in the mock-up).
DONUT = ["#13306A", "#2D5BAA", "#5E86C6", "#93AEDC", "#BFD0EC", "#DCE5F4"]

UP = "#1E9E62"          # change in the good direction
DOWN = "#D2453F"        # change in the bad direction
ALERT_BG = {"red": "#FCEBEA", "amber": "#FDF4E1", "info": "#E9F0FB"}
ALERT_FG = {"red": "#C93A34", "amber": "#C98A10", "info": "#2D5BAA"}


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
