VIS_KEYWORDS = [
    "chart", "plot", "graph", "visualize", "visualisation",
    "bar chart", "line chart", "draw", "scatter", "histogram"
]

def wants_chart(text: str) -> bool:
    """Return True only if user explicitly asks for a visualization."""
    text = text.lower()
    return any(k in text for k in VIS_KEYWORDS)
