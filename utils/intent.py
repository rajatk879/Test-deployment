VIS_KEYWORDS = [
    "chart", "plot", "graph", "visualize", "visualisation", "visualization",
    "bar chart", "line chart", "draw", "scatter", "histogram", "pie chart",
    "show me", "display", "compare", "trend", "over time", "by", "across",
    "distribution", "breakdown", "analysis", "overview"
]

def wants_chart(text: str) -> bool:
    """Return True if user explicitly asks for a visualization or uses common visualization phrases."""
    text = text.lower()
    # Check for explicit visualization keywords
    if any(k in text for k in VIS_KEYWORDS):
        return True
    # Check for common patterns that suggest visualization
    if any(phrase in text for phrase in ["show me", "display", "compare", "trend", "over time"]):
        return True
    return False
