import matplotlib.pyplot as plt
import base64
from io import BytesIO
from utils.llm_factory import load_llm

class SummarizerAgent:
    def __init__(self):
        # Increase max_tokens to 4000 to prevent response truncation
        self.llm = load_llm(temp=0.2, max_tokens=4000)

    def summarize(self, q, df):
        prompt = f"""
You are a senior data analyst. Summarize insights in 2 short bullet points.

Question: {q}

Data sample:
{df.head().to_string()}

Provide clear, short, human-friendly insights.
"""
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    
    # ---------------------------------------------
    # Detect chart type based on question
    # ---------------------------------------------
    def detect_chart_type(self, question: str):
        q = question.lower()

        if "line" in q or "trend" in q or "time series" in q:
            return "line"
        if "bar" in q or "compare" in q or "comparison" in q:
            return "bar"
        if "scatter" in q or "relationship" in q or "correlation" in q:
            return "scatter"
        if "hist" in q or "distribution" in q:
            return "hist"
        if "pie" in q:
            return "pie"

        return "auto"  # fallback

    def generate_viz(self, question, df):
        if df.empty:
            return None, None

        chart_type = self.detect_chart_type(question)
        plt.figure(figsize=(8, 4))

        # Auto-select columns
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()

        # default selections
        x = non_numeric_cols[0] if non_numeric_cols else df.columns[0]
        y = numeric_cols[0] if numeric_cols else None

        # ---------------------------------------------
        # CHART TYPE HANDLERS
        # ---------------------------------------------
        try:
            if chart_type == "line":
                if y is None:
                    return None, None
                df.plot.line(x=x, y=y)

            elif chart_type == "bar":
                if y is None:
                    return None, None
                df.plot.bar(x=x, y=y)

            elif chart_type == "scatter":
                if len(numeric_cols) < 2:
                    return None, None
                df.plot.scatter(x=numeric_cols[0], y=numeric_cols[1])

            elif chart_type == "hist":
                if y is None:
                    return None, None
                df[y].plot.hist()

            elif chart_type == "pie":
                if y is None:
                    return None, None
                df.set_index(x)[y].plot.pie(autopct="%1.1f%%")

            # fallback → auto
            else:
                df.plot()

            # ---------------------------------------------
            # Export PNG for frontend
            # ---------------------------------------------
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png")
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode("utf-8")
            plt.close()

            return encoded, "image/png"

        except Exception as e:
            print("Plot error:", e)
            return None, None