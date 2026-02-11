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

        if "line" in q or "trend" in q or "time series" in q or "over time" in q:
            return "line"
        if "bar" in q or "compare" in q or "comparison" in q or "by" in q and ("mill" in q or "recipe" in q or "flour" in q):
            return "bar"
        if "scatter" in q or "relationship" in q or "correlation" in q:
            return "scatter"
        if "hist" in q or "distribution" in q:
            return "hist"
        if "pie" in q or "breakdown" in q or "share" in q:
            return "pie"
        
        # Auto-detect based on data structure
        # If question mentions grouping by categorical variable, prefer bar chart
        if any(word in q for word in ["by mill", "by recipe", "by flour", "by region", "by month", "by week"]):
            return "bar"
        # If question mentions time-based analysis, prefer line chart
        if any(word in q for word in ["daily", "weekly", "monthly", "yearly", "over time", "across time"]):
            return "line"

        return "auto"  # fallback

    def generate_viz(self, question, df):
        if df.empty:
            return None, None

        chart_type = self.detect_chart_type(question)
        plt.figure(figsize=(10, 6))

        # Auto-select columns
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()

        # Smart column selection for MC4 domain
        # Prefer date columns for x-axis in time series
        date_cols = [col for col in df.columns if 'date' in col.lower() or col.lower() in ['date', 'week', 'month', 'year', 'period']]
        # Prefer common dimension columns
        dim_cols = [col for col in non_numeric_cols if any(term in col.lower() for term in ['mill', 'recipe', 'sku', 'flour', 'region', 'country'])]
        
        # Select x-axis: prefer date, then dimension columns, then first non-numeric
        x = None
        if date_cols:
            x = date_cols[0]
        elif dim_cols:
            x = dim_cols[0]
        elif non_numeric_cols:
            x = non_numeric_cols[0]
        else:
            x = df.columns[0]
        
        # Select y-axis: prefer common metric columns
        metric_keywords = ['tons', 'hours', 'utilization', 'capacity', 'demand', 'forecast', 'cost', 'rate', 'pct', 'percent']
        y = None
        if numeric_cols:
            # Prefer columns with metric keywords
            preferred_metrics = [col for col in numeric_cols if any(kw in col.lower() for kw in metric_keywords)]
            y = preferred_metrics[0] if preferred_metrics else numeric_cols[0]

        # ---------------------------------------------
        # CHART TYPE HANDLERS
        # ---------------------------------------------
        try:
            if chart_type == "line":
                if y is None:
                    return None, None
                plt.plot(df[x], df[y], marker='o', linewidth=2, markersize=4)
                plt.xlabel(x.replace('_', ' ').title())
                plt.ylabel(y.replace('_', ' ').title())
                plt.title(f"{y.replace('_', ' ').title()} Over Time")
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)

            elif chart_type == "bar":
                if y is None:
                    return None, None
                plt.bar(range(len(df)), df[y])
                plt.xlabel(x.replace('_', ' ').title())
                plt.ylabel(y.replace('_', ' ').title())
                plt.title(f"{y.replace('_', ' ').title()} by {x.replace('_', ' ').title()}")
                plt.xticks(range(len(df)), df[x], rotation=45, ha='right')
                plt.grid(True, alpha=0.3, axis='y')

            elif chart_type == "scatter":
                if len(numeric_cols) < 2:
                    return None, None
                plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6)
                plt.xlabel(numeric_cols[0].replace('_', ' ').title())
                plt.ylabel(numeric_cols[1].replace('_', ' ').title())
                plt.title(f"{numeric_cols[1].replace('_', ' ').title()} vs {numeric_cols[0].replace('_', ' ').title()}")
                plt.grid(True, alpha=0.3)

            elif chart_type == "hist":
                if y is None:
                    return None, None
                plt.hist(df[y], bins=min(20, len(df)), edgecolor='black', alpha=0.7)
                plt.xlabel(y.replace('_', ' ').title())
                plt.ylabel('Frequency')
                plt.title(f"Distribution of {y.replace('_', ' ').title()}")
                plt.grid(True, alpha=0.3, axis='y')

            elif chart_type == "pie":
                if y is None:
                    return None, None
                # Limit to top 10 for readability
                plot_df = df.nlargest(10, y) if len(df) > 10 else df
                plt.pie(plot_df[y], labels=plot_df[x], autopct='%1.1f%%', startangle=90)
                plt.title(f"{y.replace('_', ' ').title()} Breakdown")

            # fallback → auto
            else:
                if y is not None:
                    if x in date_cols or 'date' in str(x).lower():
                        plt.plot(df[x], df[y], marker='o')
                    else:
                        plt.bar(range(len(df)), df[y])
                        plt.xticks(range(len(df)), df[x], rotation=45, ha='right')
                    plt.xlabel(x.replace('_', ' ').title())
                    plt.ylabel(y.replace('_', ' ').title())
                    plt.grid(True, alpha=0.3)
                else:
                    df.plot()

            # ---------------------------------------------
            # Export PNG for frontend
            # ---------------------------------------------
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode("utf-8")
            plt.close()

            return encoded, "image/png"

        except Exception as e:
            print(f"Plot error: {e}")
            import traceback
            traceback.print_exc()
            return None, None