"""Chart selection from the question, the result schema and the actual rows (no hardcoded column names)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import pandas as pd
import plotly.graph_objects as go

# Reference palette (dataviz skill, validated): categorical slots in fixed order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MAX_SERIES = 7  # 8th slot is reserved for "Other"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


@dataclass
class ChartSpec:
    kind: str  # line | bar | pie | scatter | heatmap | metric | none
    x: str | None = None
    y: str | None = None
    color: str | None = None
    z: str | None = None
    y_multi: list[str] = field(default_factory=list)  # grouped bars: several measures with one unit
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    unit: str | None = None
    notes: list[str] = field(default_factory=list)


def column_unit(name: str, energy_unit: str = "kWh") -> str | None:
    n = name.lower()
    if re.search(r"pct|percent|share|ratio_pct", n):
        return "%"
    if re.search(r"kwh|energy|consumption|usage|load|demand|electricity", n):
        return energy_unit
    if "temp" in n:
        return "°C"
    if "wind" in n:
        return "m/s"
    if re.search(r"sqm|area", n):
        return "m²"
    return None


def _label(col: str, unit: str | None) -> str:
    words = [w for w in col.split("_") if w and (not unit or w.lower() != unit.lower())]
    base = " ".join(words).strip().capitalize() or col
    return f"{base} ({unit})" if unit else base


def _is_time(s: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object and len(s.dropna()):
        sample = s.dropna().astype(str).head(20)
        return bool(sample.str.match(r"^\d{4}-\d{2}(-\d{2})?([ T]\d{2}(:\d{2}){0,2})?$").all())
    return bool(re.search(r"^(ts|timestamp|date|day|week|month|datetime|hour_ts)$", name.lower()))


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    out = {"time": [], "numeric": [], "category": [], "cyclic": []}
    for c in df.columns:
        s = df[c]
        if _is_time(s, c):
            out["time"].append(c)
        elif re.search(r"^(hour|hr|hour_of_day|dow|weekday|day_of_week|month_num)$", c.lower()):
            out["cyclic"].append(c)
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            out["numeric"].append(c)
        else:
            out["category"].append(c)
    return out


def select_chart(question: str, df: pd.DataFrame | None, energy_unit: str = "kWh") -> ChartSpec:
    if df is None or df.empty:
        return ChartSpec("none", notes=["No rows to chart."])
    q = (question or "").lower()
    cols = classify_columns(df)
    num, cat, tim, cyc = cols["numeric"], cols["category"], cols["time"], cols["cyclic"]
    # prefer an energy measure as y
    num_sorted = sorted(num, key=lambda c: column_unit(c, energy_unit) != energy_unit)
    if not num:
        return ChartSpec("none", notes=["Result has no numeric column to chart."])
    y = num_sorted[0]
    unit = column_unit(y, energy_unit)
    title = (question or "Result").strip().rstrip("?")[:90]
    if len(df) == 1 and len(num) <= 3 and not tim:
        return ChartSpec("metric", y=y, unit=unit, title=title)
    wants_heat = re.search(r"heat ?map|pattern|by hour and|hour.*(weekday|day of week)", q)
    if wants_heat and len(cyc) + len(cat) + len(tim) >= 2:
        dims = (cyc + cat + tim)[:2]
        return ChartSpec("heatmap", x=dims[0], y=dims[1], z=y, unit=unit, title=title,
                         x_label=_label(dims[0], None), y_label=_label(dims[1], None))
    if tim:
        x = tim[0]
        color = cat[0] if cat else None
        return ChartSpec("line", x=x, y=y, color=color, unit=unit, title=title, x_label=_label(x, None),
                         y_label=_label(y, unit))
    if re.search(r"share|proportion|percent of total|breakdown|split|composition", q) and cat:
        n = df[cat[0]].nunique()
        if n <= 8 and (df[y].dropna() >= 0).all():
            return ChartSpec("pie", x=cat[0], y=y, unit=unit, title=title)
    same_unit = [c for c in num_sorted if column_unit(c, energy_unit) == unit]
    if cat and len(same_unit) >= 2 and not re.search(r"correlat|relationship|scatter|against", q):
        x = cat[0]
        return ChartSpec("bar", x=x, y=y, y_multi=same_unit[:MAX_SERIES], unit=unit, title=title,
                         x_label=_label(x, None), y_label=_label("value", unit))
    if len(num) >= 2 and (re.search(r"\bvs\b|versus|correlat|relationship|against|scatter", q) or not (cat or cyc)):
        x2 = num_sorted[1] if num_sorted[0] == y else num_sorted[0]
        xu = column_unit(x2, energy_unit)
        return ChartSpec("scatter", x=x2, y=y, color=cat[0] if cat else None, unit=unit, title=title,
                         x_label=_label(x2, xu), y_label=_label(y, unit))
    if cyc:
        x = cyc[0]
        return ChartSpec("line", x=x, y=y, color=cat[0] if cat else None, unit=unit, title=title,
                         x_label=_label(x, None), y_label=_label(y, unit))
    if cat:
        x = cat[0]
        color = cat[1] if len(cat) > 1 else None
        return ChartSpec("bar", x=x, y=y, color=color, unit=unit, title=title, x_label=_label(x, None),
                         y_label=_label(y, unit))
    return ChartSpec("none", notes=["No suitable chart for this result shape."])


def _fold_series(df: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    """Keep the MAX_SERIES largest series; sum the rest into 'Other' (identity never by cycled colour)."""
    if not spec.color or df[spec.color].nunique() <= MAX_SERIES + 1:
        return df
    top = df.groupby(spec.color)[spec.y].sum().nlargest(MAX_SERIES).index
    d = df.copy()
    d[spec.color] = d[spec.color].where(d[spec.color].isin(top), "Other")
    spec.notes.append(f"Series beyond the top {MAX_SERIES} are summed into 'Other'.")
    return d.groupby([spec.x, spec.color], as_index=False, sort=False)[spec.y].sum(min_count=1)


def _layout(fig: go.Figure, spec: ChartSpec) -> go.Figure:
    fig.update_layout(
        title=dict(text=spec.title, font=dict(size=16, color=INK)), template="simple_white",
        font=dict(color=INK2, size=12), margin=dict(l=60, r=20, t=60, b=50), hovermode="x unified" if spec.kind == "line" else "closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None),
    )
    fig.update_xaxes(title_text=spec.x_label, showgrid=False, linecolor=GRID)
    fig.update_yaxes(title_text=spec.y_label, gridcolor=GRID, linecolor=GRID, zeroline=False)
    return fig


def render(spec: ChartSpec, df: pd.DataFrame) -> go.Figure | None:
    if spec.kind in ("none", "metric") or df is None or df.empty:
        return None
    spec = replace(spec, notes=list(spec.notes))  # never mutate a cached spec
    fig = go.Figure()
    unit_sfx = f" {spec.unit}" if spec.unit else ""
    if spec.kind in ("line", "bar", "scatter"):
        d = _fold_series(df, spec) if spec.kind == "line" else df
        if spec.y_multi:  # wide measures -> long format, one colour per measure
            d = d.melt(id_vars=[spec.x], value_vars=spec.y_multi, var_name="measure", value_name="value")
            spec.color, spec.y = "measure", "value"
        groups = [(None, d)] if not spec.color else list(d.groupby(spec.color, sort=False))
        if spec.color:  # stable colour per entity: order by name, not by rank
            groups.sort(key=lambda g: (str(g[0]) == "Other", str(g[0])))
        for i, (name, g) in enumerate(groups[: MAX_SERIES + 1]):
            col = SERIES[min(i, len(SERIES) - 1)] if name != "Other" else SERIES[-1]
            hover = f"%{{x}}<br>%{{y:,.2f}}{unit_sfx}<extra>{name or ''}</extra>"
            if spec.kind == "line":
                g = g.sort_values(spec.x)
                fig.add_scatter(x=g[spec.x], y=g[spec.y], mode="lines", name=str(name or spec.y),
                                line=dict(width=2, color=col), hovertemplate=hover, connectgaps=False)
            elif spec.kind == "bar":
                fig.add_bar(x=g[spec.x].astype(str), y=g[spec.y], name=str(name or spec.y), marker_color=col,
                            hovertemplate=hover, texttemplate="%{y:,.0f}", textposition="outside", cliponaxis=False)
            else:
                fig.add_scatter(x=g[spec.x], y=g[spec.y], mode="markers", name=str(name or spec.y),
                                marker=dict(size=8, color=col, line=dict(width=1, color="#fcfcfb")), hovertemplate=hover)
        fig.update_layout(showlegend=bool(spec.color), barmode="group", bargap=0.25)
        if spec.kind == "bar" and not spec.color and len(df) > 12:
            fig.update_traces(texttemplate=None)
    elif spec.kind == "pie":
        fig.add_pie(labels=df[spec.x].astype(str), values=df[spec.y], hole=0.5, sort=False,
                    marker=dict(colors=SERIES[: len(df)], line=dict(color="#fcfcfb", width=2)),
                    hovertemplate=f"%{{label}}: %{{value:,.1f}}{unit_sfx} (%{{percent}})<extra></extra>")
    elif spec.kind == "heatmap":
        piv = df.pivot_table(index=spec.y, columns=spec.x, values=spec.z, aggfunc="mean")
        fig.add_heatmap(z=piv.values, x=[str(c) for c in piv.columns], y=[str(i) for i in piv.index],
                        colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
                        colorbar=dict(title=spec.unit or spec.z), hovertemplate=f"%{{x}} / %{{y}}: %{{z:,.2f}}{unit_sfx}<extra></extra>")
    return _layout(fig, spec)


def forecast_figure(history: pd.DataFrame, fc: pd.DataFrame, title: str, unit: str = "kWh") -> go.Figure:
    """history: ts, actual; fc: ts, expected, lower, upper[, actual]."""
    fig = go.Figure()
    fig.add_scatter(x=fc["ts"], y=fc["upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fc["ts"], y=fc["lower"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(42,120,214,0.18)", name="80% interval", hovertemplate=f"lower %{{y:,.1f}} {unit}<extra></extra>")
    fig.add_scatter(x=history["ts"], y=history["actual"], mode="lines", name="Actual (history)",
                    line=dict(width=2, color=INK2), hovertemplate=f"%{{y:,.1f}} {unit}<extra>actual</extra>")
    fig.add_scatter(x=fc["ts"], y=fc["expected"], mode="lines", name="Expected / forecast",
                    line=dict(width=2, color=SERIES[0], dash="dash"), hovertemplate=f"%{{y:,.1f}} {unit}<extra>expected</extra>")
    if "actual" in fc and fc["actual"].notna().any():
        fig.add_scatter(x=fc["ts"], y=fc["actual"], mode="lines", name="Actual (evaluation)",
                        line=dict(width=2, color=SERIES[1]), hovertemplate=f"%{{y:,.1f}} {unit}<extra>actual</extra>")
    spec = ChartSpec("line", title=title, x_label="Time (site local)", y_label=f"Hourly electricity ({unit})")
    return _layout(fig, spec)
