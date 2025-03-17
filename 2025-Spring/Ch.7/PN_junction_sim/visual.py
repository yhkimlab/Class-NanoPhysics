
import plotly.graph_objects as go
from abc import ABC, abstractmethod

# -----------------------------------------------------------------------------
# General Plotly Plot Helper
# -----------------------------------------------------------------------------
FIG_WIDTH = 1200
FIG_HEIGHT = 800

class PlotlyPlotter:
    @staticmethod
    def build_figure(title, x_data, traces, xaxis_title="x", yaxis_title="y",
                     yaxis_type="linear", y_range=None, width=FIG_WIDTH, height=FIG_HEIGHT):
        """
        General method to build a Plotly figure.
        """
        fig = go.Figure()
        for trace in traces:
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=trace.get("y"),
                    mode=trace.get("mode", "lines"),
                    name=trace.get("name", ""),
                    line=trace.get("line", {})
                )
            )
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            width=width, height=height
        )
        fig.update_yaxes(type=yaxis_type, tickformat=".2e")
        if y_range is not None:
            fig.update_yaxes(range=y_range)
        return fig

# -----------------------------------------------------------------------------
# Plot Classes (Concrete Implementations)
# -----------------------------------------------------------------------------
class AbstractPlot(ABC):
    @abstractmethod
    def build(self, data, y_range=None):
        """
        Build and return a Plotly Figure.
        'data' is expected to be a simulation result dictionary or a list of results (for full sweep plots).
        'y_range' is an optional tuple to fix the y–axis.
        """
        pass

class DopingDensityPlot(AbstractPlot):
    def build(self, data, y_range=None):
        x = data['x']
        traces = [
            {"y": data['Dop_x'], "name": "Dop_x", "mode": "lines"},
            {"y": data['Dop_y'], "name": "Dop_y", "mode": "lines"}
        ]
        return PlotlyPlotter.build_figure(
            title="Doping Density",
            x_data=x,
            traces=traces,
            xaxis_title="x (µm)",
            yaxis_title="Doping (/cm³)",
            y_range=y_range
        )

class CarrierDensityPlot(AbstractPlot):
    def build(self, data, y_range=None):
        x = data['x']
        traces = [
            {"y": data['p_plot'], "name": "p", "mode": "lines", "line": {"color": "red"}},
            {"y": data['n_plot'], "name": "n", "mode": "lines", "line": {"color": "blue"}}
        ]
        return PlotlyPlotter.build_figure(
            title="Carrier Density",
            x_data=x,
            traces=traces,
            xaxis_title="x (µm)",
            yaxis_title="Carrier Density (/cm³)",
            yaxis_type="log",
            y_range=y_range
        )

class NetChargeDensityPlot(AbstractPlot):
    def build(self, data, y_range=None):
        x = data['x']
        traces = [{"y": data['Charge'], "name": "Charge", "mode": "lines"}]
        return PlotlyPlotter.build_figure(
            title="Net Charge Density",
            x_data=x,
            traces=traces,
            xaxis_title="x (µm)",
            yaxis_title="Charge (/cm³)",
            y_range=y_range
        )

class ElectricFieldPlot(AbstractPlot):
    def build(self, data, y_range=None):
        x = data['x']
        traces = [{"y": data['E_field_plot'], "name": "E Field", "mode": "lines"}]
        return PlotlyPlotter.build_figure(
            title="Electric Field",
            x_data=x,
            traces=traces,
            xaxis_title="x (µm)",
            yaxis_title="E Field (kV/cm)",
            y_range=y_range
        )

class ElectrostaticPotentialPlot(AbstractPlot):
    def build(self, data, y_range=None):
        x = data['x']
        traces = [{"y": data['E_potential'], "name": "Potential", "mode": "lines"}]
        return PlotlyPlotter.build_figure(
            title="Electrostatic Potential",
            x_data=x,
            traces=traces,
            xaxis_title="x (µm)",
            yaxis_title="Potential (V)",
            y_range=y_range
        )

class EnergyBandsPlot(AbstractPlot):
    def build(self, data, y_range=None):
        x = data['x']
        shift = data['v_applied'] / 2
        traces = [
            {"y": data['E_band_con'] + shift, "name": "Conduction Band", "mode": "lines"},
            {"y": data['E_band_int'] + shift, "name": "Intrinsic Level", "mode": "lines", "line": {"dash": "dash"}},
            {"y": data['E_band_val'] + shift, "name": "Valence Band", "mode": "lines"},
            {"y": data['Fp'] + shift, "name": "Fp", "mode": "lines"},
            {"y": data['Fn'] + shift, "name": "Fn", "mode": "lines"}
        ]
        return PlotlyPlotter.build_figure(
            title="Energy Bands",
            x_data=x,
            traces=traces,
            xaxis_title="x (µm)",
            yaxis_title="Energy (eV)",
            y_range=y_range
        )

class CurrentDensityDistributionPlot(AbstractPlot):
    def build(self, data, y_range=None):
        x = data['x']
        traces = [
            {"y": data['current_density']['Jp'], "name": "Hole current", "mode": "lines"},
            {"y": data['current_density']['Jn'], "name": "Electron current", "mode": "lines"},
            {"y": data['current_density']['Jt'], "name": "Total current", "mode": "lines"}
        ]
        return PlotlyPlotter.build_figure(
            title="Current Density Distribution",
            x_data=x,
            traces=traces,
            xaxis_title="x (µm)",
            yaxis_title="Current Density (A/cm²)",
            yaxis_type="log",
            y_range=y_range
        )

class JVCurvePlot(AbstractPlot):
    def build(self, data, y_range=None):
        # data is expected to be a full results list.
        v_vals = [r['v_applied'] for r in data]
        JV_vals = [r['current_density']['JV_plot'] for r in data]
        traces = [{"y": JV_vals, "name": "J-V Curve", "mode": "lines+markers"}]
        return PlotlyPlotter.build_figure(
            title="J-V Curve",
            x_data=v_vals,
            traces=traces,
            xaxis_title="Bias (V)",
            yaxis_title="Current Density (A/cm²)"
        )

class DepletionRegionLengthPlot(AbstractPlot):
    def build(self, data, y_range=None):
        # data is expected to be a full results list.
        v_vals = [r['v_applied'] for r in data]
        depletion_vals = [r['depletion_length'] for r in data]
        traces = [{"y": depletion_vals, "name": "Depletion Region Length", "mode": "lines+markers"}]
        return PlotlyPlotter.build_figure(
            title="Depletion Region Length",
            x_data=v_vals,
            traces=traces,
            xaxis_title="Bias (V)",
            yaxis_title="Depletion Length (µm)"
        )

# -----------------------------------------------------------------------------
# Plot Factory
# -----------------------------------------------------------------------------
class PlotFactory:
    """
    The factory class that returns an instance of a concrete plot class
    based on the requested plot type.
    """
    _plot_mapping = {
        "Doping Density": DopingDensityPlot,
        "Carrier Density": CarrierDensityPlot,
        "Net Charge Density": NetChargeDensityPlot,
        "Electric Field": ElectricFieldPlot,
        "Electrostatic Potential": ElectrostaticPotentialPlot,
        "Energy Bands": EnergyBandsPlot,
        "Current Density Distribution": CurrentDensityDistributionPlot,
        "J-V Curve": JVCurvePlot,
        "Depletion Region Length": DepletionRegionLengthPlot
    }
    
    @staticmethod
    def create_plot(plot_type):
        if plot_type in PlotFactory._plot_mapping:
            return PlotFactory._plot_mapping[plot_type]()
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
