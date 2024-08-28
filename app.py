from pathlib import Path

import pandas as pd
import yfinance as yf
from faicons import icon_svg
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import output_widget, render_plotly
from stocks import stocks
from Benchmark import Benchmark
import plotly.graph_objs as go
import yfinance as yf

# Default to the last 6 months
end = pd.Timestamp.now()
start = end - pd.Timedelta(weeks=26)

app_dir = Path(__file__).parent

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize("ticker", "Select Stocks", choices=stocks, selected="AAPL"),
        ui.input_selectize("ticker2", "Select Benchmark", choices=Benchmark, selected="SPY"),
        ui.input_date_range("dates", "Select dates", start=start, end=end),
    ),
    ui.page_navbar(
        ui.nav_panel("Dashboard",
            ui.layout_column_wrap(
                ui.value_box(
                    "Current Price",
                    ui.output_ui("price"),
                    showcase=icon_svg("dollar-sign"),
                    width=1/3
                ),
                ui.value_box(
                    "Change",
                    ui.output_ui("change"),
                    showcase=ui.output_ui("change_icon"),
                    width=1/3
                ),
                ui.value_box(
                    "Percent Change",
                    ui.output_ui("change_percent"),
                    showcase=icon_svg("percent"),
                    width=1/3
                ),
                fill=False,
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Price History"),
                    output_widget("price_history"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header("Latest Data Asset"),
                    ui.output_data_frame("latest_data"),
                    ui.card_header("Latest Data Benchmark"),
                    ui.output_data_frame("latest_data2"),
                ),
                col_widths=[9, 3],
            ),
        ),
        ui.nav_panel("Risk Management",
            ui.layout_column_wrap(
                ui.value_box(
                    "Current Price",
                    ui.output_ui("price_vix"),
                    showcase=icon_svg("dollar-sign"),
                    width=1/3
                ),
                ui.value_box(
                    "Change",
                    ui.output_ui("change_vix"),
                    showcase=ui.output_ui("change_icon_vix"),
                    width=1/3
                ),
                ui.value_box(
                    "Percent Change",
                    ui.output_ui("change_percent_VIX"),
                    showcase=icon_svg("percent"),
                    width=1/3
                ),
                fill=False,
            ),
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("VIX History"),
                    output_widget("vix_history"),
                    full_screen=True,
                ),
            )
        )
    ),
    ui.include_css(app_dir / "styles.css"),
    title="Stock Explorer",
    fillable=True,
)


def server(input: Inputs, output: Outputs, session: Session):
    @reactive.calc
    def get_ticker():
        return yf.Ticker(input.ticker())
    
    @reactive.calc
    def get_ticker2():
        return yf.Ticker(input.ticker2())
    
    @reactive.calc
    def get_ticker3():
        ticker_vix = '^VIX'
        return ticker_vix

    @reactive.calc
    def get_data():
        dates = input.dates()
        return get_ticker().history(start=dates[0], end=dates[1])
    
    @reactive.calc
    def get_data2():
        dates = input.dates()
        return get_ticker2().history(start=dates[0], end=dates[1])
    
    @reactive.calc
    def get_data3():
        dates = input.dates()
        ticker = get_ticker3()  # Renvoie '^VIX'
        vix_data = yf.Ticker(ticker)  # Obtient l'objet ticker
        return vix_data.history(start=dates[0], end=dates[1])
    
    @reactive.calc
    def get_change():
        close = get_data()["Close"]
        return close.iloc[-1] - close.iloc[-2]
    
    @reactive.calc
    def get_change_vix():
        close = get_data3()["Close"]
        return close.iloc[-1] - close.iloc[-2]

    @reactive.calc
    def get_change_percent():
        close = get_data()["Close"]
        change = close.iloc[-1] - close.iloc[-2]
        return change / close.iloc[-2] * 100
    
    @reactive.calc
    def get_change_percent_VIX():
        close = get_data3()["Close"]
        change = close.iloc[-1] - close.iloc[-2]
        return 2

    @render.ui
    def price():
        close = get_data()["Close"]
        return f"{close.iloc[-1]:.2f}"
    
    @render.ui
    def price_vix():
        close = get_data3()["Close"]
        return f"{close.iloc[-1]:.2f}"

    @render.ui
    def change():
        return f"${get_change():.2f}"
    
    @render.ui
    def change_vix():
        return f"${get_change_vix():.2f}"

    @render.ui
    def change_icon():
        change = get_change()
        icon = icon_svg("arrow-up" if change >= 0 else "arrow-down")
        icon.add_class(f"text-{('success' if change >= 0 else 'danger')}")
        return icon
    
    @render.ui
    def change_icon_vix():
        change = get_change_vix()
        icon = icon_svg("arrow-up" if change >= 0 else "arrow-down")
        icon.add_class(f"text-{('success' if change >= 0 else 'danger')}")
        return icon

    @render.ui
    def change_percent():
        return f"{get_change_percent():.2f}%"
    
    @render.ui
    def change_percent_vix():
        return f"{get_change_percent_VIX():.2f}%"
    
    @render_plotly
    def price_history():
        data = get_data()  # Utiliser les données du DataFrame df1
        data_bench = get_data2()
        close_prices = data["Close"]  # Extraire les prix de clôture
        close_bench = data_bench["Close"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=close_prices.index,  # Dates
            y=close_prices,  # Prix de clôture
            mode='lines',  # Mode lignes
            line=dict(color='blue'),  # Couleur bleue pour la ligne
            name='Close Price',
            yaxis='y1' # Nom de la série
        ))
        fig.add_trace(go.Scatter(
            x=close_bench.index,  # Dates
            y=close_bench,  # Volume
            mode='lines',                 # Mode lignes
            line=dict(color='red'),       # Couleur rouge pour la ligne
            name='Close Price Benchmark',   # Nom de la série
            yaxis='y2'  
        ))    
        fig.update_layout(
            title='Price and Volume History',
            xaxis_title='Date',
            yaxis_title='Close Price (USD)',
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right'
            ),
            # plot_bgcolor='white',  # Fond blanc pour le graphique
            # paper_bgcolor='white',  # Fond blanc pour le papier
            # barmode='overlay'  # Mode de superposition pour les barres
        )
        return fig

    @render_plotly
    def vix_history():
        data = get_data3()  # Utiliser les données du DataFrame df1
        close_vix = data["Close"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=close_vix.index,  # Dates
            y=close_vix,  # Prix de clôture
            mode='lines',  # Mode lignes
            line=dict(color='green'),  # Couleur verte pour la ligne
            name='VIX Level',
        ))
        fig.update_layout(
            title='VIX History',
            xaxis_title='Date',
            yaxis_title='VIX Level',
            # plot_bgcolor='white',  # Fond blanc pour le graphique
            # paper_bgcolor='white'  # Fond blanc pour le papier
        )
        return fig

    @render.data_frame
    def latest_data():
        x = get_data()[:1].T.reset_index()
        x.columns = ["Category", "Value"]
        x["Value"] = x["Value"].apply(lambda v: f"{v:.1f}")
        return x

    @render.data_frame
    def latest_data2():
        x = get_data2()[:1].T.reset_index()
        x.columns = ["Category", "Value"]
        x["Value"] = x["Value"].apply(lambda v: f"{v:.1f}")
        return x

app = App(app_ui, server)

# Exécuter l'application
if __name__ == "__main__":
    app.run()
