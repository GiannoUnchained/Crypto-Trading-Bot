# Simple Crypto Trading Bot

Ein einfacher Paper-Trading-Bot für Ethereum (ETH), der über die CoinGecko-API den aktuellen ETH/USD-Preis abfragt und fiktive Trades durchführt.

## Funktionen

- Regelmäßige Abfrage des ETH/USD-Preises über CoinGecko
- Einfache Trading-Strategie (1% Preisänderung)
- Portfolio-Simulation
- CSV-Logging aller Trades

## Installation

1. Erstelle eine virtuelle Umgebung:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

2. Installiere die Abhängigkeiten:
```bash
pip install -r requirements.txt
```

## Verwendung

1. Starte den Bot:
```bash
python3 trading_bot.py
```

2. Starte das Dashboard:
```bash
streamlit run dashboard.py
```

Der Bot läuft dann im Hintergrund und führt regelmäßig Price-Checks und Trades durch. Alle Trades werden in der Datei `trade_history.csv` protokolliert und im Dashboard angezeigt.

## Portfolio

Der Bot startet mit:
- 1000 USD
- 0 ETH

## Logging

Alle Trades werden in der `trade_history.csv`-Datei protokolliert mit folgenden Informationen:
- Timestamp
- Aktion (buy/sell)
- Preis
- Menge
- USD-Balance
- ETH-Balance

## Beenden

Drücke `Ctrl+C` um den Bot zu beenden.
