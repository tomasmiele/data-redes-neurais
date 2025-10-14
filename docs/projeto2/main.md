# Projeto 2 - Regression - Tomas Miele e Yuri Tabacof

## 1. Dataset Selection

**Nome do dataset:** S&P 500 Historical Data — Stock Market Index Prediction  
**Fonte:** Yahoo Finance / Kaggle  
**URL:** https://www.kaggle.com/datasets/whenamancodes/sp-500-stock-data  
**Tamanho:** ~15.000 registros diários, 7 variáveis (Open, High, Low, Close, Adj Close, Volume, Date)  

**Tarefa:** Regressão — prever o **preço de fechamento (Close)** do índice S&P 500 com base em variáveis históricas de mercado.  

**Justificativa da escolha:**  
- O problema é **financeiro e prático**, ligado à **previsão de preços** — uma aplicação clássica de regressão contínua.  
- Os dados representam **séries temporais reais** com ruído, tendência e sazonalidade, o que desafia o modelo e permite testar **preprocessing, regularização e tuning do MLP**.  
- Contém **variáveis correlacionadas e contínuas** (preço de abertura, volume, máximas e mínimas), adequadas para exploração de **relações não lineares** via redes neurais.  
- O volume de dados é **suficiente** (>1.000 amostras) e com múltiplos atributos (>5), atendendo integralmente aos critérios do projeto.  
- Permite ainda incorporar **engenharia de features financeiras**, como retornos logarítmicos, médias móveis e volatilidade, tornando o problema mais robusto e próximo de aplicações reais de *quantitative finance*.  
- A base é **pública e reproduzível**, podendo ser facilmente obtida via API do Yahoo Finance (`yfinance`) ou baixada do Kaggle, sem restrições de uso.  

