import pandas as pd # Utilizada para manipulação e análise de dados, principalmente em tabelas (DataFrames).
import numpy as np # Biblioteca para operações matemáticas e manipulação de arrays numéricos.
import matplotlib.pyplot as plt # Usada para criar gráficos.
from sklearn.model_selection import train_test_split # Função da "sklearn" para dividir os dados em conjuntos de treino e teste.
from sklearn.linear_model import LinearRegression # Classe para criar um modelo de regressão linear.
from sklearn.metrics import mean_absolute_error, mean_squared_error # Funções que calculam métricas de erro para avaliar a precisão do modelo.

# Exemplo: Dados simulados para vendas
data = {
    'data': pd.date_range(start='2023-01-01', periods=100, freq='D'), # Cria uma série de **100** datas diárias, começando em 1 de janeiro de 2023.
    'vendas': np.random.randint(20, 100, size=100) + np.linspace(1, 50, 100) # (np.random.randint) Gera uma lista de **100** números inteiros aleatórios entre 20 e 100, que simula valores de vendas.
} # (np.linspace) Cria uma sequência de **100** números igualmente espaçados de 1 a 50, para adicionar uma tendência de crescimento nas vendas.
df = pd.DataFrame(data) # Cria um DataFrame "df" com as colunas data e vendas.

# Preprocessamento dos dados
df['dias'] = (df['data'] - df['data'].min()).dt.days # Calcula o número de dias desde a primeira data. Isso transforma a data em um número inteiro, facilitando o uso no modelo de regressão.
X = df[['dias']]
y = df['vendas'] # Define a variável dependente "y", que é o que queremos prever (vendas).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Divide X e y em conjuntos de treino e teste. Aqui, 20% dos dados são reservados para teste (test_size=0.2), e 80% são usados para treino.

# Treinamento do modelo
modelo = LinearRegression() # Cria uma instância do modelo de regressão linear.
modelo.fit(X_train, y_train) # Treina o modelo usando os dados de treinamento (X_train e y_train).

# Avaliação do modelo
previsoes = modelo.predict(X_test) # Usa o modelo treinado para fazer previsões nas variáveis "X_test".
mae = mean_absolute_error(y_test, previsoes) # Calcula o Erro Médio Absoluto (MAE), que é a média das diferenças absolutas entre previsões e valores reais.
mse = mean_squared_error(y_test, previsoes) # Calcula o Erro Quadrático Médio (MSE), que é a média dos erros elevados ao quadrado, penalizando erros maiores.
rmse = np.sqrt(mse) #Calcula a Raiz do Erro Quadrático Médio (RMSE), que traz o MSE para a mesma escala dos dados originais.

print(f"Erro Médio Absoluto (MAE): {mae}")
print(f"Erro Quadrático Médio (MSE): {mse}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse}")

dias_futuros = np.array([[i] for i in range(df['dias'].max() + 1, df['dias'].max() + 31)])
previsoes_futuras = modelo.predict(dias_futuros)

# Visualizar os dados reais e previsões
plt.plot(df['dias'], df['vendas'], label='Dados reais')
plt.scatter(X_test, previsoes, color='red', label='Previsões no conjunto de teste')
plt.xlabel('Dias')
plt.ylabel('Vendas')
plt.legend()
plt.show()

modelo = LinearRegression()
modelo.fit(X_train, y_train)
