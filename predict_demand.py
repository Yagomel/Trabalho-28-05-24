import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score # 
import matplotlib.pyplot as plt 


temperatura = np.random.rand(100, 1) * 40  # Temperatura em Celsius
hora_do_dia = np.random.rand(100, 1) * 24  # Hora do dia em horas
demanda_energia = (temperatura * 2) + (hora_do_dia * 3) + np.random.randn(100, 1) * 10

dados = pd.DataFrame(np.concatenate([temperatura, hora_do_dia, demanda_energia], axis=1), columns=['Temperatura', 'Hora do Dia', 'Demanda de Energia'])

X = dados[['Temperatura', 'Hora do Dia']]
y = dados['Demanda de Energia']
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.4, random_state=42)


modelo = LinearRegression()
modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)

erro_medio_quadratico = mean_squared_error(y_teste, previsoes)
coeficiente_de_determinacao = r2_score(y_teste, previsoes)
print("Erro Médio Quadrático:", erro_medio_quadratico)
print("Coeficiente de Determinação (R²):", coeficiente_de_determinacao)

plt.scatter(y_teste, previsoes)
plt.plot([y_teste.min(), y_teste.max()], [y_teste.min(), y_teste.max()], 'k--', lw=2)  # Linha de tendência
plt.xlabel('Demanda Real de Energia')
plt.ylabel('Demanda Prevista de Energia')
plt.title('Previsões de Demanda de Energia')
plt.show()
