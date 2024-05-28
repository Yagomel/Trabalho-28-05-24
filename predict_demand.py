
# Carregando as bibliotecas necessárias
library(tcltk)
library(ggplot2)

# Definindo a semente para a geração de dados aleatórios
set.seed(42)

# Gerando dados sintéticos
temperatura <- runif(100, min = 0, max = 40)  # Temperatura em Celsius
hora_do_dia <- runif(100, min = 0, max = 24)  # Hora do dia em horas
demanda_energia <- (temperatura * 2) + (hora_do_dia * 3) + rnorm(100, sd = 10)

# Criando um data frame
dados <- data.frame(Temperatura = temperatura, Hora_do_Dia = hora_do_dia, Demanda_de_Energia = demanda_energia)

# Treinamento do modelo de regressão linear
modelo <- lm(Demanda_de_Energia ~ Temperatura + Hora_do_Dia, data = dados)

# Previsões do modelo
predicoes <- predict(modelo, newdata = dados)

# Salvando a visualização dos resultados em um arquivo temporário
plot_file <- tempfile(fileext = ".png")
ggplot(dados, aes(x = Demanda_de_Energia, y = predicoes)) +
  geom_point() +
  labs(x = "Demanda Real de Energia", y = "Demanda Prevista de Energia", title = "Previsões de Demanda de Energia") +
  theme_minimal() +
  ggsave(plot_file)

# Criando a interface Tk
root <- tktoplevel()
tkwm.title(root, "Previsões de Demanda de Energia")

# Adicionando o gráfico à interface
img <- tkimage.create("photo", file = plot_file)
lbl <- tklabel(root, image = img)
tkpack(lbl)

# Executando a interface Tk
tk.mainloop()
